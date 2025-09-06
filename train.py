import torch
import os
import time
import math
import logging
import traceback
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.metrics import  r2_score, accuracy_score, roc_auc_score, f1_score, mean_absolute_error,confusion_matrix,precision_score,recall_score,mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR, OneCycleLR
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
from rdkit.Chem import Draw
import svgwrite
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from scipy import stats
import pandas as pd
from TwistDAN_data_preprocess import dataload,create_optimized_dataloaders
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import argparse
from sklearn.preprocessing import KBinsDiscretizer
import random
import json
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from TwistDAN import TwistDAN
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Dict, Any
import os


os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")




def optimize_cuda_performance():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("CUDA performance optimizations applied")

def check_cuda_compatibility():
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        torch_version = torch.__version__
        print(f"CUDA version: {cuda_version}")
        print(f"PyTorch version: {torch_version}")
        major_cuda = int(cuda_version.split('.')[0])
        if torch_version.startswith('2.0') and major_cuda < 11:
            print("WARNING: PyTorch 2.0+ requires CUDA 11.0+")
        elif torch_version.startswith('1.') and major_cuda < 10:
            print("WARNING: PyTorch 1.x typically requires CUDA 10.0+")

def ensure_model_on_device(model, device):
    if device.type == 'cuda':
        for param in model.parameters():
            if not param.is_cuda:
                print(f"Warning: Found parameter not on CUDA, moving it now")
                param.data = param.data.to(device)
    return model



import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

#########################################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description='GTSA Model Training')
    parser.add_argument('-n', '--num-epochs', type=int, default=200)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
    parser.add_argument('-p', '--result-path', type=str, default='results')
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('--hp-search', action='store_true', help='Run hyperparameter search')
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    

def assign_bin(value, bins):
    return np.digitize(value, bins) - 1

class FocalLoss(nn.Module):
    def __init__(self, beta=0.9, gamma=2.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
    def forward(self, pred, target, class_counts=None):
        if class_counts is None:
            class_counts = torch.bincount(target.long())

        effective_num = 1.0 - torch.pow(self.beta, class_counts)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / torch.sum(weights) * len(class_counts)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        sample_weights = weights[target.long()]
        focal_loss = sample_weights * (1 - pt)**self.gamma * bce_loss
        
        return focal_loss.mean()




def augment_data(data, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    if random.random() < 0.4:
        noise = torch.randn_like(data.x) * 0.1
        data.x = data.x + noise

    if random.random() < 0.3:
        edge_mask = torch.rand(data.edge_index.size(1)) > 0.15
        data.edge_index = data.edge_index[:, edge_mask]

    if random.random() < 0.2:
        mask = torch.rand(data.x.size()) > 0.1
        data.x = data.x * mask.float()
    return data

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

##############################################################################################################

def compute_metrics(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred > threshold).astype(int)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary),
        "recall": recall_score(y_true, y_pred_binary),
        "f1": f1_score(y_true, y_pred_binary),
        "roc_auc": roc_auc_score(y_true, y_pred),
        "pr_auc": pr_auc,  
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred)
    }
    
    return metrics

def format_metrics(metrics, prefix=""):
    return (f"{prefix} "
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}, "
            f"ROC-AUC: {metrics['roc_auc']:.4f}, "
            f"PR-AUC: {metrics['pr_auc']:.4f}, " 
            f"Sensitivity: {metrics['sensitivity']:.4f}, "
            f"Specificity: {metrics['specificity']:.4f}, "
            f"MSE: {metrics['mse']:.4f}, "
            f"MAE: {metrics['mae']:.4f}, "
            f"RMSE: {metrics['rmse']:.4f}, "
            f"R2: {metrics['r2']:.4f}")
#################################################################################


def create_ensemble_model(fold_results, num_node_features, num_edge_features, num_node_types, num_edge_types, device_manager):
    logging.info(f"Starting ensemble model creation with {len(fold_results)} fold results")
    ensemble_models = []
    if not fold_results:
        raise ValueError("Fold results is empty")
    
    for fold_idx, (model_path, val_loss) in enumerate(fold_results):
        try:
            logging.info(f"Processing fold {fold_idx + 1}/{len(fold_results)}")
            logging.info(f"Loading model from path: {model_path}")
            if not os.path.exists(model_path):
                logging.error(f"Model file not found: {model_path}")
                continue

            model = TwistDAN(
                in_dim=num_node_features,
                hidden_dim=256,
                num_layers=6,
                num_heads=8,
                dropout=0.2,
                num_classes=1,
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
                processing_steps=4
            ).to(device_manager.device)

            try:
                checkpoint = torch.load(model_path)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                model.load_state_dict(state_dict)
                logging.info(f"Successfully loaded state dict for fold {fold_idx + 1}")
            except Exception as e:
                logging.error(f"Error loading state dict for fold {fold_idx + 1}: {str(e)}")
                continue
            
            model = model.to(device_manager.device)
            model.eval()
            try:
                test_x = torch.randn(2, num_node_features).to(device_manager.device)
                test_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(device_manager.device)
                test_batch = torch.tensor([0, 0], dtype=torch.long).to(device_manager.device)
                with torch.no_grad():
                    test_data = Data(x=test_x, edge_index=test_edge_index, batch=test_batch)
                    test_output = model(test_data)
                    
                if test_output is not None:
                    ensemble_models.append(model)
                    logging.info(f"Successfully verified model for fold {fold_idx + 1}")
                else:
                    logging.error(f"Model verification failed for fold {fold_idx + 1}: null output")
                    
            except Exception as e:
                logging.error(f"Error verifying model for fold {fold_idx + 1}: {str(e)}")
                continue
                
        except Exception as e:
            logging.error(f"Error processing fold {fold_idx + 1}: {str(e)}")
            continue
    
    if not ensemble_models:
        raise ValueError("No models could be loaded for ensemble. Check the logs for details.")
    
    logging.info(f"Successfully created ensemble with {len(ensemble_models)} models")
    return EnsembleModel(ensemble_models)


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        logging.info(f"Initialized EnsembleModel with {len(models)} models")
    
    def forward(self, data):
        outputs = []
        for idx, model in enumerate(self.models):
            try:
                model.eval()
                with torch.no_grad():
                    output = model(data)
                    
                if output is not None:
                    outputs.append(output)
                else:
                    logging.warning(f"Model {idx} produced null output")
                    
            except Exception as e:
                logging.error(f"Error in model {idx} prediction: {str(e)}")
                continue
        
        if not outputs:
            raise RuntimeError("No valid outputs from any ensemble model")
        stacked_outputs = torch.stack(outputs)
        return torch.mean(stacked_outputs, dim=0)
    
class DeviceManager:
    def __init__(self, device=None):
        self.device = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        print(f"DeviceManager initialized with device: {self.device}")

    def prepare_model(self, model):
        return model.to(self.device)

    def prepare_batch(self, batch):
        return process_batch_to_device(batch, self.device)

    def prepare_optimizer(self, optimizer):
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        return optimizer



def process_batch_to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [b.to(device) if hasattr(b, 'to') else b for b in batch]

    return batch.to(device)


def debug_batch(batch, prefix=""):
    print(f"\n{prefix} Batch Debug Information:")
    print("-" * 50)

    if isinstance(batch, tuple):
        print(" Batch contains both graph and SMILES data")
        graph_data, smiles_data = batch
        print(f"SMILES data present: {len(smiles_data)} sequences")
        print("\nGraph Data:")
        _debug_graph_data(graph_data)
        print("\nSMILES Data Sample:")
        print(smiles_data[:3] if len(smiles_data) > 3 else smiles_data)
    else:
        print(" Batch contains only graph data (no SMILES)")
        _debug_graph_data(batch)

def _debug_graph_data(data):
    print(f"Data type: {type(data)}")
    
    if hasattr(data, 'x'):
        print(f"Node features (x):")
        print(f"  Shape: {data.x.shape}")
        print(f"  Device: {data.x.device}")
        print(f"  Type: {data.x.dtype}")
    
    if hasattr(data, 'edge_index'):
        print(f"Edge index:")
        print(f"  Shape: {data.edge_index.shape}")
        print(f"  Device: {data.edge_index.device}")
    
    if hasattr(data, 'y'):
        print(f"Labels (y):")
        print(f"  Shape: {data.y.shape}")
        print(f"  Device: {data.y.device}")

def check_smiles_usage(model, loader=None):
    print("\nChecking SMILES Usage:")
    print("-" * 50)
    print("\n1. Checking Model Configuration:")
    has_smiles_processor = hasattr(model, 'smiles_processor')
    has_cross_attention = hasattr(model, 'cross_attention')
    print(f"Model has SMILES processor: {'Yes' if has_smiles_processor else 'No'}")
    print(f"Model has cross attention: {'Yes' if has_cross_attention else 'No'}")
    has_smiles = False
    if loader is not None:
        print("\n2. Checking Dataloader:")
        try:
            first_batch = next(iter(loader))
            has_smiles = isinstance(first_batch, tuple) and len(first_batch) == 2
            print(f"Dataloader provides SMILES data: {'Yes' if has_smiles else 'No'}")
        except Exception as e:
            print(f"Error checking dataloader: {str(e)}")
            print("Assuming no SMILES data in dataloader")
    else:
        print("\n2. Dataloader Check: Skipped (no loader provided)")

    print("\n3. Model Architecture:")
    for name, module in model.named_children():
        print(f"  - {name}: {type(module).__name__}")
    
    return {
        'model_has_processor': has_smiles_processor,
        'model_has_attention': has_cross_attention,
        'dataloader_has_smiles': has_smiles if loader is not None else None
    }


def train_and_evaluate(model, loaders, criterion, optimizer, scheduler, num_epochs, device):
    best_val_loss = float('inf')
    best_model = None
    early_stopping = EarlyStopping(patience=10)
    aux_criterion = nn.BCEWithLogitsLoss()
    aux_weight = 0.2
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        logging.info(f"Starting training epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(tqdm(loaders['train'], desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            try:
                batch = batch.to(device)
                optimizer.zero_grad()
                outputs = model(batch)
                if isinstance(outputs, tuple):
                    if len(outputs) == 2:
                        class_part, aux_output = outputs
                        if isinstance(class_part, tuple):
                            class_output = class_part[0]
                        else:
                            class_output = class_part
                    else:
                        class_output = outputs[0]
                        aux_output = torch.zeros(batch.batch.max().item() + 1, 1).to(device)
                else:
                    class_output = outputs
                    aux_output = torch.zeros(batch.batch.max().item() + 1, 1).to(device)
                targets = batch.y.float().view(-1)
                class_loss = nn.BCEWithLogitsLoss()(class_output, targets)
                batch_size = batch.batch.max().item() + 1
                aux_target = torch.zeros(batch_size, 1).to(device)
                aux_loss = aux_criterion(aux_output, aux_target)
                loss = class_loss + aux_weight * aux_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_preds.extend(class_output.detach().cpu().numpy())
                train_labels.extend(targets.cpu().numpy())
                if batch_idx % 100 == 0:
                    logging.info(f"  Batch {batch_idx}/{len(loaders['train'])}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                logging.error(f"Error in training batch {batch_idx}: {str(e)}")
                traceback.print_exc()
                continue
        
        avg_train_loss = train_loss / len(loaders['train']) if train_loss > 0 else float('inf')
        train_metrics = compute_metrics(np.array(train_labels), np.array(train_preds))
        logging.info(f"Training metrics: {format_metrics(train_metrics, 'Train')}")
        if scheduler is not None:
            scheduler.step()
            logging.info(f"Learning rate updated to: {optimizer.param_groups[0]['lr']:.6f}")

        logging.info(f"Starting validation after epoch {epoch+1}")
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for val_batch_idx, batch in enumerate(tqdm(loaders['val'], desc="Validation")):
                try:
                    batch = batch.to(device)
                    outputs = model(batch, head_idx=0)
                    if isinstance(outputs, tuple):
                        class_output = outputs[0]
                    else:
                        class_output = outputs
                    if isinstance(class_output, tuple):
                        logging.warning(f"Warning: Nested tuple detected in class_output")
                        class_output = class_output[0]
                        
                    if not torch.is_tensor(class_output):
                        logging.warning(f"Warning: class_output is not a tensor: {type(class_output)}")
                        continue
                        
                    targets = batch.y.float().view(-1)
                    loss = criterion(class_output, targets)
                    val_loss += loss.item()
                    val_preds.extend(class_output.detach().cpu().numpy())
                    val_labels.extend(targets.cpu().numpy())

                    if val_batch_idx % 50 == 0:
                        logging.info(f"  Validation Batch {val_batch_idx}/{len(loaders['val'])}, Loss: {loss.item():.4f}")
                                
                except Exception as e:
                    logging.error(f"Error in validation batch {val_batch_idx}: {str(e)}")
                    traceback.print_exc()
                    continue
        avg_val_loss = val_loss / len(loaders['val']) if val_loss > 0 else float('inf')
        val_metrics = compute_metrics(np.array(val_labels), np.array(val_preds))
        logging.info(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        logging.info(f"  Train Loss: {avg_train_loss:.4f}")
        logging.info(f"  Validation Loss: {avg_val_loss:.4f}")
        logging.info("  Training: " + format_metrics(train_metrics, "Train"))
        logging.info("  Validation: " + format_metrics(val_metrics, "Valid"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = {k: v.cpu() for k, v in model.state_dict().items()}
            logging.info(f"\nNew best model saved! (val_loss: {best_val_loss:.4f})")

        if early_stopping(avg_val_loss):
            logging.info("Early stopping triggered")
            break
    
    return best_model, best_val_loss, val_metrics

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, desc="Testing", leave=False)):
            try:
                data = data.to(DEVICE)
                outputs = model(data)
                if isinstance(outputs, tuple) and len(outputs) > 0:
                    output = outputs[0]  # Use only main output
                else:
                    output = outputs
                loss = criterion(output.view(-1), data.y.float())
                test_loss += loss.item()
                
                preds = torch.sigmoid(output).cpu().numpy().flatten()
                labels = data.y.cpu().numpy().flatten()
                
                test_preds.extend(preds)
                test_labels.extend(labels)
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    if not test_preds:
        raise ValueError("No predictions were generated during testing")
    
    test_loss /= len(test_loader)
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_metrics = compute_metrics(test_labels, test_preds)
    return test_loss, test_metrics, test_preds, test_labels

def evaluate_multiple_test_sets(model, test_loaders, criterion):
    results = {}
    for name, loader in test_loaders.items():
        test_loss, test_metrics, test_pred, test_label = test_model(model, loader, criterion)
        results[name] = {
            'loss': test_loss,
            'metrics': test_metrics,
            'predictions': test_pred,
            'labels': test_label
        }
        print(f"\nResults for {name}:")
        print(f'Loss: {test_loss:.6f}')
        print(format_metrics(test_metrics, name))
    return results


###########################################################################################################################

def analyze_target_distribution(dataset):
    y_values = [data.y.item() for data in dataset]
    mean = np.mean(y_values)
    median = np.median(y_values)
    std = np.std(y_values)
    skewness = stats.skew(y_values)
    kurtosis = stats.kurtosis(y_values)
    print(f"Target Distribution Statistics:")
    print(f"Mean: {mean:.4f}")
    print(f"Median: {median:.4f}")
    print(f"Standard Deviation: {std:.4f}")
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
    plt.figure(figsize=(10, 6))
    plt.hist(y_values, bins=50, edgecolor='black')
    plt.title("Distribution of Target Values")
    plt.xlabel("Target Value")
    plt.ylabel("Frequency")
    plt.savefig("target_distribution.png")
    plt.close()
    return y_values


##################################################################################################


def extract_dataset(dataloader):
    if dataloader is None:
        return None
    return dataloader.dataset

def create_optimizer_and_scheduler(model, train_loader, num_epochs):
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.003,  # Peak learning rate
        total_steps=total_steps,  
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=10.0,  # Initial lr = max_lr/10
        final_div_factor=1000.0,
        verbose=True
    )

    return optimizer, scheduler


def setup_cuda_device():
    global DEVICE
    print("\nChecking CUDA Setup...")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Checking possible issues:")
        print(f"\n1. PyTorch Installation:")
        print(f"   - PyTorch version: {torch.__version__}")
        
        try:
            from torch.backends import cudnn
            print(f"   - CUDA enabled in PyTorch build: {torch.backends.cudnn.enabled}")
        except ImportError:
            print("   - PyTorch may not be built with CUDA support")
        try:
            import subprocess
            nvidia_smi = subprocess.check_output("nvidia-smi", shell=True)
            print("\n2. NVIDIA Driver:")
            print(f"   - NVIDIA SMI available: Yes")
            print(f"   - Driver output: {nvidia_smi.decode().split('n')[0]}")
        except Exception as e:
            print("\n2. NVIDIA Driver:")
            print(f"   - Error running nvidia-smi: {str(e)}")
            print(f"   - Please check if NVIDIA drivers are installed")
        
        # Check environment variables
        print("\n3. Environment Variables:")
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
        print(f"   - CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
        DEVICE = torch.device('cpu')
        return DEVICE
    
    try:
        gpu_id = 0
        if torch.cuda.device_count() > 1:
            free_memory = []
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                free_memory.append(torch.cuda.memory_reserved())
            gpu_id = free_memory.index(min(free_memory))
        
        DEVICE = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(DEVICE)
        print("\nCUDA Setup Successful:")
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"GPU ID: {gpu_id}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA Version: {torch.backends.cudnn.version()}")
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"\nMemory Status:")
        print(f"Total GPU Memory: {total_memory:.2f} GB")
        print(f"Reserved Memory: {reserved_memory:.2f} GB")
        print(f"Allocated Memory: {allocated_memory:.2f} GB")
        print(f"Free Memory: {(total_memory - reserved_memory):.2f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        return DEVICE
    
    except Exception as e:
        print(f"\nError during CUDA setup: {str(e)}")
        print("Falling back to CPU")
        DEVICE = torch.device('cpu')
        return DEVICE

def initialize_training_device():
    global DEVICE 
    DEVICE = setup_cuda_device()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        torch.set_grad_enabled(True)
        if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') is None:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        print("\nCUDA Memory Management:")
        print("- Enabled gradient computation")
        print("- Cleared CUDA cache")
        print("- Set memory allocation configuration")
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("- Enabled TF32 for Ampere+ GPUs")
    else:
        print("\nUsing CPU for training (CUDA not available)")
        print("Note: Training will be significantly slower on CPU")
    return DEVICE

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, pred, target):
        if isinstance(pred, tuple):
            pred = pred[0]
            
        bce_loss = self.bce(pred, target)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return (bce_loss + focal_loss).mean()


def optimize_memory():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8) 
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

def create_optimized_model_config():
    return {
        'batch_size': 256,  
        'learning_rate': 3e-4,  
        'weight_decay': 0.01,
        'warmup_steps': 100,
        'gradient_accumulation_steps': 4,
        'mixed_precision': True,
        'num_workers': 4,
        'pin_memory': True
    }

#############################################################

def display_metrics_summary(results, all_predictions, all_labels):
    metrics_data = []
    for dataset_name, result in results.items():
        metrics = result['metrics']
        loss = result['loss']
    
        metrics_data.append({
            'Dataset': dataset_name,
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1-score': f"{metrics['f1']:.3f}",
            'ROC-AUC': f"{metrics['roc_auc']:.3f}",
            'PR-AUC': f"{metrics['pr_auc']:.3f}",  # Added PR-AUC
            'Sensitivity': f"{metrics['sensitivity']:.3f}",
            'Specificity': f"{metrics['specificity']:.3f}",
            'R2': f"{metrics['r2']:.3f}",
            'Loss': f"{loss:.3f}"
        })
    
    df = pd.DataFrame(metrics_data)

    print("Confusion Matrices:")
    print("=" * 50)
    for dataset_name in results.keys():
        if dataset_name in all_predictions and dataset_name in all_labels:
            pred_labels = (np.array(all_predictions[dataset_name]) > 0.5).astype(int)
            true_labels = np.array(all_labels[dataset_name])
            cm = confusion_matrix(true_labels, pred_labels)
            print(f"\n{dataset_name} Confusion Matrix:")
            print(cm)
            print("-" * 50)
    print("\nMetrics Summary:")
    print("=" * 150)  
    header = "Dataset    "
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "R2", "Loss"]
    for metric in metrics:
        header += f"{metric:<12}"
    print(header)
    print("-" * 150) 
    for _, row in df.iterrows():
        line = f"{row['Dataset']:<10}"
        for metric in metrics:
            line += f"{row[metric]:<12}"
        print(line)
    print("=" * 150) 
    return df

def save_results(results, all_predictions, all_labels, result_path):
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        metrics_df = display_metrics_summary(results, all_predictions, all_labels)
        metrics_df.to_csv(os.path.join(result_path, 'metrics_summary.csv'), index=False)
        with open(os.path.join(result_path, 'metrics_summary.txt'), 'w') as f:
            f.write("Confusion Matrices:\n")
            f.write("=" * 50 + "\n")
            for dataset_name in results.keys():
                if dataset_name in all_predictions and dataset_name in all_labels:
                    pred_labels = (np.array(all_predictions[dataset_name]) > 0.5).astype(int)
                    true_labels = np.array(all_labels[dataset_name])
                    cm = confusion_matrix(true_labels, pred_labels)
                    
                    f.write(f"\n{dataset_name} Confusion Matrix:\n")
                    f.write(str(cm) + "\n")
                    f.write("-" * 50 + "\n")
            
            f.write("\nMetrics Summary:\n")
            f.write("=" * 130 + "\n")
            metrics_df.to_string(f, index=False)
            f.write("\n" + "=" * 130 + "\n")
        plt.figure(figsize=(20, 20), dpi=300)
        plt.subplot(2, 2, 1)
        for i, (name, preds) in enumerate(all_predictions.items()):
            if name in all_labels:
                plt.subplot(2, len(all_predictions), i + 1)
                pred_labels = (np.array(preds) > 0.5).astype(int)
                cm = confusion_matrix(all_labels[name], pred_labels)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
        
        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(result_path, 'confusion_matrices.png'), 
                    bbox_inches='tight', 
                    pad_inches=0.5,
                    dpi=300)
        plt.close()
        plt.figure(figsize=(20, 10), dpi=300)
        plt.subplot(1, 2, 1)
        for name, preds in all_predictions.items():
            if name in all_labels:
                exact_roc_auc = results[name]['metrics']['roc_auc']
                fpr, tpr, _ = roc_curve(all_labels[name], preds)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {exact_roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.subplot(1, 2, 2)
        for name, preds in all_predictions.items():
            if name in all_labels:
                exact_pr_auc = results[name]['metrics']['pr_auc']
                precision, recall, _ = precision_recall_curve(all_labels[name], preds)
                plt.plot(recall, precision, label=f'{name} (PR-AUC = {exact_pr_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='lower left')
        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(result_path, 'performance_curves.png'),
                    bbox_inches='tight',
                    pad_inches=0.5,
                    dpi=300)
        plt.close()
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'PR-AUC']
        plot_data = metrics_df[['Dataset'] + metrics_to_plot].copy()
        plot_data[metrics_to_plot] = plot_data[metrics_to_plot].astype(float)
        plt.figure(figsize=(15, 8), dpi=300)
        x = np.arange(len(plot_data['Dataset']))
        width = 0.15
        for i, metric in enumerate(metrics_to_plot):
            plt.bar(x + i*width, plot_data[metric], width, label=metric)

        plt.xlabel('Datasets')
        plt.ylabel('Score')
        plt.title('Comparison of Metrics Across Datasets')
        plt.xticks(x + width*2.5, plot_data['Dataset'])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(result_path, 'metrics_comparison.png'),
                    bbox_inches='tight',
                    pad_inches=0.5,
                    dpi=300)
        plt.close()
        save_curve_coordinates(all_predictions, all_labels, results, result_path)
        
    except Exception as e:
        logging.error(f"Error in save_results: {str(e)}")
        traceback.print_exc()

def save_model(model, optimizer, val_loss, config, result_path, dataset_mapping, name='model'):
    os.makedirs(result_path, exist_ok=True)

    model_save_path = os.path.join(result_path, f'{name}_checkpoint.pth')
    if isinstance(model, dict):
        model_state = model
    else:
        model_state = model.state_dict()
    optimizer_state = None
    if optimizer is not None:
        if isinstance(optimizer, dict):
            optimizer_state = optimizer
        else:
            optimizer_state = optimizer.state_dict()
    torch.save({
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'val_loss': val_loss,
        'model_config': config,
        'dataset': name,
        'dataset_mapping': dataset_mapping,  
        'timestamp': time.strftime("%Y%m%d_%H%M%S")
    }, model_save_path)
    metadata_path = os.path.join(result_path, f'{name}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            'dataset': name,
            'dataset_mapping': dataset_mapping,
            'timestamp': time.strftime("%Y%m%d_%H%M%S"),
            'val_loss': val_loss,
            'model_config': {k: str(v) for k, v in config.items()} 
        }, f, indent=4)
    
    logging.info(f"Model saved to {model_save_path} with dataset mapping metadata")
    return model_save_path

def run_consistent_evaluation(model, test_loaders, criterion, optimizer, model_config, result_path):
    set_seed(42)
    results = {}
    all_predictions = {}
    all_labels = {}
    if isinstance(model, dict):
        loaded_model = TwistDAN(**model_config).to(DEVICE)
        loaded_model.load_state_dict(model)
        model = loaded_model
    
    model.eval()
    with torch.no_grad():
        for name, loader in test_loaders.items():
            if loader is None or len(loader) == 0:
                logging.warning(f"Skipping {name} dataset (empty loader)")
                continue
                
            logging.info(f"Evaluating on {name} dataset...")
            
            try:
                set_seed(42)
                test_loss, metrics, predictions, labels = test_model(
                    model=model,
                    test_loader=loader,
                    criterion=criterion)

                results[name] = {
                    'loss': test_loss,
                    'metrics': metrics}
                all_predictions[name] = predictions
                all_labels[name] = labels
                save_model(
                    model=model,
                    optimizer=optimizer,
                    val_loss=test_loss,
                    config=model_config,
                    result_path=result_path,
                    name=name)

                logging.info(f"Results for {name}:")
                logging.info(f"Loss: {test_loss:.4f}")
                logging.info(format_metrics(metrics, name))

            except Exception as e:
                logging.error(f"Error evaluating {name}: {str(e)}")
                continue
    
    save_results(results, all_predictions, all_labels, result_path)
    return results



def setup_training(args, num_node_features, num_edge_features, num_node_types, num_edge_types, device, train_loader):
    model = TwistDAN(
    in_dim=num_node_features,
    hidden_dim=128, 
    num_layers=4,
    num_heads=4,
    dropout=0,    
    num_classes=1,
    num_node_types=num_node_types,
    num_edge_types=num_edge_types,
    processing_steps=3  
).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999))
    criterion = CombinedLoss(alpha=0.25, gamma=2.0)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.num_epochs
    scheduler = OneCycleLR(
    optimizer,
    max_lr=5e-4,    
    total_steps=total_steps,
    pct_start=0.3,  
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
    return model, optimizer, criterion, scheduler

###############################################################
def plot_and_save_curves(all_predictions, all_labels, result_path):
    curves_dir = os.path.join(result_path, 'curves')
    os.makedirs(curves_dir, exist_ok=True)
    curve_data = {}
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.figsize'] = (20, 8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    colors = ['#1f77b4', '#17becf', '#2ca02c', '#ff7f0e']
    for idx, (dataset_name, predictions) in enumerate(all_predictions.items()):
        if dataset_name not in all_labels:
            continue
            
        labels = all_labels[dataset_name]
        fpr, tpr, thresholds_roc = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        curve_data[dataset_name] = {
            'roc': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds_roc.tolist(),
                'auc': float(roc_auc)}}
        
        ax1.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2, 
                label=f'{dataset_name} (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    for idx, (dataset_name, predictions) in enumerate(all_predictions.items()):
        if dataset_name not in all_labels:
            continue
            
        labels = all_labels[dataset_name]
        precision, recall, thresholds_pr = precision_recall_curve(labels, predictions)
        pr_auc = average_precision_score(labels, predictions)
        if dataset_name in curve_data:
            curve_data[dataset_name]['pr'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds_pr.tolist() if len(thresholds_pr) == len(precision) else 
                             thresholds_pr.tolist() + [float('nan')],
                'auc': float(pr_auc)
            }
        
        ax2.plot(recall, precision, color=colors[idx % len(colors)], lw=2,
                label=f'{dataset_name} (PR-AUC = {pr_auc:.3f})')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.4, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower left')
    plt.tight_layout()
    fig.savefig(os.path.join(curves_dir, 'performance_curves.png'),
                dpi=300, bbox_inches='tight', format='png')
    fig.savefig(os.path.join(curves_dir, 'performance_curves.pdf'),
                dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    with open(os.path.join(curves_dir, 'curve_data.json'), 'w') as f:
        json.dump(curve_data, f, indent=4)
    summary_stats = {}
    for dataset_name, data in curve_data.items():
        roc_data = np.array(data['roc']['fpr']), np.array(data['roc']['tpr'])
        pr_data = np.array(data['pr']['recall']), np.array(data['pr']['precision'])
        summary_stats[dataset_name] = {
            'roc_auc': data['roc']['auc'],
            'pr_auc': data['pr']['auc'],
            'optimal_threshold': data['roc']['thresholds'][
                np.argmax(np.array(data['roc']['tpr']) - np.array(data['roc']['fpr']))]}

    with open(os.path.join(curves_dir, 'curve_summary.json'), 'w') as f:
        json.dump(summary_stats, f, indent=4)
    return curve_data, summary_stats

def save_curve_coordinates(curve_data, result_path):
    curves_dir = os.path.join(result_path, 'curves')
    os.makedirs(curves_dir, exist_ok=True)
    for dataset_name, data in curve_data.items():
        roc_df = pd.DataFrame({
            'FPR': data['roc']['fpr'],
            'TPR': data['roc']['tpr'],
            'Threshold': data['roc']['thresholds']})
        roc_df.to_csv(os.path.join(curves_dir, f'{dataset_name}_roc_coordinates.csv'), 
                      index=False)
        pr_df = pd.DataFrame({
            'Recall': data['pr']['recall'],
            'Precision': data['pr']['precision'],
            'Threshold': data['pr']['thresholds'] if len(data['pr']['thresholds']) == len(data['pr']['recall']) 
                        else np.append(data['pr']['thresholds'], np.nan)})
        pr_df.to_csv(os.path.join(curves_dir, f'{dataset_name}_pr_coordinates.csv'), 
                     index=False)
        
################################################################################################
def main():
    optimize_cuda_performance()
    torch.cuda.empty_cache()
    try:
        args = parse_args()
        set_seed(42)  # Ensure reproducibility
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(args.result_path, f'run_{run_timestamp}')
        os.makedirs(result_path, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(result_path, 'training.log')),
                logging.StreamHandler()])
        device_manager = DeviceManager()  
        DEVICE = device_manager.device 
        logging.info(f"Training will use device: {DEVICE}")
        if torch.cuda.is_available():
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logging.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logging.info("Loading and preprocessing datasets...")
        try:
            data_result = dataload()
            if len(data_result) != 12:
                raise ValueError(f"Expected 12 values from dataload, got {len(data_result)}")

            (train_dataset, val_dataset, test_dataset, 
             ts1_dataset, ts2_dataset, ts3_dataset,
             num_node_features, num_edge_features,
             num_node_types, num_edge_types, class_weights,
             smiles_config) = data_result
            datasets = [train_dataset, val_dataset, test_dataset, 
                       ts1_dataset, ts2_dataset, ts3_dataset]
            loaders = create_optimized_dataloaders(
                datasets=datasets,
                batch_size=256,
                num_workers=4)

            logging.info(f"""Data loading completed:
            - Node features: {num_node_features}
            - Edge features: {num_edge_features}
            - Node types: {num_node_types}
            - Edge types: {num_edge_types}""")

        except Exception as e:
            logging.error(f"Error in data loading: {str(e)}")
            raise
        logging.info("Initializing model and training components...")
        model, optimizer, criterion, scheduler = setup_training(
            args=args,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            device=DEVICE,
            train_loader=loaders['train'])

        original_forward = model.forward
        smiles_status = check_smiles_usage(model, loaders.get('train'))
        if smiles_status['model_has_processor'] and smiles_status['dataloader_has_smiles']:
            logging.info("SMILES processing enabled and data available")
        else:
            logging.info("Running in graph-only mode")
        logging.info("Starting training pipeline...")
        best_model, best_val_loss, best_metrics = train_and_evaluate(
            model=model,
            loaders=loaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            device=DEVICE
        )
        model_save_path = os.path.join(result_path, 'best_model.pth')
        torch.save({
            'model_state_dict': best_model,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'model_config': {
                'in_dim': num_node_features,
                'hidden_dim': 128,
                'num_layers':4,
                'num_heads': 4,
                'dropout': 0,
                'num_classes': 1,
                'num_node_types': num_node_types,
                'num_edge_types': num_edge_types,
                'processing_steps': 3
            }
        }, model_save_path)
        logging.info(f"Best model saved to {model_save_path}")
        logging.info("Running final evaluation and generating visualizations...")
        model.load_state_dict(best_model)
        test_loaders = {
            'test': loaders['test'],
            'ts1': loaders['ts1'],
            'ts2': loaders['ts2'],
            'ts3': loaders['ts3']
        }
        results = {}
        all_predictions = {}
        all_labels = {}
        for name, loader in test_loaders.items():
            if loader is None:
                continue

            logging.info(f"\nEvaluating on {name} dataset...")
            try:
                test_loss, metrics, predictions, labels = test_model(
                    model=model,
                    test_loader=loader,
                    criterion=criterion)

                results[name] = {
                    'loss': test_loss,
                    'metrics': metrics}
                all_predictions[name] = predictions
                all_labels[name] = labels
                logging.info(f"Results for {name}:")
                logging.info(f"Loss: {test_loss:.4f}")
                logging.info(format_metrics(metrics, name))

            except Exception as e:
                logging.error(f"Error evaluating {name}: {str(e)}")
                continue
        logging.info("Generating and saving performance curves...")
        try:
            curve_data, summary_stats = plot_and_save_curves(
                all_predictions, all_labels, result_path)
            save_curve_coordinates(curve_data, result_path)
            logging.info("\nPerformance Summary:")
            for dataset_name, stats in summary_stats.items():
                logging.info(f"\n{dataset_name}:")
                logging.info(f"  ROC AUC: {stats['roc_auc']:.3f}")
                logging.info(f"  PR AUC: {stats['pr_auc']:.3f}")
                logging.info(f"  Optimal threshold: {stats['optimal_threshold']:.3f}")

        except Exception as e:
            logging.error(f"Error generating performance curves: {str(e)}")
            traceback.print_exc()
        try:
            save_results(results, all_predictions, all_labels, result_path)
            logging.info(f"\nTraining complete. Results saved in {result_path}")
        except Exception as e:
            logging.error(f"Error saving final results: {str(e)}")

    except Exception as e:
        logging.error(f"Critical error in main execution: {str(e)}")
        traceback.print_exc()

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Training session ended.")
        plt.close('all')

if __name__ == "__main__":
    main()