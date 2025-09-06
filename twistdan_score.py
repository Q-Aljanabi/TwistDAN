import torch
import pandas as pd
import numpy as np
import logging
import os
import json
from tqdm import tqdm
import warnings
from datetime import datetime
import re
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    print("RDKit imported successfully")
except ImportError:
    print("Warning: RDKit not available. Install with: conda install -c conda-forge rdkit")

try:
    from torch_geometric.data import Data, Batch
    print("PyTorch Geometric imported successfully")
except ImportError:
    print("Warning: PyTorch Geometric not available. Install with: pip install torch-geometric")
from TwistDAN_data_preprocess import MoleculeProcessor, CustomMoleculeTokenizer
from TwistDAN import TwistDAN


warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

class SAPredictor:
    def __init__(self, model_path='best_model.pth', device=None):
        self.model_path = model_path
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.molecule_processor = MoleculeProcessor()
        self.tokenizer = CustomMoleculeTokenizer()
        self.model = None
        self.model_config = None
        self.total_processed = 0
        self.failed_conversions = 0
        logging.info(f"TwistDAN Predictor initialized. Using device: {self.device}")
    
    def load_model(self):
        try:
            logging.info(f"Loading model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
                self.model_config = checkpoint['model_config']
                model_state_dict = checkpoint['model_state_dict']
            else:
                logging.warning("Model config not found in checkpoint, using defaults")
                self.model_config = {
                    'in_dim': 12,  
                    'hidden_dim': 128,
                    'num_layers': 4,
                    'num_heads': 4,
                    'dropout': 0,
                    'num_classes': 1,
                    'num_node_types': 13, 
                    'num_edge_types': 5,   
                    'processing_steps': 4}
                model_state_dict = checkpoint
            self.model = TwistDAN(**self.model_config)
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            self.model.eval()
            logging.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return False
    
    def is_valid_smiles(self, smiles):
        if not isinstance(smiles, str):
            return False
        
        if len(smiles.strip()) < 2:
            return False
        
        if re.match(r'^\d+[-|]\d+', smiles):
            return False
        
        if any(word in smiles.lower() for word in ['database', 'discovery', 'chebi', 'drugbank']):
            return False
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def smiles_to_graph(self, smiles):
        try:
            if not self.is_valid_smiles(smiles):
                self.failed_conversions += 1
                return None
            encoded = self.tokenizer.encode(smiles)
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.failed_conversions += 1
                return self._create_fallback_graph(input_ids, attention_mask)
            AllChem.Compute2DCoords(mol)
            atom_features = []
            node_types = []
            for atom in mol.GetAtoms():
                atom_features.append(self.molecule_processor.get_atom_features(atom))
                node_types.append(self.molecule_processor.atom_types.get(
                    atom.GetSymbol(), self.molecule_processor.atom_types['other']))
            
            x = torch.tensor(atom_features, dtype=torch.float)
            node_types = torch.tensor(node_types, dtype=torch.long)
            edge_indices = []
            edge_features = []
            edge_types = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices += [[i, j], [j, i]]
                bond_feature = self.molecule_processor.get_bond_features(bond)
                edge_features += [bond_feature, bond_feature]
                bond_type = self.molecule_processor.bond_types.get(
                    bond.GetBondType(), self.molecule_processor.bond_types['other'])
                edge_types += [bond_type, bond_type]
            if not edge_indices:  # If no bonds, add self-loop
                edge_indices = [[0, 0]]
                edge_features = [[1.0, 0.0, 0.0, 0.0, 0.0]]
                edge_types = [self.molecule_processor.bond_types['other']]
            
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            edge_types = torch.tensor(edge_types, dtype=torch.long)
            return Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_types=node_types,
                edge_types=edge_types,
                smiles_input_ids=input_ids,
                smiles_attention_mask=attention_mask)
            
        except Exception as e:
            logging.warning(f"Error processing SMILES {smiles}: {str(e)}")
            self.failed_conversions += 1
            return None
    
    def _create_fallback_graph(self, input_ids=None, attention_mask=None):
        x = torch.tensor([[6, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.float)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float)
        node_types = torch.tensor([self.molecule_processor.atom_types['C']], dtype=torch.long)
        edge_types = torch.tensor([self.molecule_processor.bond_types['other']], dtype=torch.long)
        
        if input_ids is None:
            input_ids = torch.zeros(self.tokenizer.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.tokenizer.max_length, dtype=torch.long)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_types=node_types,
            edge_types=edge_types,
            smiles_input_ids=input_ids,
            smiles_attention_mask=attention_mask)
    
    def predict_single(self, smiles):
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Failed to load model")

        graph_data = self.smiles_to_graph(smiles)
        if graph_data is None:
            return None
            
        graph_data = graph_data.to(self.device)
        # Predict
        with torch.no_grad():
            model_output = self.model(graph_data)
            if isinstance(model_output, tuple):
                class_output = model_output[0]
            else:
                class_output = model_output
            probability = torch.sigmoid(class_output).cpu().item()
        
        return probability
    
    def predict_batch(self, smiles_list, batch_size=32, skip_invalid=True):
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Failed to load model")
        
        predictions = []
        valid_indices = []
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Predicting TwistDAN scores"):
            batch_smiles = smiles_list[i:i + batch_size]
            batch_graphs = []
            batch_valid_indices = []
            for j, smiles in enumerate(batch_smiles):
                self.total_processed += 1
                graph_data = self.smiles_to_graph(smiles)
                
                if graph_data is not None:
                    batch_graphs.append(graph_data)
                    batch_valid_indices.append(i + j)

                elif not skip_invalid:
                    fallback_graph = self._create_fallback_graph()
                    batch_graphs.append(fallback_graph)
                    batch_valid_indices.append(i + j)
            
            if not batch_graphs:

                predictions.extend([None] * len(batch_smiles))
                continue

            try:
                batch_data = Batch.from_data_list(batch_graphs).to(self.device)
                
                with torch.no_grad():
                    model_output = self.model(batch_data)
                    if isinstance(model_output, tuple):
                        class_output = model_output[0]
                    else:
                        class_output = model_output

                    probabilities = torch.sigmoid(class_output).cpu().numpy()
                    batch_predictions = [None] * len(batch_smiles)
                    for k, prob in enumerate(probabilities):
                        original_idx = batch_valid_indices[k] - i
                        if 0 <= original_idx < len(batch_smiles):
                            batch_predictions[original_idx] = float(prob)
                    
                    predictions.extend(batch_predictions)
                    
            except Exception as e:
                logging.error(f"Error processing batch {i}: {str(e)}")
                predictions.extend([None] * len(batch_smiles))
        
        return predictions
    
    def clean_smiles_data(self, df, smiles_column='smiles'):
        logging.info(f"Cleaning SMILES data. Original size: {len(df)}")
        
        # Remove NaN values
        df_clean = df[df[smiles_column].notna()].copy()
        
        # Remove obvious database entries
        mask = df_clean[smiles_column].str.contains(
            r'ANPDB|Discovery|database|ChEBI|DrugBank|GNPS|KNApSaCK|NANPDB|NPACT|NPASS|NPEdia|NuBBEDB|SANCDB|TCMDB|TIPdb|VietHerb|\d+-\d+-\d+',
            na=False, case=False, regex=True)
        df_clean = df_clean[~mask]

        df_clean = df_clean[df_clean[smiles_column].str.count(r'\|') <= 2]
        df_clean = df_clean[df_clean[smiles_column].str.len().between(3, 500)]
        if len(df_clean) < 10000:  # Only for smaller datasets
            logging.info("Performing RDKit validation (may take a while for large datasets)")
            valid_mask = df_clean[smiles_column].apply(self.is_valid_smiles)
            df_clean = df_clean[valid_mask]
        
        logging.info(f"After cleaning: {len(df_clean)} molecules ({len(df) - len(df_clean)} removed)")
        
        return df_clean
    
    def predict_from_csv(self, input_csv, smiles_column='smiles', output_csv=None, batch_size=32, max_molecules=None, clean_data=True):
        logging.info(f"Reading CSV file: {input_csv}")
        try:
            df = pd.read_csv(input_csv)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
        
        if smiles_column not in df.columns:
            raise ValueError(f"Column '{smiles_column}' not found in CSV. Available columns: {list(df.columns)}")
        if clean_data:
            df = self.clean_smiles_data(df, smiles_column)
        if max_molecules and len(df) > max_molecules:
            logging.info(f"Limiting to first {max_molecules} molecules for testing")
            df = df.head(max_molecules)
        
        logging.info(f"Processing {len(df)} molecules")
        self.total_processed = 0
        self.failed_conversions = 0
        smiles_list = df[smiles_column].tolist()
        predictions = self.predict_batch(smiles_list, batch_size)
        df = df.copy()
        df['TwistDAN_Score'] = predictions
        result_columns = [smiles_column, 'TwistDAN_Score']
        df = df[result_columns]
        if output_csv:
            df.to_csv(output_csv, index=False)
            logging.info(f"Results saved to: {output_csv}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = f"TwistDAN_predictions_{timestamp}.csv"
            df.to_csv(output_csv, index=False)
            logging.info(f"Results saved to: {output_csv}")

        valid_predictions = df[df['TwistDAN_Score'].notna()]['TwistDAN_Score'].values
        logging.info(f"Prediction Summary:")
        logging.info(f"  - Total molecules processed: {len(df)}")
        logging.info(f"  - Successful predictions: {len(valid_predictions)}")
        logging.info(f"  - Failed predictions: {len(df) - len(valid_predictions)}")
        logging.info(f"  - Success rate: {len(valid_predictions)/len(df)*100:.1f}%")
        if len(valid_predictions) > 0:
            logging.info(f"  - Mean TwistDAN Score: {np.mean(valid_predictions):.3f}")
            logging.info(f"  - Std TwistDAN Score: {np.std(valid_predictions):.3f}")
            logging.info(f"  - Min TwistDAN Score: {np.min(valid_predictions):.3f}")
            logging.info(f"  - Max TwistDAN Score: {np.max(valid_predictions):.3f}")
        
        return df


def main():
    predictor = SAPredictor(model_path='best_model.pth')
    smiles_list = [
        'COc4ccc3nc(NC(=O)CSc2nnc(c1ccccc1C)[nH]2)sc3c4',  # High complexity
        'OC8Cc7c(O)c(C2C(O)C(c1ccc(O)c(O)c1)Oc3cc(O)ccc23)c(O)c(C5C(O)C(c4ccc(O)c(O)c4)Oc6cc(O)ccc56)c7OC8c9ccc(O)c(O)c9',
        'NC(=O)Nc1nsnc1C(=O)Nc2ccccc2',  # Medium-high
        'CCc2ccc(c1ccccc1)cc2',          # Medium
        'CC(C)(C)c4ccc(C(=O)Nc3nc2C(CC(=O)NCC#C)C1(C)CCC(O)C(C)(CO)C1Cc2s3)cc4',
        'COc1ccccc1c2ccccc2',            # Medium-low
        'CC(=O)Nc1ccccc1NC(=O)COc2ccccc2',  # Medium
        'CSc2ccc(OCC(=O)Nc1ccc(C(C)C)cc1)cc2',  # Medium
        'Cc2ccc(C(=O)Nc1ccccc1)cc2',     # Low-medium
        'COc1ccc(Cl)cc1',                # Low
        'c1ccc2c(c1)ccc3c2ccc4c3cccc4',  # Medium (anthracene)
        'CC(C)C1CCC(C(C)C)CC1'          # Low
    ]
    
    try:
        logging.info(f"Processing {len(smiles_list)} SMILES")
        predictions = predictor.predict_batch(smiles_list, batch_size=32, skip_invalid=False)
        predictions = [pred if pred is not None else 0.5 for pred in predictions]

        results_df = pd.DataFrame({
            'SMILES': smiles_list,
            'TwistDAN_Score': predictions })
        output_file = 'TwistDAN_scores.csv'
        results_df.to_csv(output_file, index=False)
        logging.info("Prediction completed successfully!")
        print(f"Processed {len(results_df)} molecules")
        print(f"Results saved to {output_file}")
        print("\nTwistDAN Scores:")
        for i, (smiles, score) in enumerate(zip(smiles_list, predictions)):
            print(f"{i+1:2d}. {score:.4f}")
        
    except Exception as e:
        logging.error(f"Error in prediction process: {e}")

if __name__ == "__main__":
    main()