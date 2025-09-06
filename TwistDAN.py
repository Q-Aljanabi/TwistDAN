# TwistDAN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data
import warnings
import logging 
import time
import os

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class TwistDAN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_heads, dropout, num_classes, num_node_types, num_edge_types, processing_steps):
        super().__init__()
        self.graph_processor = MolecularGraphEncoder(
            in_dim, hidden_dim, num_layers, num_heads, 
            dropout, num_classes, num_node_types, 
            num_edge_types, processing_steps
        )
        self.domain_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim//2, 1)
            ) for _ in range(4)  
        ])
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data, domain_idx=0, alpha=1.0):
        graph_features = self.graph_processor(data)
        domain_features = GradientReversalLayer.apply(graph_features, alpha)
        domain_output = self.domain_classifiers[domain_idx](domain_features)
        class_output = self.graph_classifier(graph_features).squeeze(-1)
        if isinstance(class_output, tuple):
            class_output = class_output[0]
            
        return class_output, domain_output
   
class MolecularGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_heads, dropout, num_classes, num_node_types, num_edge_types, processing_steps):
        super().__init__()
        self.node_embedding = nn.Linear(in_dim, hidden_dim) # Node feature encoding
        self.gat_layers = nn.ModuleList([                   # Graph Attention layers
            GATConv(
                hidden_dim, 
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([                  # Layer normalization
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim) # Node type embeddings
        self.global_attention = nn.Sequential(                              # Global attention pooling
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.node_embedding(x)                      # Initial node feature embedding
        h = self.dropout(h)
        if hasattr(data, 'node_types'):                 # Add node type information if available
            node_type_embed = self.node_type_embedding(data.node_types)
            h = h + node_type_embed
        
        for gat, norm in zip(self.gat_layers, self.layer_norms):  # Process through GAT layers
            h_prev = h
            h = gat(h, edge_index)
            h = norm(h)
            h = h + h_prev                                          # Residual connection
            h = F.relu(h)
            h = self.dropout(h)
    
        attention_weights = self.global_attention(h)                # Global pooling
        attention_weights = F.softmax(attention_weights, dim=0)
        h_graph = torch.sum(h * attention_weights, dim=0)
        h_graph = h_graph + global_mean_pool(h, batch)              # Additional global mean pooling
        
        return h_graph
    

class DomainAdaptiveLayer(nn.Module):
    def __init__(self, feature_dim, domains=4):
        super().__init__()
        self.domain_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim//2),
                nn.ReLU(),
                nn.Linear(feature_dim//2, 1)
            ) for _ in range(domains)
        ])
        self.domain_adapters = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(domains)
        ])
        
    def forward(self, x, domain_idx=0, alpha=1.0):
        domain_pred = self.domain_classifiers[domain_idx](GradientReversalLayer.apply(x, alpha))
        adapted_features = self.domain_adapters[domain_idx](x)
        return adapted_features, domain_pred
    
class ContrastiveLearning(nn.Module):
    def __init__(self, hidden_dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x1, x2):              # Project features
        z1 = F.normalize(self.projection(x1), dim=-1)
        z2 = F.normalize(self.projection(x2), dim=-1)
        similarity = torch.matmul(z1, z2.T) / self.temperature                 # Compute similarity
        labels = torch.arange(similarity.size(0), device=similarity.device)    # Contrastive loss
        loss = F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.T, labels)
        return loss / 2

