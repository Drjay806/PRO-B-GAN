"""
Prot-B-GAN: Progressive Adversarial Framework for Knowledge Graph Completion
=============================================================================

A complete pipeline that adapts progressive adversarial training from computer vision 
to knowledge graph completion, addressing hub bias and improving tail diversity.

Features:
- R-GCN preprocessing for structure-aware embeddings
- Three-tier progressive training (Pretrain → RL → Full Adversarial)
- Hard negative mining for hub bias mitigation
- Comprehensive early stopping for efficient training
- Optional verbose output or progress bars
- Plug-and-play interface for any CSV/Excel data
- Modular embedding initialization (R-GCN, ComplEx, DistMult, TransE, Random)
- Dynamic training schedule rebalancing

Installation (Google Colab/CUDA 11.8):
    !pip install "numpy<2.0"
    !pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    !pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib \
        -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
    !pip install torch-geometric
    !pip install scikit-learn pandas matplotlib tqdm

Local Installation:
    pip install "numpy<2.0"
    pip install torch==2.0.0 torchvision torchaudio
    pip install torch-geometric
    pip install scikit-learn pandas matplotlib tqdm

Usage:
    # Basic usage with your original settings
    python prot_b_gan.py --data_root /path/to/data --train_file train.csv --val_file val.csv --test_file test.csv
    
    # Custom embedding initialization
    python prot_b_gan.py --data_root /path/to/data --embedding_init distmult --distmult_epochs 150
    
    # Full customization
    python prot_b_gan.py --data_root /path/to/data --embed_dim 256 --epochs 50 --g_lr 0.001 --verbose
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import pickle
import argparse
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import RGCNConv
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score, matthews_corrcoef, roc_auc_score, precision_score, recall_score
from sklearn.decomposition import PCA
import numpy as np
from collections import deque
import time
import random
from tqdm import tqdm
from abc import ABC, abstractmethod

# =============================================================================
# VERSION COMPATIBILITY CHECKS
# =============================================================================

def check_environment():
    """Check environment compatibility and print versions"""
    print("Environment Check:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   NumPy: {np.__version__}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        import torch_geometric
        print(f"   PyTorch Geometric: {torch_geometric.__version__}")
    except ImportError:
        print("PyTorch Geometric not found!")
        print("Please install with the commands in the docstring")
        return False
    
    # Check numpy version compatibility
    numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
    if numpy_version >= (2, 0):
        print("WARNING: NumPy 2.0+ detected. Some PyTorch operations may be incompatible.")
        print("   Recommended: pip install \"numpy<2.0\"")
    
    # Check torch version compatibility  
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version != (2, 0):
        print("WARNING: This code is optimized for PyTorch 2.0.0")
        print("Current version may work but compatibility is not guaranteed")
    
    print("Environment check complete\n")
    return True

# =============================================================================
# COMPATIBILITY FIXES
# =============================================================================

# Fix for numpy/torch compatibility issues
def safe_tensor_to_numpy(tensor):
    """Safely convert tensor to numpy with version compatibility"""
    if hasattr(tensor, 'detach'):
        return tensor.detach().cpu().numpy()
    return tensor.cpu().numpy()

# Fix for torch geometric compatibility
def safe_rgcn_conv(*args, **kwargs):
    """Wrapper for RGCNConv with version compatibility fixes"""
    try:
        return RGCNConv(*args, **kwargs)
    except Exception as e:
        print(f"RGCNConv initialization error: {e}")
        print("This might be due to version incompatibility")
        raise

# =============================================================================
# MODULAR EMBEDDING INITIALIZATION SYSTEM
# =============================================================================

class EmbeddingInitializer(ABC):
    """Abstract base class for different embedding initialization strategies"""
    
    @abstractmethod
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, progress_manager, verbose=True):
        """Initialize node and relation embeddings. Returns: (node_embeddings, relation_embeddings)"""
        pass
    
    @abstractmethod
    def get_name(self):
        """Return the name of this initialization method"""
        pass

class RandomInitializer(EmbeddingInitializer):
    """Random embedding initialization"""
    
    def __init__(self, embed_dim=500):
        self.embed_dim = embed_dim
    
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, progress_manager, verbose=True):
        print_progress(f"Initializing random embeddings (dim={self.embed_dim})...", verbose)
        
        node_embeddings = torch.randn(num_entities, self.embed_dim, device=device)
        relation_embeddings = torch.randn(num_relations, self.embed_dim, device=device)
        nn.init.xavier_uniform_(node_embeddings)
        nn.init.xavier_uniform_(relation_embeddings)
        
        print_progress("Random embedding initialization complete", verbose)
        return node_embeddings, relation_embeddings
    
    def get_name(self):
        return "Random"

class TransEInitializer(EmbeddingInitializer):
    """TransE embedding initialization - excellent for hierarchical relations"""
    
    def __init__(self, embed_dim=500, epochs=100, lr=0.01, margin=1.0):
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.lr = lr
        self.margin = margin
    
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, progress_manager, verbose=True):
        print_progress(f"Initializing TransE embeddings (dim={self.embed_dim})...", verbose)
        
        entity_emb = nn.Embedding(num_entities, self.embed_dim).to(device)
        relation_emb = nn.Embedding(num_relations, self.embed_dim).to(device)
        
        nn.init.xavier_uniform_(entity_emb.weight)
        nn.init.xavier_uniform_(relation_emb.weight)
        
        # Normalize entity embeddings
        with torch.no_grad():
            entity_emb.weight.data = F.normalize(entity_emb.weight.data, p=2, dim=1)
        
        optimizer = optim.Adam(list(entity_emb.parameters()) + list(relation_emb.parameters()), lr=self.lr)
        train_triples = torch.tensor(train_df.values, dtype=torch.long).to(device)
        
        progress_manager.start_tier(0, self.epochs)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            batch_size = min(1024, len(train_triples))
            batch_idx = torch.randperm(len(train_triples))[:batch_size]
            batch_triples = train_triples[batch_idx]
            
            h, r, t = batch_triples[:, 0], batch_triples[:, 1], batch_triples[:, 2]
            h_emb, r_emb, t_emb = entity_emb(h), relation_emb(r), entity_emb(t)
            
            # TransE: ||h + r - t||
            pos_scores = torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)
            
            # Negative sampling
            neg_h = torch.randint(0, num_entities, (batch_size,), device=device)
            neg_t = torch.randint(0, num_entities, (batch_size,), device=device)
            
            neg_h_emb = entity_emb(neg_h)
            neg_t_emb = entity_emb(neg_t)
            
            # Corrupt head or tail randomly
            corrupt_head = torch.rand(batch_size, device=device) > 0.5
            neg_scores = torch.where(
                corrupt_head,
                torch.norm(neg_h_emb + r_emb - t_emb, p=2, dim=1),
                torch.norm(h_emb + r_emb - neg_t_emb, p=2, dim=1)
            )
            
            # Margin loss
            loss = F.relu(self.margin + pos_scores - neg_scores).mean()
            loss.backward()
            optimizer.step()
            
            # Normalize entity embeddings
            with torch.no_grad():
                entity_emb.weight.data = F.normalize(entity_emb.weight.data, p=2, dim=1)
            
            if verbose and epoch % 10 == 0:
                print_progress(f"  TransE Epoch {epoch}: Loss = {loss.item():.4f}", verbose)
            
            progress_manager.update_tier(0, epoch, {'loss': loss.item()})
        
        progress_manager.close_tier(0)
        print_progress("TransE embedding initialization complete", verbose)
        
        return entity_emb.weight.detach(), relation_emb.weight.detach()
    
    def get_name(self):
        return "TransE"

class DistMultInitializer(EmbeddingInitializer):
    """DistMult embedding initialization - simple and effective baseline"""
    
    def __init__(self, embed_dim=500, epochs=100, lr=0.01, regularization=0.01):
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.lr = lr
        self.regularization = regularization
    
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, progress_manager, verbose=True):
        print_progress(f"Initializing DistMult embeddings (dim={self.embed_dim})...", verbose)
        
        entity_emb = nn.Embedding(num_entities, self.embed_dim).to(device)
        relation_emb = nn.Embedding(num_relations, self.embed_dim).to(device)
        nn.init.xavier_uniform_(entity_emb.weight)
        nn.init.xavier_uniform_(relation_emb.weight)
        
        optimizer = optim.Adam(list(entity_emb.parameters()) + list(relation_emb.parameters()), lr=self.lr)
        train_triples = torch.tensor(train_df.values, dtype=torch.long).to(device)
        
        progress_manager.start_tier(0, self.epochs)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            batch_size = min(1024, len(train_triples))
            batch_idx = torch.randperm(len(train_triples))[:batch_size]
            batch_triples = train_triples[batch_idx]
            
            h, r, t = batch_triples[:, 0], batch_triples[:, 1], batch_triples[:, 2]
            h_emb, r_emb, t_emb = entity_emb(h), relation_emb(r), entity_emb(t)
            
            # DistMult scoring: sum(h * r * t)
            pos_scores = (h_emb * r_emb * t_emb).sum(dim=1)
            
            # Negative sampling
            neg_t = torch.randint(0, num_entities, (batch_size,), device=device)
            neg_t_emb = entity_emb(neg_t)
            neg_scores = (h_emb * r_emb * neg_t_emb).sum(dim=1)
            
            # Logistic loss
            pos_loss = F.logsigmoid(pos_scores).mean()
            neg_loss = F.logsigmoid(-neg_scores).mean()
            loss = -(pos_loss + neg_loss)
            
            # L2 regularization
            reg_loss = self.regularization * (h_emb.norm(p=2).pow(2) + r_emb.norm(p=2).pow(2) + t_emb.norm(p=2).pow(2)) / (3 * batch_size)
            total_loss = loss + reg_loss
            total_loss.backward()
            optimizer.step()
            
            if verbose and epoch % 10 == 0:
                print_progress(f"  DistMult Epoch {epoch}: Loss = {total_loss.item():.4f}", verbose)
            
            progress_manager.update_tier(0, epoch, {'loss': total_loss.item()})
        
        progress_manager.close_tier(0)
        print_progress("DistMult embedding initialization complete", verbose)
        
        return entity_emb.weight.detach(), relation_emb.weight.detach()
    
    def get_name(self):
        return "DistMult"

class ComplExInitializer(EmbeddingInitializer):
    """ComplEx embedding initialization - handles asymmetric relations well"""
    
    def __init__(self, embed_dim=500, epochs=100, lr=0.01, regularization=0.01):
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.lr = lr
        self.regularization = regularization
    
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, progress_manager, verbose=True):
        print_progress(f"Initializing ComplEx embeddings (dim={self.embed_dim})...", verbose)
        
        # ComplEx uses complex embeddings: real and imaginary parts
        entity_real = nn.Embedding(num_entities, self.embed_dim).to(device)
        entity_imag = nn.Embedding(num_entities, self.embed_dim).to(device)
        relation_real = nn.Embedding(num_relations, self.embed_dim).to(device)
        relation_imag = nn.Embedding(num_relations, self.embed_dim).to(device)
        
        # Xavier initialization
        for emb in [entity_real, entity_imag, relation_real, relation_imag]:
            nn.init.xavier_uniform_(emb.weight, gain=0.1)
        
        optimizer = optim.Adam(
            list(entity_real.parameters()) + list(entity_imag.parameters()) +
            list(relation_real.parameters()) + list(relation_imag.parameters()),
            lr=self.lr
        )
        
        train_triples = torch.tensor(train_df.values, dtype=torch.long).to(device)
        progress_manager.start_tier(0, self.epochs)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            batch_size = min(1024, len(train_triples))
            batch_idx = torch.randperm(len(train_triples))[:batch_size]
            batch_triples = train_triples[batch_idx]
            
            h, r, t = batch_triples[:, 0], batch_triples[:, 1], batch_triples[:, 2]
            
            # Get complex embeddings
            h_real, h_imag = entity_real(h), entity_imag(h)
            r_real, r_imag = relation_real(r), relation_imag(r)
            t_real, t_imag = entity_real(t), entity_imag(t)
            
            # ComplEx scoring: Re(<h, r, conj(t)>)
            pos_scores = (
                (h_real * r_real * t_real).sum(dim=1) +
                (h_real * r_imag * t_imag).sum(dim=1) +
                (h_imag * r_real * t_imag).sum(dim=1) -
                (h_imag * r_imag * t_real).sum(dim=1)
            )
            
            # Negative sampling
            neg_t = torch.randint(0, num_entities, (batch_size,), device=device)
            neg_t_real, neg_t_imag = entity_real(neg_t), entity_imag(neg_t)
            
            neg_scores = (
                (h_real * r_real * neg_t_real).sum(dim=1) +
                (h_real * r_imag * neg_t_imag).sum(dim=1) +
                (h_imag * r_real * neg_t_imag).sum(dim=1) -
                (h_imag * r_imag * neg_t_real).sum(dim=1)
            )
            
            # Logistic loss
            pos_loss = F.logsigmoid(pos_scores).mean()
            neg_loss = F.logsigmoid(-neg_scores).mean()
            loss = -(pos_loss + neg_loss)
            
            # L2 regularization
            reg_loss = self.regularization * (
                h_real.norm(p=2).pow(2) + h_imag.norm(p=2).pow(2) +
                r_real.norm(p=2).pow(2) + r_imag.norm(p=2).pow(2) +
                t_real.norm(p=2).pow(2) + t_imag.norm(p=2).pow(2)
            ) / (3 * batch_size)
            
            total_loss = loss + reg_loss
            total_loss.backward()
            optimizer.step()
            
            if verbose and epoch % 10 == 0:
                print_progress(f"  ComplEx Epoch {epoch}: Loss = {total_loss.item():.4f}", verbose)
            
            progress_manager.update_tier(0, epoch, {'loss': total_loss.item()})
        
        progress_manager.close_tier(0)
        
        # Combine real and imaginary parts for final embeddings
        final_entity_emb = torch.cat([entity_real.weight, entity_imag.weight], dim=1)
        final_relation_emb = torch.cat([relation_real.weight, relation_imag.weight], dim=1)
        
        print_progress("ComplEx embedding initialization complete", verbose)
        return final_entity_emb.detach(), final_relation_emb.detach()
    
    def get_name(self):
        return "ComplEx"

class RGCNInitializer(EmbeddingInitializer):
    """R-GCN based embedding initialization - structure-aware"""
    
    def __init__(self, embed_dim=500, rgcn_epochs=50, rgcn_lr=0.01, rgcn_layers=2):
        self.embed_dim = embed_dim
        self.rgcn_epochs = rgcn_epochs
        self.rgcn_lr = rgcn_lr
        self.rgcn_layers = rgcn_layers
    
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, progress_manager, verbose=True):
        return preprocess_with_rgcn(
            train_df, val_df, test_df, num_entities, num_relations, device, progress_manager, verbose,
            embed_dim=self.embed_dim, epochs=self.rgcn_epochs, lr=self.rgcn_lr, layers=self.rgcn_layers
        )
    
    def get_name(self):
        return "R-GCN"

def create_embedding_initializer(args):
    """Factory function to create embedding initializer based on args"""
    
    method = args.embedding_init.lower()
    
    if method == 'rgcn':
        return RGCNInitializer(
            embed_dim=args.embed_dim,
            rgcn_epochs=args.rgcn_epochs,
            rgcn_lr=args.rgcn_lr,
            rgcn_layers=args.rgcn_layers
        )
    elif method == 'random':
        return RandomInitializer(embed_dim=args.embed_dim)
    elif method == 'transe':
        return TransEInitializer(
            embed_dim=args.embed_dim,
            epochs=args.transe_epochs,
            lr=args.transe_lr,
            margin=args.transe_margin
        )
    elif method == 'distmult':
        return DistMultInitializer(
            embed_dim=args.embed_dim,
            epochs=args.distmult_epochs,
            lr=args.distmult_lr,
            regularization=args.distmult_regularization
        )
    elif method in ['complex', 'complEx']:
        return ComplExInitializer(
            embed_dim=args.embed_dim,
            epochs=args.complex_epochs,
            lr=args.complex_lr,
            regularization=args.complex_regularization
        )
    else:
        raise ValueError(f"Unknown embedding initialization method: {method}")

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_data(data_root, train_file, val_file, test_file, debug_mode=False, max_train=None, max_val=None, max_test=None):
    """Load data from CSV files with flexible column naming"""
    
    print(f"Loading data from {data_root}...")
    
    # Load CSV files
    train_df = pd.read_csv(os.path.join(data_root, train_file))
    val_df = pd.read_csv(os.path.join(data_root, val_file))
    test_df = pd.read_csv(os.path.join(data_root, test_file))
    
    possible_head_cols = ['H', 'head', 'subject', 'h', 'HEAD', 'SUBJECT']
    possible_rel_cols = ['R', 'relation', 'predicate', 'r', 'RELATION', 'PREDICATE']
    possible_tail_cols = ['T', 'tail', 'object', 't', 'TAIL', 'OBJECT']
    
    head_col = next((col for col in possible_head_cols if col in train_df.columns), None)
    rel_col = next((col for col in possible_rel_cols if col in train_df.columns), None)
    tail_col = next((col for col in possible_tail_cols if col in train_df.columns), None)
    
    if not all([head_col, rel_col, tail_col]):
        raise ValueError(f"Could not find head/relation/tail columns. Found columns: {list(train_df.columns)}")
    
    print(f"Detected columns: Head='{head_col}', Relation='{rel_col}', Tail='{tail_col}'")
    
    # Standardize column names
    for df in [train_df, val_df, test_df]:
        df.rename(columns={head_col: 'H', rel_col: 'R', tail_col: 'T'}, inplace=True)
    
    if debug_mode:
        if max_train and len(train_df) > max_train:
            train_df = train_df.sample(max_train, random_state=42)
        if max_val and len(val_df) > max_val:
            val_df = val_df.sample(max_val, random_state=42)
        if max_test and len(test_df) > max_test:
            test_df = test_df.sample(max_test, random_state=42)
        print(f"DEBUG MODE: Using {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    print(f"Data loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test triples")
    
    return train_df, val_df, test_df

def create_entity_relation_mappings(train_df, val_df, test_df):
    """Create entity and relation ID mappings"""
    
    # Combine all data
    all_data = pd.concat([train_df, val_df, test_df])
    
    # Get unique entities and relations
    all_entities = sorted(set(all_data['H'].unique()) | set(all_data['T'].unique()))
    all_relations = sorted(set(all_data['R'].unique()))
    
    # Create mappings
    entity_to_id = {entity: idx for idx, entity in enumerate(all_entities)}
    relation_to_id = {relation: idx for idx, relation in enumerate(all_relations)}
    
    # Convert dataframes to use IDs
    for df in [train_df, val_df, test_df]:
        df['H'] = df['H'].map(entity_to_id)
        df['R'] = df['R'].map(relation_to_id)
        df['T'] = df['T'].map(entity_to_id)
    
    num_entities = len(all_entities)
    num_relations = len(all_relations)
    
    print(f"Entities: {num_entities:,}, Relations: {num_relations:,}")
    
    return train_df, val_df, test_df, num_entities, num_relations, entity_to_id, relation_to_id

class TripletDataset(Dataset):
    """Dataset for knowledge graph triplets"""
    def __init__(self, triples):
        self.triples = triples
    def __len__(self):
        return len(self.triples)
    def __getitem__(self, idx):
        return torch.tensor(self.triples[idx], dtype=torch.long)

# =============================================================================
# R-GCN PREPROCESSING MODULE
# =============================================================================

class RGCNDistMult(nn.Module):
    """R-GCN + DistMult for embedding preprocessing with version compatibility"""
    
    def __init__(self, num_entities, num_relations, hidden_dim=500, num_bases=None, num_layers=2):
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        
        # Set num_bases as in paper
        if num_bases is None:
            num_bases = min(num_relations, max(25, num_relations // 4))
        
        print(f"R-GCN Configuration: {num_entities:,} entities, {num_relations:,} relations")
        print(f"Hidden dim: {hidden_dim}, Bases: {num_bases}, Layers: {num_layers}")
        
        # Entity and relation embeddings
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim)
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        
        # R-GCN layers with compatibility wrapper
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            try:
                # Try standard initialization
                layer = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases)
                self.layers.append(layer)
            except Exception as e:
                print(f"Warning: RGCNConv layer {i} initialization failed: {e}")
                print("Attempting fallback initialization...")
                # Fallback without num_bases if needed
                try:
                    layer = RGCNConv(hidden_dim, hidden_dim, num_relations)
                    self.layers.append(layer)
                    print(f"   Fallback successful for layer {i}")
                except Exception as e2:
                    print(f"   Fallback failed: {e2}")
                    raise RuntimeError(f"Could not initialize R-GCN layer {i}. Check PyTorch Geometric installation.")
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward_entities(self, edge_index, edge_type):
        """Forward pass through R-GCN with error handling"""
        try:
            x = self.entity_embedding.weight
            for i, layer in enumerate(self.layers):
                x = layer(x, edge_index, edge_type)
                x = F.relu(x)
                x = self.dropout(x)
            return x
        except Exception as e:
            print(f"Error in R-GCN forward pass: {e}")
            print("Edge index shape:", edge_index.shape if hasattr(edge_index, 'shape') else 'Unknown')
            print("Edge type shape:", edge_type.shape if hasattr(edge_type, 'shape') else 'Unknown')
            raise
    
    def distmult_score(self, h, r, t):
        """DistMult scoring function"""
        return (h * r * t).sum(dim=-1)

def preprocess_with_rgcn(train_df, val_df, test_df, num_entities, num_relations, device, progress_manager, verbose=True, embed_dim=500, epochs=50, lr=0.01, layers=2):
    """Generate initial embeddings using R-GCN preprocessing with early stopping"""
    
    print_progress("Starting R-GCN preprocessing for structure-aware embedding initialization...", verbose)
    
    # Build graph structure
    all_data = pd.concat([train_df, val_df, test_df])
    all_triples = torch.tensor(all_data.values, dtype=torch.long)
    edge_index = all_triples[:, [0, 2]].t().contiguous().to(device)
    edge_type = all_triples[:, 1].to(device)
    
    # Initialize R-GCN model
    rgcn_model = RGCNDistMult(num_entities, num_relations, embed_dim, num_layers=layers).to(device)
    optimizer = optim.Adam(rgcn_model.parameters(), lr=lr)
    
    # Training and validation data
    train_triples = torch.tensor(train_df.values, dtype=torch.long).to(device)
    val_triples = torch.tensor(val_df.values, dtype=torch.long).to(device)
    
    print_progress(f"Training R-GCN for up to {epochs} epochs with early stopping...", verbose)
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    # Start progress tracking
    progress_manager.start_tier(0, epochs)
    
    for epoch in range(epochs):
        # Training
        rgcn_model.train()
        optimizer.zero_grad()
        
        entity_embeddings = rgcn_model.forward_entities(edge_index, edge_type)
        
        # DistMult training
        pos_heads = entity_embeddings[train_triples[:, 0]]
        pos_rels = rgcn_model.relation_embedding(train_triples[:, 1])
        pos_tails = entity_embeddings[train_triples[:, 2]]
        pos_scores = rgcn_model.distmult_score(pos_heads, pos_rels, pos_tails)
        
        # Simple negative sampling
        neg_tails = torch.randint(0, num_entities, (len(train_triples),), device=device)
        neg_tail_embs = entity_embeddings[neg_tails]
        neg_scores = rgcn_model.distmult_score(pos_heads, pos_rels, neg_tail_embs)
        
        # Margin loss
        train_loss = F.relu(1.0 + neg_scores - pos_scores).mean()
        train_loss.backward()
        optimizer.step()
        
        # Validation (every 5 epochs)
        if epoch % 5 == 0:
            rgcn_model.eval()
            with torch.no_grad():
                val_entity_embeddings = rgcn_model.forward_entities(edge_index, edge_type)
                val_pos_heads = val_entity_embeddings[val_triples[:, 0]]
                val_pos_rels = rgcn_model.relation_embedding(val_triples[:, 1])
                val_pos_tails = val_entity_embeddings[val_triples[:, 2]]
                val_pos_scores = rgcn_model.distmult_score(val_pos_heads, val_pos_rels, val_pos_tails)
                
                val_neg_tails = torch.randint(0, num_entities, (len(val_triples),), device=device)
                val_neg_tail_embs = val_entity_embeddings[val_neg_tails]
                val_neg_scores = rgcn_model.distmult_score(val_pos_heads, val_pos_rels, val_neg_tail_embs)
                
                val_loss = F.relu(1.0 + val_neg_scores - val_pos_scores).mean()
            
            # Early stopping check (lower validation loss is better)
            best_val_loss, patience_counter, should_stop = check_early_stopping(
                -val_loss.item(), -best_val_loss, patience_counter, 10, verbose
            )
            best_val_loss = -best_val_loss  # Convert back to positive
            
            if verbose and epoch % 10 == 0:
                print_progress(f"  R-GCN Epoch {epoch}: Train Loss = {train_loss.item():.4f}, Val Loss = {val_loss.item():.4f}", verbose)
            
            if should_stop:
                print_progress(f"  R-GCN early stopping at epoch {epoch}", verbose)
                break
        
        # Update progress
        progress_manager.update_tier(0, epoch, {'loss': train_loss.item()})
    
    progress_manager.close_tier(0)
    
    training_time = time.time() - start_time
    print_progress(f"R-GCN preprocessing complete in {training_time/60:.1f} minutes", verbose)
    
    # Extract final embeddings
    with torch.no_grad():
        final_node_embeddings = rgcn_model.forward_entities(edge_index, edge_type)
        final_rel_embeddings = rgcn_model.relation_embedding.weight
    
    return final_node_embeddings, final_rel_embeddings

# =============================================================================
# GAN MODELS
# =============================================================================

class Generator(nn.Module):
    """Progressive GAN Generator adapted for knowledge graphs"""
    
    def __init__(self, embed_dim=500, noise_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.noise_dim = noise_dim
        
        input_dim = embed_dim * 2 + noise_dim
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(512, embed_dim)
        )
        
        self.residual_proj = nn.Linear(input_dim, embed_dim)
        self.apply(self._gentle_init)
    
    def _gentle_init(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, h_emb, r_emb, noise=None):
        if noise is None:
            noise = torch.randn(h_emb.shape[0], self.noise_dim, device=h_emb.device) * 0.1
        
        x = torch.cat([h_emb, r_emb, noise], dim=-1)
        refined = self.layers(x)
        residual = self.residual_proj(x) * 0.02
        
        return refined + residual

class Discriminator(nn.Module):
    """Progressive GAN Discriminator adapted for knowledge graphs"""
    
    def __init__(self, embed_dim=500, hidden_dim=1024, dropout=0.3, use_skip=True):
        super().__init__()
        in_features = embed_dim * 3
        
        # Main layers
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.drop1 = nn.Dropout(dropout)
        
        # Residual block
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.drop2 = nn.Dropout(dropout)
        
        # Output layers
        self.fc3 = nn.Linear(hidden_dim, in_features)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        self.drop3 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(in_features, 1)
        
        # Skip connection
        self.use_skip = use_skip
        if use_skip:
            self.skip_linear = nn.Linear(in_features, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, head_emb, rel_emb, tail_emb):
        x_input = torch.cat([head_emb, rel_emb, tail_emb], dim=-1).float()
        if x_input.dim() > 2:
            x_input = x_input.view(x_input.size(0), -1)
        
        # Main path
        x = self.fc1(x_input)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        # Residual block
        res = self.fc2(x)
        res = self.bn2(res)
        res = self.act2(res)
        res = self.drop2(res)
        res = res + x
        
        # Output
        x = self.fc3(res)
        x = self.act3(x)
        x = self.drop3(x)
        out = self.fc_out(x).view(-1)
        
        # Skip connection
        if self.use_skip:
            skip = self.skip_linear(x_input).view(-1)
            out = out + skip
        
        return out

# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def check_early_stopping(current_metric, best_metric, patience_counter, patience_limit, verbose=True):
    """Check if early stopping should trigger"""
    if current_metric > best_metric + 1e-5:  
        return current_metric, 0, False  
    else:
        patience_counter += 1
        should_stop = patience_counter >= patience_limit
        if verbose and should_stop:
            print(f"Early stopping triggered: no improvement for {patience_limit} epochs")
        return best_metric, patience_counter, should_stop

def print_progress(message, verbose=True):
    """Conditional printing based on verbose mode"""
    if verbose:
        print(message)

class TierProgressManager:
    """Manages progress tracking and early stopping for each tier"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.current_tier = 0
        self.tier_names = ["R-GCN Preprocessing", "Tier 1: Warm-up", "Tier 2: RL Training", "Tier 3: Full Adversarial"]
        self.progress_bars = {}
    
    def start_tier(self, tier_idx, max_epochs):
        """Start progress tracking for a tier"""
        self.current_tier = tier_idx
        if not self.verbose and tier_idx < len(self.tier_names):
            desc = self.tier_names[tier_idx]
            self.progress_bars[tier_idx] = tqdm(total=max_epochs, desc=desc, unit="epoch")
    
    def update_tier(self, tier_idx, current_epoch, metrics=None):
        """Update progress for current tier"""
        if not self.verbose and tier_idx in self.progress_bars:
            if metrics:
                desc = f"{self.tier_names[tier_idx]} (Hit@10: {metrics.get('hit10', 0):.3f})"
                self.progress_bars[tier_idx].set_description(desc)
            self.progress_bars[tier_idx].update(1)
    
    def close_tier(self, tier_idx):
        """Close progress bar for completed tier"""
        if not self.verbose and tier_idx in self.progress_bars:
            self.progress_bars[tier_idx].close()
            del self.progress_bars[tier_idx]
    
    def close_all(self):
        """Close all progress bars"""
        for pbar in self.progress_bars.values():
            pbar.close()
        self.progress_bars.clear()

def distmult_score(h, r, t):
    """DistMult scoring function"""
    return (h * r * t).sum(dim=-1)

def compute_metrics(y_true, y_probs, threshold=0.5):
    """Compute classification metrics with version compatibility"""
    try:
        # Convert to numpy arrays safely
        if torch.is_tensor(y_true):
            y_true = safe_tensor_to_numpy(y_true)
        if torch.is_tensor(y_probs):
            y_probs = safe_tensor_to_numpy(y_probs)
        
        y_true = np.array(y_true, dtype=np.float64)
        y_probs = np.array(y_probs, dtype=np.float64)
        y_pred = (y_probs >= threshold).astype(int)
        
        # Handle edge cases
        if len(set(y_true)) <= 1:
            return {"F1": 0.0, "AUPR": 0.0, "MCC": 0.0, "AUC": 0.0}
        
        results = {}
        
        # F1 Score
        try:
            results["F1"] = float(f1_score(y_true, y_pred, zero_division=0))
        except Exception as e:
            print(f"Warning: F1 score calculation failed: {e}")
            results["F1"] = 0.0
        
        # AUPR
        try:
            results["AUPR"] = float(average_precision_score(y_true, y_probs))
        except Exception as e:
            print(f"Warning: AUPR calculation failed: {e}")
            results["AUPR"] = 0.0
        
        # MCC
        try:
            results["MCC"] = float(matthews_corrcoef(y_true, y_pred))
        except Exception as e:
            print(f"Warning: MCC calculation failed: {e}")
            results["MCC"] = 0.0
        
        # AUC
        try:
            if len(set(y_true)) > 1:
                results["AUC"] = float(roc_auc_score(y_true, y_probs))
            else:
                results["AUC"] = 0.0
        except Exception as e:
            print(f"Warning: AUC calculation failed: {e}")
            results["AUC"] = 0.0
        
        return results
        
    except Exception as e:
        print(f"Error in compute_metrics: {e}")
        return {"F1": 0.0, "AUPR": 0.0, "MCC": 0.0, "AUC": 0.0}

def get_hard_negatives(h_emb, r_emb, t_emb, node_emb, k=5):
    """Find hard negative samples by selecting nodes with high similarity to true tail"""
    with torch.no_grad():
        # Calculate similarity to all nodes
        sim = torch.matmul(F.normalize(t_emb, dim=-1), F.normalize(node_emb, dim=-1).T)
        
        # Get top-k similar nodes (hard negatives)
        _, hard_neg_idx = sim.topk(k, dim=1)
        
        # Sample one hard negative per example
        batch_size = h_emb.shape[0]
        selected_idx = torch.randint(0, k, (batch_size,), device=h_emb.device)
        final_neg_idx = torch.gather(hard_neg_idx, 1, selected_idx.unsqueeze(1)).squeeze(1)
        
        # Get embeddings for these hard negatives
        hard_neg_emb = node_emb[final_neg_idx]
        
        return hard_neg_emb

def generate_balanced_hard_negatives(h_emb, r_emb, node_emb, num_hard=10, num_medium=8, num_easy=7):
    """Generate balanced hard negatives for discriminator training"""
    batch_size = h_emb.shape[0]
    
    with torch.no_grad():
        # Calculate DistMult scores for all entities
        h_expand = h_emb.unsqueeze(1).expand(-1, node_emb.size(0), -1)
        r_expand = r_emb.unsqueeze(1).expand(-1, node_emb.size(0), -1)
        all_nodes = node_emb.unsqueeze(0).expand(batch_size, -1, -1)
        dm_scores = (h_expand * r_expand * all_nodes).sum(dim=-1)
        
        # Get different difficulty levels
        hard_vals, hard_idxs = dm_scores.topk(num_hard, dim=1)
        
        sorted_scores, sorted_idxs = dm_scores.sort(dim=1, descending=True)
        start_idx = node_emb.size(0) // 3
        end_idx = start_idx + num_medium
        medium_idxs = sorted_idxs[:, start_idx:end_idx]
        
        easy_idxs = torch.randint(0, node_emb.size(0), (batch_size, num_easy), device=h_emb.device)
        
        all_neg_idxs = torch.cat([hard_idxs, medium_idxs, easy_idxs], dim=1)
    
    return all_neg_idxs

def compute_composite_rl_loss(tier2_epoch, tier3_epoch, current_tier, h_emb, r_emb, fake, t_emb, discriminator, rl_start_epoch, full_system_epoch):
    """Three-tier RL loss progression with explicit tier tracking"""
    
    if current_tier == 1:
        return torch.tensor(0.0, device=h_emb.device), {}
    
    elif current_tier == 2:
        # Tier 2: DistMult-only RL
        if tier2_epoch < rl_start_epoch:
            return torch.tensor(0.0, device=h_emb.device), {}
        
        dm_scores = distmult_score(h_emb, r_emb, fake)
        dm_component = torch.sigmoid(dm_scores / 5.0)
        simple_reward = dm_component - 0.5
        rl_loss = -simple_reward.mean()
        return rl_loss, {'dm_component': dm_component.mean().item()}
    
    else:  # current_tier == 3
        # Tier 3: Full composite RL
        if tier3_epoch < full_system_epoch:
            # Still in transition, use DistMult only
            dm_scores = distmult_score(h_emb, r_emb, fake)
            dm_component = torch.sigmoid(dm_scores / 5.0)
            simple_reward = dm_component - 0.5
            rl_loss = -simple_reward.mean()
            return rl_loss, {'dm_component': dm_component.mean().item()}
        else:
            # Full composite system
            dm_scores = distmult_score(h_emb, r_emb, fake)
            dm_component = torch.sigmoid(dm_scores / 5.0)
            
            with torch.no_grad():
                disc_scores = discriminator(h_emb.detach(), r_emb.detach(), fake.detach())
                disc_component = torch.sigmoid(disc_scores)
            
            composite_reward = 0.3 * disc_component + 0.7 * dm_component - 0.5
            rl_loss = -composite_reward.mean()
            
            return rl_loss, {
                'disc_component': disc_component.mean().item(),
                'dm_component': dm_component.mean().item(),
                'composite_reward': composite_reward.mean().item()
            }

def bias_mitigation_loss(fake, t_emb, node_emb, h_emb, r_emb):
    """Hub bias mitigation through hard negative mining"""
    with torch.no_grad():
        sim = torch.matmul(F.normalize(t_emb, dim=-1), F.normalize(node_emb, dim=-1).T)
        _, hard_neg_idx = sim.topk(3, dim=1)
        selected_idx = torch.randint(0, 3, (h_emb.shape[0],), device=h_emb.device)
        final_neg_idx = torch.gather(hard_neg_idx, 1, selected_idx.unsqueeze(1)).squeeze(1)
        hard_neg_emb = node_emb[final_neg_idx]
    
    pos_cos = F.cosine_similarity(fake, t_emb, dim=-1)
    hard_neg_cos = F.cosine_similarity(fake, hard_neg_emb, dim=-1)
    margin_loss = F.relu(0.2 + hard_neg_cos - pos_cos).mean()
    
    batch_var = torch.var(fake, dim=0).mean()
    diversity_loss = torch.exp(-2 * batch_var)
    
    return margin_loss + 0.1 * diversity_loss

def rebalance_training_schedule(epoch, disc_f1):
    """Adjust training frequency based on dominance"""
    if disc_f1 > 0.9:
        d_update_freq = 50
        g_steps_per_d = 10
    elif disc_f1 > 0.85:
        d_update_freq = 20
        g_steps_per_d = 5
    else:
        d_update_freq = 10
        g_steps_per_d = 1
    
    return d_update_freq, g_steps_per_d

def evaluate_hit_at_k(data_loader, generator, node_emb, rel_emb, device, hit_at_k_list):
    """Evaluate Hit@K metrics with safety check for empty data"""
    generator.eval()
    hits_dict = {k: 0 for k in hit_at_k_list}
    total_examples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            h, r, t = [b.to(device) for b in batch.T]
            h_emb, r_emb_batch, t_emb = node_emb[h], rel_emb(r), node_emb[t]
            pred = generator(h_emb, r_emb_batch)
            
            norm_pred = F.normalize(pred, dim=-1)
            norm_nodes = F.normalize(node_emb, dim=-1)
            sims = torch.matmul(norm_pred, norm_nodes.T)
            
            for k in hit_at_k_list:
                topk = sims.topk(k, dim=1).indices
                for i in range(len(t)):
                    if t[i].item() in topk[i]:
                        hits_dict[k] += 1
            total_examples += len(t)
    
    # Safety check for division by zero
    if total_examples == 0:
        hit_at_k_ratios = {k: 0.0 for k in hit_at_k_list}
    else:
        hit_at_k_ratios = {k: hits_dict[k] / total_examples for k in hit_at_k_list}
    
    generator.train()
    return hit_at_k_ratios

def validate(val_loader, generator, discriminator, node_emb, rel_emb, device):
    """Run validation metrics"""
    generator.eval()
    discriminator.eval()
    
    val_true_labels, val_pred_probs = [], []
    val_cos_sim = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            h, r, t = [b.to(device) for b in batch.T]
            head_emb, relations_emb, tail_embed = node_emb[h], rel_emb(r), node_emb[t]
            fake = generator(head_emb, relations_emb)
            
            real_score = discriminator(head_emb, relations_emb, tail_embed)
            fake_score = discriminator(head_emb, relations_emb, fake)
            
            val_true_labels.extend([1]*len(real_score) + [0]*len(fake_score))
            val_pred_probs.extend(torch.sigmoid(torch.cat([real_score, fake_score])).cpu().numpy())
            val_cos_sim += F.cosine_similarity(fake, tail_embed).mean().item()
    
    val_metrics = compute_metrics(np.array(val_true_labels), np.array(val_pred_probs))
    val_cos_avg = val_cos_sim / len(val_loader) if len(val_loader) > 0 else 0.0
    
    generator.train()
    discriminator.train()
    return val_metrics, val_cos_avg

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_prot_b_gan(args):
    """Main training pipeline for Prot-B-GAN with fixed issues"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.verbose:
        print(f" Prot-B-GAN: Progressive Adversarial Training for Knowledge Graph Completion")
        print(f"Device: {device}")
        print(f"Parameters: embed_dim={args.embed_dim}, epochs={args.epochs}, batch_size={args.batch_size}")
        print(f"Embedding init: {args.embedding_init}")
        print(f"{'='*70}")
    else:
        print(f" Prot-B-GAN Training Started (Device: {device})")
    
    # Initialize progress manager
    progress_manager = TierProgressManager(verbose=args.verbose)
    
    try:
        # =========================================================================
        # Data Loading and Preprocessing
        # =========================================================================
        
        # Load and process data
        train_df, val_df, test_df = load_data(
            args.data_root, args.train_file, args.val_file, args.test_file,
            debug_mode=args.debug, max_train=20000 if args.debug else None,
            max_val=4000 if args.debug else None, max_test=4000 if args.debug else None
        )
        train_df, val_df, test_df, num_entities, num_relations, entity_to_id, relation_to_id = create_entity_relation_mappings(
            train_df, val_df, test_df
        )
        
        # Create data loaders
        train_triples = list(zip(train_df["H"], train_df["R"], train_df["T"]))
        val_triples = list(zip(val_df["H"], val_df["R"], val_df["T"]))
        test_triples = list(zip(test_df["H"], test_df["R"], test_df["T"]))
        
        train_loader = DataLoader(TripletDataset(train_triples), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TripletDataset(val_triples), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(TripletDataset(test_triples), batch_size=args.batch_size, shuffle=False)
        
        # Modular embedding initialization
        embedding_initializer = create_embedding_initializer(args)
        print_progress(f"Using {embedding_initializer.get_name()} embedding initialization", args.verbose)
        
        final_node_embeddings, final_rel_embeddings = embedding_initializer.initialize_embeddings(
            train_df, val_df, test_df, num_entities, num_relations, device, progress_manager, args.verbose
        )
        
        # Initialize embeddings
        node_emb = nn.Parameter(final_node_embeddings.clone().detach()).to(device)
        rel_emb = nn.Embedding(num_relations, args.embed_dim).to(device)
        rel_emb.weight.data.copy_(final_rel_embeddings)
        
        # =========================================================================
        # Model Initialization
        # =========================================================================
        
        generator = Generator(args.embed_dim, args.noise_dim).to(device)
        discriminator = Discriminator(args.embed_dim, args.hidden_dim).to(device)
        
        g_opt = optim.Adam(list(generator.parameters()) + [node_emb], lr=args.g_lr)
        d_opt = optim.Adam(discriminator.parameters(), lr=args.d_lr)
        
        print_progress(f" Model Architecture:", args.verbose)
        print_progress(f"   Generator parameters: {sum(p.numel() for p in generator.parameters()):,}", args.verbose)
        print_progress(f"   Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}", args.verbose)
        
        # =========================================================================
        # Training Metrics Tracking
        # =========================================================================
        
        # Training history
        training_history = {
            'losses': [], 'cos_sims': [], 'd_losses': [], 'g_losses': [],
            'f1_history': [], 'aupr_history': [], 'mcc_history': [], 'auc_history': [],
            'val_f1_history': [], 'val_aupr_history': [], 'val_mcc_history': [], 'val_auc_history': [],
            'train_hitks_dict': {k: [] for k in args.hit_at_k},
            'val_hitks_dict': {k: [] for k in args.hit_at_k},
            'real_acc_list': [], 'fake_acc_list': [],
            'collapse_hist': deque(maxlen=5), 'diversity_hist': deque(maxlen=5),
            'tier_transitions': []  
        }
        
        # Training state with early stopping for each tier
        best_val_hit10 = 0.0
        best_epoch = 0
        
        # Tier-specific early stopping and epoch tracking
        tier1_best = 0.0
        tier1_patience = 0
        tier2_best = 0.0  
        tier2_patience = 0
        tier3_best = 0.0
        tier3_patience = 0
        
        # Explicit tier epoch counters (FIXED: Simplified tier tracking)
        tier1_start_epoch = 1
        tier2_start_epoch = None
        tier3_start_epoch = None
        tier2_epoch_count = 0
        tier3_epoch_count = 0
        
        best_pretrain_state = None
        current_tier = 1
        
        bce_loss = nn.BCEWithLogitsLoss()
        
        # Dynamic training schedule variables
        current_d_freq = args.d_update_freq
        current_g_steps = 1
        g_step_counter = 0
        
        print_progress(f"\n Starting Three-Tier Progressive Training", args.verbose)
        print_progress(f"Tier 1: Enhanced Warm-up (up to {args.pretrain_epochs} epochs)", args.verbose)
        print_progress(f"Tier 2: Structure-aware RL (patience: {args.tier2_patience})", args.verbose)
        print_progress(f"Tier 3: Full Adversarial (patience: {args.tier3_patience})", args.verbose)
        print_progress(f"{'='*70}", args.verbose)
        
        # =========================================================================
        # Three-Tier Training Loop with Comprehensive Early Stopping
        # =========================================================================
        
        for epoch in range(1, args.epochs + 1):
            
            # Initialize rl_metrics at epoch level (FIXED: Variable scope issue)
            rl_metrics = {}
            
            # =================================================================
            # TIER 1: Enhanced Reconstruction Warm-up with Early Stopping
            # =================================================================
            
            if current_tier == 1:
                if epoch == 1:
                    progress_manager.start_tier(1, args.pretrain_epochs)
                
                generator.train()
                
                for step, batch in enumerate(train_loader):
                    h, r, t = [b.to(device) for b in batch.T]
                    h_emb, r_emb_batch, t_emb = node_emb[h], rel_emb(r), node_emb[t]
                    
                    # Sample negatives
                    batch_size = h_emb.size(0)
                    neg_idx = torch.randint(0, node_emb.size(0), (batch_size, args.n_pre_neg), device=device)
                    neg_t_embs = node_emb[neg_idx]
                    
                    # Generator forward with reconstruction + margin loss
                    g_opt.zero_grad()
                    fake = generator(h_emb, r_emb_batch)
                    rec_loss = F.mse_loss(fake, t_emb) + 0.1 * (1 - F.cosine_similarity(fake, t_emb).mean())
                    
                    fake_exp = fake.unsqueeze(1).expand(-1, args.n_pre_neg, -1)
                    pos_cos = F.cosine_similarity(fake, t_emb, dim=-1)
                    neg_cos = F.cosine_similarity(fake_exp, neg_t_embs, dim=-1)
                    margin_term = 0.35 + neg_cos - pos_cos.unsqueeze(1)
                    margin_loss = F.relu(margin_term).mean()
                    
                    pretrain_loss = rec_loss + args.alpha_pretrain * margin_loss
                    pretrain_loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_norm)
                    g_opt.step()
                
                # Validate pretrain progress
                val_hitks = evaluate_hit_at_k(val_loader, generator, node_emb, rel_emb, device, args.hit_at_k)
                this_hit10 = val_hitks[10]
                
                # Early stopping for Tier 1
                tier1_best, tier1_patience, should_stop_tier1 = check_early_stopping(
                    this_hit10, tier1_best, tier1_patience, args.pretrain_patience, args.verbose
                )
                
                if this_hit10 > best_val_hit10:
                    best_val_hit10 = this_hit10
                    best_pretrain_state = {
                        "generator": generator.state_dict(),
                        "node_emb": node_emb.detach().cpu().clone()
                    }
                
                print_progress(f"[Tier 1] Epoch {epoch}: Hit@10 = {this_hit10:.4f} (best: {tier1_best:.4f})", args.verbose)
                progress_manager.update_tier(1, epoch, {'hit10': this_hit10})
                
                # Check if should transition to Tier 2
                if should_stop_tier1 or epoch >= args.pretrain_epochs:
                    print_progress(f"[Tier 1] Complete at epoch {epoch} - transitioning to Tier 2", args.verbose)
                    progress_manager.close_tier(1)
                    
                    # Load best pretrain state
                    if best_pretrain_state:
                        generator.load_state_dict(best_pretrain_state["generator"])
                        node_emb.data.copy_(best_pretrain_state["node_emb"].to(device))
                    
                    node_emb.requires_grad_(False)
                    g_opt = optim.Adam(generator.parameters(), lr=1e-4)
                    current_tier = 2
                    tier2_start_epoch = epoch + 1  # FIXED: Explicit tier tracking
                    training_history['tier_transitions'].append(('Tier 2', epoch + 1))
                    
                    print_progress(">> Transitioning to Tier 2: Structure-aware RL", args.verbose)
                continue
            
            # =================================================================
            # TIER 2 & 3: RL + Adversarial Training with Early Stopping
            # =================================================================
            
            # FIXED: Simplified progress tracking
            if current_tier == 2 and epoch == tier2_start_epoch:
                progress_manager.start_tier(2, args.tier2_patience)
                tier2_epoch_count = 0
            elif current_tier == 3 and epoch == tier3_start_epoch:
                progress_manager.start_tier(3, args.tier3_patience)
                tier3_epoch_count = 0
            
            # Update epoch counters
            if current_tier == 2:
                tier2_epoch_count += 1
            elif current_tier == 3:
                tier3_epoch_count += 1
            
            generator.train()
            discriminator.train()
            
            total_loss, total_cos = 0.0, 0.0
            total_d_loss, total_g_loss = 0.0, 0.0
            true_labels, pred_probs = [], []
            
            # Training loop for current tier
            for step, batch in enumerate(train_loader):
                if args.debug and args.max_debug_steps and step >= args.max_debug_steps:
                    break
                
                h, r, t = [b.to(device) for b in batch.T]
                h_emb, r_emb_batch, t_emb = node_emb[h], rel_emb(r), node_emb[t]
                
                # =============================================================
                # Discriminator Training (Tier 3 only) with Dynamic Rebalancing
                # =============================================================
                
                if current_tier == 3 and step % current_d_freq == 0 and g_step_counter >= current_g_steps:
                    d_opt.zero_grad()
                    
                    # Real samples
                    real_scores = discriminator(h_emb.detach(), r_emb_batch.detach(), t_emb.detach())
                    
                    # Generator fake samples
                    with torch.no_grad():
                        fake_samples = generator(h_emb.detach(), r_emb_batch.detach())
                    fake_scores = discriminator(h_emb.detach(), r_emb_batch.detach(), fake_samples)
                    
                    # Hard negative samples
                    neg_indices = generate_balanced_hard_negatives(h_emb.detach(), r_emb_batch.detach(), node_emb)
                    batch_size = h_emb.shape[0]
                    selected_neg_idx = torch.randint(0, neg_indices.shape[1], (batch_size,), device=h_emb.device)
                    final_neg_idx = torch.gather(neg_indices, 1, selected_neg_idx.unsqueeze(1)).squeeze(1)
                    hard_neg_samples = node_emb[final_neg_idx]
                    hard_neg_scores = discriminator(h_emb.detach(), r_emb_batch.detach(), hard_neg_samples)
                    
                    # Soft labels
                    real_labels = torch.full_like(real_scores, 0.8)
                    fake_labels = torch.full_like(fake_scores, 0.2)
                    hard_neg_labels = torch.full_like(hard_neg_scores, 0.1)
                    
                    # Discriminator loss
                    d_loss = (bce_loss(real_scores, real_labels) +
                             bce_loss(fake_scores, fake_labels) +
                             bce_loss(hard_neg_scores, hard_neg_labels)) / 3
                    
                    d_loss.backward()
                    d_opt.step()
                    total_d_loss += d_loss.item()
                    
                    # Reset generator counter (FIXED: Removed redundant reset)
                    g_step_counter = 0
                    
                    # Track discriminator performance
                    with torch.no_grad():
                        all_real_scores = real_scores
                        all_fake_scores = torch.cat([fake_scores, hard_neg_scores])
                        
                        true_labels.extend([1]*len(all_real_scores) + [0]*len(all_fake_scores))
                        pred_probs.extend(torch.sigmoid(torch.cat([all_real_scores, all_fake_scores])).cpu().numpy())
                
                # =============================================================
                # Generator Training with Progressive Complexity
                # =============================================================
                
                g_opt.zero_grad()
                fake = generator(h_emb, r_emb_batch)
                
                loss_components = []
                
                # Refinement loss
                refinement_loss = F.mse_loss(fake, t_emb)
                loss_components.append(args.refinement_weight * refinement_loss)
                
                # DistMult loss
                dm_scores = distmult_score(h_emb, r_emb_batch, fake)
                dm_loss = -torch.tanh(dm_scores / 10.0).mean()
                loss_components.append(args.distmult_weight * dm_loss)
                
                # RL loss (starts in Tier 2) - FIXED: Simplified epoch calculation
                if current_tier >= 2:
                    rl_loss, rl_metrics = compute_composite_rl_loss(
                        tier2_epoch_count, tier3_epoch_count, current_tier, 
                        h_emb, r_emb_batch, fake, t_emb, discriminator, 
                        args.rl_start_epoch, args.full_system_epoch
                    )
                    if rl_loss.item() != 0:
                        loss_components.append(args.rl_weight * rl_loss)
                
                # Bias mitigation
                bias_loss = bias_mitigation_loss(fake, t_emb, node_emb, h_emb, r_emb_batch)
                loss_components.append(args.bias_weight * bias_loss)
                
                # Adversarial loss (Tier 3 only)
                if current_tier == 3:
                    adv_scores = discriminator(h_emb, r_emb_batch, fake)
                    adv_loss = -torch.tanh(adv_scores / 5.0).mean()
                    loss_components.append(args.adv_weight * adv_loss)
                
                # Cosine margin loss with hard negatives
                batch_size = h_emb.shape[0]
                with torch.no_grad():
                    h_expand = h_emb.unsqueeze(1).expand(-1, node_emb.size(0), -1)
                    r_expand = r_emb_batch.unsqueeze(1).expand(-1, node_emb.size(0), -1)
                    all_nodes = node_emb.unsqueeze(0).expand(batch_size, -1, -1)
                    dm_scores_gen = (h_expand * r_expand * all_nodes).sum(dim=-1)
                    hard_vals, hard_idxs = dm_scores_gen.topk(args.hard_neg_k, dim=1)
                
                rand_k = args.n_neg - args.hard_neg_k
                rand_idxs = torch.randint(0, node_emb.size(0), (batch_size, rand_k), device=device)
                neg_indices_gen = torch.cat([hard_idxs, rand_idxs], dim=1)
                neg_t_embs = node_emb[neg_indices_gen]
                
                fake_expanded = fake.unsqueeze(1).expand(-1, args.n_neg, -1)
                pos_cos = F.cosine_similarity(fake, t_emb, dim=-1)
                neg_cos = F.cosine_similarity(fake_expanded, neg_t_embs, dim=-1)
                margin_term = 0.30 + neg_cos - pos_cos.unsqueeze(1)
                cos_margin = F.relu(margin_term).mean()
                
                # L2 and diversity losses
                l2 = (fake - t_emb.detach()).pow(2).mean()
                fake_std = fake.std(dim=0).mean()
                diversity_loss = 0.2 * torch.exp(-5 * fake_std)
                
                # Final generator loss
                final_gen_loss = sum(loss_components) + args.g_guidance_weight * cos_margin + args.l2_reg_weight * l2 + diversity_loss
                final_gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_norm)
                g_opt.step()
                total_g_loss += final_gen_loss.item()
                
                # Track generator steps for dynamic scheduling
                g_step_counter += 1
                
                # Track training metrics
                with torch.no_grad():
                    cos_sim = F.cosine_similarity(fake, t_emb).mean().item()
                    total_loss += final_gen_loss.item()
                    total_cos += cos_sim
            
            # =================================================================
            # Epoch-level Evaluation and Early Stopping Check
            # =================================================================
            
            effective_steps = min(len(train_loader), args.max_debug_steps if args.debug and args.max_debug_steps else float("inf"))
            if effective_steps == 0:
                avg_loss = avg_cos = avg_d_loss = avg_g_loss = 0.0
            else:
                avg_loss = total_loss / effective_steps
                avg_cos = total_cos / effective_steps
                avg_d_loss = total_d_loss / max(1, effective_steps // current_d_freq) if current_tier == 3 else 0.0
                avg_g_loss = total_g_loss / effective_steps
            
            # Update training history
            training_history['losses'].append(avg_loss)
            training_history['cos_sims'].append(avg_cos)
            training_history['d_losses'].append(avg_d_loss)
            training_history['g_losses'].append(avg_g_loss)
            
            # Compute discriminator metrics
            if true_labels and current_tier == 3:
                disc_metrics = compute_metrics(np.array(true_labels), np.array(pred_probs))
                training_history['f1_history'].append(disc_metrics['F1'])
                training_history['aupr_history'].append(disc_metrics['AUPR'])
                training_history['mcc_history'].append(disc_metrics['MCC'])
                training_history['auc_history'].append(disc_metrics['AUC'])
                
                # Rebalance training schedule based on discriminator performance
                current_d_freq, current_g_steps = rebalance_training_schedule(epoch, disc_metrics['F1'])
                
            else:
                disc_metrics = {'F1': 0, 'AUPR': 0, 'MCC': 0, 'AUC': 0}
                # Use default values when no metrics available
                current_d_freq, current_g_steps = rebalance_training_schedule(epoch, 0.0)
            
            # Hit@K evaluation
            train_hit_at_k = evaluate_hit_at_k(train_loader, generator, node_emb, rel_emb, device, args.hit_at_k)
            for k in args.hit_at_k:
                training_history['train_hitks_dict'][k].append(train_hit_at_k[k])
            
            # Validation
            val_metrics, val_cos_avg = validate(val_loader, generator, discriminator, node_emb, rel_emb, device)
            training_history['val_f1_history'].append(val_metrics['F1'])
            training_history['val_aupr_history'].append(val_metrics['AUPR'])
            training_history['val_mcc_history'].append(val_metrics['MCC'])
            training_history['val_auc_history'].append(val_metrics['AUC'])
            
            val_hit_at_k = evaluate_hit_at_k(val_loader, generator, node_emb, rel_emb, device, args.hit_at_k)
            for k in args.hit_at_k:
                training_history['val_hitks_dict'][k].append(val_hit_at_k[k])
            
            current_hit10 = val_hit_at_k[10]
            
            # Early stopping check for current tier
            should_transition = False
            
            if current_tier == 2:
                tier2_best, tier2_patience, should_stop_tier2 = check_early_stopping(
                    current_hit10, tier2_best, tier2_patience, args.tier2_patience, args.verbose
                )
                if should_stop_tier2:
                    print_progress(f"[Tier 2] Early stopping at epoch {epoch} - transitioning to Tier 3", args.verbose)
                    progress_manager.close_tier(2)
                    current_tier = 3
                    tier3_start_epoch = epoch + 1  # FIXED: Explicit tier tracking
                    training_history['tier_transitions'].append(('Tier 3', epoch + 1))
                    should_transition = True
            
            elif current_tier == 3:
                tier3_best, tier3_patience, should_stop_tier3 = check_early_stopping(
                    current_hit10, tier3_best, tier3_patience, args.tier3_patience, args.verbose
                )
                if should_stop_tier3:
                    print_progress(f"[Tier 3] Early stopping at epoch {epoch} - training complete", args.verbose)
                    progress_manager.close_tier(3)
                    break
            
            # Update global best
            if current_hit10 > best_val_hit10:
                best_val_hit10 = current_hit10
                best_epoch = epoch
            
            # Progress logging
            tier_name = f"Tier {current_tier}"
            if current_tier == 2:
                tier_metrics = {'hit10': current_hit10, 'best': tier2_best, 'patience': tier2_patience}
            else:
                tier_metrics = {'hit10': current_hit10, 'best': tier3_best, 'patience': tier3_patience}
            
            if args.verbose:
                rl_info = ""
                if current_tier >= 2 and rl_metrics:  # FIXED: Check if rl_metrics has content
                    if current_tier == 2:
                        rl_info = f" | RL_DM: {rl_metrics.get('dm_component', 0):.3f}"
                    else:
                        rl_info = f" | RL_Disc: {rl_metrics.get('disc_component', 0):.3f} RL_DM: {rl_metrics.get('dm_component', 0):.3f}"
                
                schedule_info = f" | D_freq:{current_d_freq} G_steps:{current_g_steps}" if current_tier == 3 else ""
                print_progress(f"[{tier_name}] E{epoch:03d} | Loss {avg_loss:.4f} | CosSim {avg_cos:.3f} | "
                              f"F1 {disc_metrics['F1']:.4f}{rl_info}{schedule_info}", args.verbose)
                
                print_progress("Train: " + " ".join([f"Hit@{k}: {train_hit_at_k[k]:.4f}" for k in args.hit_at_k]), args.verbose)
                print_progress("Val:   " + " ".join([f"Hit@{k}: {val_hit_at_k[k]:.4f}" for k in args.hit_at_k]) + f" | F1 {val_metrics['F1']:.4f}", args.verbose)
            
            # Update progress bar
            progress_manager.update_tier(current_tier, epoch, tier_metrics)
            
            # Tier transition notifications
            if should_transition:
                print_progress(f" TIER 3: Full composite RL + Adversarial system activated", args.verbose)
            
            # =================================================================
            # Checkpointing
            # =================================================================
            
            if current_hit10 > best_val_hit10 - 1e-5:  
                checkpoint = {
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "node_emb": node_emb.detach().cpu(),
                    "rel_emb": rel_emb.state_dict(),
                    "g_opt_state": g_opt.state_dict(),
                    "d_opt_state": d_opt.state_dict(),
                    "epoch": epoch,
                    "current_tier": current_tier,
                    "best_val_hit10": best_val_hit10,
                    "best_epoch": best_epoch,
                    "args": vars(args),
                    "entity_to_id": entity_to_id,
                    "relation_to_id": relation_to_id,
                    "training_history": training_history,
                    "final_rl_metrics": rl_metrics,
                    "final_disc_metrics": disc_metrics
                }
                
                os.makedirs(args.output_dir, exist_ok=True)
                checkpoint_path = os.path.join(args.output_dir, "best_prot_b_gan_checkpoint.pt")
                torch.save(checkpoint, checkpoint_path)
                
                if args.verbose and current_hit10 > best_val_hit10 - 1e-5:
                    print_progress(f" Saved checkpoint (Val Hit@10: {current_hit10:.4f}) at epoch {epoch}", args.verbose)
        
        # =========================================================================
        # Final Evaluation
        # =========================================================================
        
        progress_manager.close_all()
        
        print(f"\n TRAINING COMPLETED!")
        print(f"Best validation Hit@10: {best_val_hit10:.4f} achieved at epoch {best_epoch}")
        print(f"Final tier reached: {current_tier}")
        print(f"{'='*70}")
        
        # Load best model for final evaluation (FIXED: Complete checkpoint loading)
        checkpoint_path = os.path.join(args.output_dir, "best_prot_b_gan_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            generator.load_state_dict(checkpoint['generator'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            # FIXED: Load all components
            node_emb.data.copy_(checkpoint['node_emb'].to(device))
            rel_emb.load_state_dict(checkpoint['rel_emb'])
        
        # Final test evaluation
        test_hit_at_k = evaluate_hit_at_k(test_loader, generator, node_emb, rel_emb, device, args.hit_at_k)
        final_val_metrics, _ = validate(val_loader, generator, discriminator, node_emb, rel_emb, device)
        
        print(f"\n FINAL RESULTS:")
        print(f"Test Hit@1:  {test_hit_at_k[1]:.4f} ({test_hit_at_k[1]*100:.1f}%)")
        print(f"Test Hit@5:  {test_hit_at_k[5]:.4f} ({test_hit_at_k[5]*100:.1f}%)")
        print(f"Test Hit@10: {test_hit_at_k[10]:.4f} ({test_hit_at_k[10]*100:.1f}%)")
        print(f"Final Discriminator F1: {final_val_metrics['F1']:.4f}")
        
        # Save final results
        final_results = {
            "test_hit_at_k": test_hit_at_k,
            "final_val_metrics": final_val_metrics,
            "training_summary": {
                "total_epochs_trained": epoch,
                "best_epoch": best_epoch,
                "best_val_hit10": best_val_hit10,
                "final_test_hit10": test_hit_at_k[10],
                "final_tier_reached": current_tier,
                "tier_transitions": training_history['tier_transitions']
            }
        }
        
        results_path = os.path.join(args.output_dir, "final_results.pt")
        torch.save(final_results, results_path)
        
        print(f"\n All results saved to: {args.output_dir}")
        print(f"   - best_prot_b_gan_checkpoint.pt: Complete model checkpoint")
        print(f"   - final_results.pt: Final evaluation metrics")
        
        return training_history, final_results
    
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        progress_manager.close_all()
        return None, None
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        progress_manager.close_all()
        raise

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    # Check environment compatibility first
    if not check_environment():
        print("Environment check failed. Please install required packages.")
        return 1
    
    parser = argparse.ArgumentParser(description='Prot-B-GAN: Progressive Adversarial Training for Knowledge Graph Completion')
    
    # Data and I/O
    parser.add_argument('--data_root', type=str, required=True, help='Root directory containing data files')
    parser.add_argument('--train_file', type=str, default='train.csv', help='Training data file (CSV)')
    parser.add_argument('--val_file', type=str, default='val.csv', help='Validation data file (CSV)')
    parser.add_argument('--test_file', type=str, default='test.csv', help='Test data file (CSV)')
    parser.add_argument('--output_dir', type=str, default='./prot_b_gan_output', help='Output directory for results')
    
    # Model Architecture
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--noise_dim', type=int, default=64, help='Noise dimension for generator')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Hidden dimension for discriminator')
    
    # Training Parameters
    parser.add_argument('--epochs', type=int, default=30, help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--g_lr', type=float, default=1e-3, help='Generator learning rate')
    parser.add_argument('--d_lr', type=float, default=5e-5, help='Discriminator learning rate')
    
    # Embedding Initialization
    parser.add_argument('--embedding_init', type=str, default='rgcn', 
                        choices=['rgcn', 'random', 'transe', 'distmult', 'complex'],
                        help='Embedding initialization method')
    
    # R-GCN Parameters
    parser.add_argument('--rgcn_epochs', type=int, default=50, help='R-GCN preprocessing epochs')
    parser.add_argument('--rgcn_lr', type=float, default=0.01, help='R-GCN learning rate')
    parser.add_argument('--rgcn_layers', type=int, default=2, help='Number of R-GCN layers')
    parser.add_argument('--rgcn_early_stopping_patience', type=int, default=10, help='R-GCN early stopping patience')
    
    # TransE Parameters
    parser.add_argument('--transe_epochs', type=int, default=100, help='TransE epochs')
    parser.add_argument('--transe_lr', type=float, default=0.01, help='TransE learning rate')
    parser.add_argument('--transe_margin', type=float, default=1.0, help='TransE margin')
    
    # DistMult Parameters
    parser.add_argument('--distmult_epochs', type=int, default=100, help='DistMult epochs')
    parser.add_argument('--distmult_lr', type=float, default=0.01, help='DistMult learning rate')
    parser.add_argument('--distmult_regularization', type=float, default=0.01, help='DistMult L2 regularization')
    
    # ComplEx Parameters
    parser.add_argument('--complex_epochs', type=int, default=100, help='ComplEx epochs')
    parser.add_argument('--complex_lr', type=float, default=0.01, help='ComplEx learning rate')
    parser.add_argument('--complex_regularization', type=float, default=0.01, help='ComplEx L2 regularization')
    
    # Three-Tier System
    parser.add_argument('--pretrain_epochs', type=int, default=200, help='Tier 1 pretraining epochs')
    parser.add_argument('--pretrain_patience', type=int, default=20, help='Tier 1 early stopping patience')
    parser.add_argument('--rl_start_epoch', type=int, default=20, help='Epoch to start RL training')
    parser.add_argument('--full_system_epoch', type=int, default=25, help='Epoch to start full adversarial training')
    
    # Early Stopping
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='General early stopping patience')
    parser.add_argument('--tier2_patience', type=int, default=20, help='Tier 2 early stopping patience')
    parser.add_argument('--tier3_patience', type=int, default=25, help='Tier 3 early stopping patience')
    
    # Loss Weights
    parser.add_argument('--g_guidance_weight', type=float, default=2.0, help='Generator guidance weight')
    parser.add_argument('--l2_reg_weight', type=float, default=0.1, help='L2 regularization weight')
    parser.add_argument('--refinement_weight', type=float, default=0.7, help='Refinement loss weight')
    parser.add_argument('--distmult_weight', type=float, default=0.2, help='DistMult component weight')
    parser.add_argument('--rl_weight', type=float, default=0.1, help='RL loss weight')
    parser.add_argument('--bias_weight', type=float, default=0.1, help='Bias mitigation weight')
    parser.add_argument('--adv_weight', type=float, default=0.05, help='Adversarial loss weight')
    
    # Hard Negative Mining
    parser.add_argument('--n_neg', type=int, default=50, help='Number of negative samples')
    parser.add_argument('--hard_neg_k', type=int, default=50, help='Number of hard negatives')
    parser.add_argument('--n_pre_neg', type=int, default=30, help='Number of pretraining negatives')
    parser.add_argument('--alpha_pretrain', type=float, default=1.5, help='Pretraining alpha weight')
    
    # Training Schedule
    parser.add_argument('--d_update_freq', type=int, default=10, help='Discriminator update frequency')
    parser.add_argument('--hit_at_k', type=int, nargs='+', default=[1, 5, 10], help='Hit@K values to evaluate')
    
    # Debug and Misc
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with smaller datasets')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output (default: progress bars only)')
    parser.add_argument('--skip_env_check', action='store_true', help='Skip environment compatibility check')
    parser.add_argument('--max_debug_steps', type=int, default=200, help='Max steps in debug mode')
    
    args = parser.parse_args()
    
    # Apply debug mode settings
    if args.debug:
        args.epochs = min(args.epochs, 40)
        args.pretrain_epochs = min(args.pretrain_epochs, 20)
        args.rgcn_epochs = min(args.rgcn_epochs, 10)
        args.early_stopping_patience = 5
        args.tier2_patience = 5
        args.tier3_patience = 8
        print("DEBUG MODE ENABLED - Reduced epochs and patience")
    
    if not args.verbose:
        print("Running in quiet mode - progress bars will be shown instead of detailed output")
        print("Use --verbose flag for detailed training output")
    
    # Additional environment setup for Google Colab
    if 'google.colab' in sys.modules:
        print("🔬 Google Colab detected - applying Colab-specific optimizations")
        # Reduce batch size if needed for Colab memory limits
        if args.batch_size > 32 and not args.debug:
            args.batch_size = 32
            print(f"   Reduced batch size to {args.batch_size} for Colab compatibility")
        
        # Set matplotlib backend for Colab
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for Colab
        except:
            pass
    
    # Verify data files exist
    data_files = [args.train_file, args.val_file, args.test_file]
    for file in data_files:
        full_path = os.path.join(args.data_root, file)
        if not os.path.exists(full_path):
            print(f"Data file not found: {full_path}")
            print(f"   Please ensure all data files exist in {args.data_root}")
            return 1
    
    print(f"All data files found in {args.data_root}")
    
    try:
        # Run training
        training_history, final_results = train_prot_b_gan(args)
        
        if training_history and final_results:
            print(f"\n Training completed successfully!")
            print(f"Final Test Hit@10: {final_results['test_hit_at_k'][10]*100:.1f}%")
            
            if final_results['training_summary']['tier_transitions']:
                print(f"Tier transitions: {final_results['training_summary']['tier_transitions']}")
            print(f"Best performance reached at epoch {final_results['training_summary']['best_epoch']}")
            print(f"Results saved to: {args.output_dir}")
            
            # Create a simple visualization if possible
            try:
                plt.figure(figsize=(10, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(training_history['val_hitks_dict'][10])
                plt.title('Validation Hit@10 Progress')
                plt.xlabel('Epoch')
                plt.ylabel('Hit@10')
                plt.grid(True)
                
                plt.subplot(1, 2, 2) 
                plt.plot(training_history['losses'])
                plt.title('Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, 'training_progress.png'), dpi=150, bbox_inches='tight')
                print(f"Training plots saved to: {os.path.join(args.output_dir, 'training_progress.png')}")
                
                if args.verbose:
                    plt.show()
                
            except Exception as e:
                print(f"Note: Could not create visualization plots: {e}")
            
        else:
            print(f"Training was interrupted or failed")
            return 1
            
    except Exception as e:
        print(f"Training failed with error: {e}")
        print(f"Try running with --debug --verbose for more detailed error information")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # Quick installation helper 
    if len(sys.argv) > 1 and sys.argv[1] == '--install':
        print(" Installing dependencies for Google Colab...")
        os.system('pip install "numpy<2.0"')
        os.system('pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
        os.system('pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib -f https://data.pyg.org/whl/torch-2.0.0+cu118.html')
        os.system('pip install torch-geometric')
        os.system('pip install scikit-learn pandas matplotlib tqdm')
        print("Installation complete! You can now run the training script.")
        sys.exit(0)
    
    sys.exit(main())
