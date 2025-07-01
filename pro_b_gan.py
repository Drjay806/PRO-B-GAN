"""
Prot-B-GAN: Clean Architecture for Knowledge Graph Completion
============================================================

A logically consistent pipeline that only does what makes sense:
- Smart warm-up logic (only when changing objectives)
- Method-consistent scoring throughout
- Progressive GAN-RL training

Installation:
    pip install "numpy<2.0"
    pip install torch==2.0.0 torchvision torchaudio
    pip install torch-geometric scikit-learn pandas matplotlib tqdm

Usage:
    python prot_b_gan_clean.py --data_root /path/to/data --embedding_init rgcn
    python prot_b_gan_clean.py --data_root /path/to/data --embedding_init transe
    python prot_b_gan_clean.py --data_root /path/to/data --embedding_init distmult
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
from sklearn.metrics import f1_score, average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.decomposition import PCA
import numpy as np
from collections import deque
import time
import random
from tqdm import tqdm
from abc import ABC, abstractmethod

# =============================================================================
# SCORING FUNCTIONS - Each method uses its own scoring throughout
# =============================================================================

def get_scoring_function(method):
    """Return method-specific scoring function - no cross-method conversions!"""
    
    def transe_score(h, r, t):
        """TransE: -||h + r - t|| (negative distance, higher is better)"""
        return -torch.norm(h + r - t, p=2, dim=-1)
    
    def distmult_score(h, r, t):
        """DistMult: sum(h * r * t)"""
        return (h * r * t).sum(dim=-1)
    
    def complex_score(h, r, t):
        """ComplEx: Re(<h, r, conj(t)>) with concatenated real/imag embeddings"""
        dim = h.shape[-1] // 2
        h_real, h_imag = h[..., :dim], h[..., dim:]
        r_real, r_imag = r[..., :dim], r[..., dim:]
        t_real, t_imag = t[..., :dim], t[..., dim:]
        
        return (h_real * r_real * t_real + 
                h_real * r_imag * t_imag + 
                h_imag * r_real * t_imag - 
                h_imag * r_imag * t_real).sum(dim=-1)
    
    def rgcn_score(h, r, t):
        """R-GCN uses DistMult scoring (standard practice)"""
        return (h * r * t).sum(dim=-1)
    
    scoring_functions = {
        'transe': transe_score,
        'distmult': distmult_score,
        'complex': complex_score,
        'rgcn': rgcn_score,
        'random': distmult_score  # Default for random init
    }
    
    if method.lower() not in scoring_functions:
        raise ValueError(f"Unknown scoring method: {method}")
    
    return scoring_functions[method.lower()]

# =============================================================================
# SMART WARM-UP LOGIC - Only when it makes sense!
# =============================================================================

def needs_warmup(embedding_method):
    """Determine if warm-up is needed - only when changing objectives"""
    warmup_needed = {
        'rgcn': True,     # R-GCN → DistMult (structure → multiplicative)
        'random': True,   # Random → Method (no training → trained)
        'transe': False,  
        'distmult': False, 
        'complex': False  
    }
    return warmup_needed.get(embedding_method.lower(), False)

def smart_warmup(node_emb, rel_emb, train_loader, device, method, epochs=50, lr=1e-2, verbose=True):
    """Smart warm-up: only train when changing objectives"""
    
    if not needs_warmup(method):
        print_progress(f"⚡ Skipping warm-up for {method} (already trained with same objective)", verbose)
        return
    
    if method.lower() == 'rgcn':
        # R-GCN → DistMult alignment
        print_progress(f"🔥 R-GCN → DistMult alignment warm-up ({epochs} epochs)...", verbose)
        target_score_fn = get_scoring_function('distmult')
    elif method.lower() == 'random':
        # Random → DistMult training (could be any method, defaulting to DistMult)
        print_progress(f"🔥 Random → DistMult training ({epochs} epochs)...", verbose)
        target_score_fn = get_scoring_function('distmult')
    else:
        return  # Should not reach here due to needs_warmup check
    
    optimizer = optim.Adam([node_emb, rel_emb.weight], lr=lr)
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            h_batch, r_batch, t_batch = [b.to(device) for b in batch.T]
            
            # Positive scores
            pos_scores = target_score_fn(node_emb[h_batch], rel_emb(r_batch), node_emb[t_batch])
            
            # Negative sampling
            neg_t = torch.randint(0, node_emb.size(0), t_batch.shape, device=device)
            neg_scores = target_score_fn(node_emb[h_batch], rel_emb(r_batch), node_emb[neg_t])
            
            # Margin loss (higher positive scores, lower negative scores)
            loss = F.relu(1.0 + neg_scores - pos_scores).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            avg_loss = total_loss / num_batches
            print_progress(f"  Warm-up Epoch {epoch}: Loss = {avg_loss:.4f}", verbose)
    
    warmup_time = time.time() - start_time
    print_progress(f" Warm-up completed in {warmup_time/60:.1f} minutes", verbose)

# =============================================================================
# EMBEDDING INITIALIZATION METHODS
# =============================================================================

class EmbeddingInitializer(ABC):
    """Abstract base class for embedding initialization"""
    
    @abstractmethod
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, verbose=True):
        pass
    
    @abstractmethod
    def get_name(self):
        pass

class RandomInitializer(EmbeddingInitializer):
    """Random Xavier initialization"""
    
    def __init__(self, embed_dim=128):
        self.embed_dim = embed_dim
    
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, verbose=True):
        print_progress(f"Initializing random embeddings (dim={self.embed_dim})...", verbose)
        
        node_embeddings = torch.randn(num_entities, self.embed_dim, device=device)
        relation_embeddings = torch.randn(num_relations, self.embed_dim, device=device)
        nn.init.xavier_uniform_(node_embeddings)
        nn.init.xavier_uniform_(relation_embeddings)
        
        print_progress("Random initialization complete", verbose)
        return node_embeddings, relation_embeddings
    
    def get_name(self):
        return "Random"

class TransEInitializer(EmbeddingInitializer):
    """TransE initialization with translational embeddings"""
    
    def __init__(self, embed_dim=128, epochs=100, lr=0.01, margin=1.0):
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.lr = lr
        self.margin = margin
    
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, verbose=True):
        print_progress(f"Training TransE embeddings ({self.epochs} epochs)...", verbose)
        
        entity_emb = nn.Embedding(num_entities, self.embed_dim).to(device)
        relation_emb = nn.Embedding(num_relations, self.embed_dim).to(device)
        
        nn.init.xavier_uniform_(entity_emb.weight)
        nn.init.xavier_uniform_(relation_emb.weight)
        
        # Normalize entity embeddings
        with torch.no_grad():
            entity_emb.weight.data = F.normalize(entity_emb.weight.data, p=2, dim=1)
        
        optimizer = optim.Adam(list(entity_emb.parameters()) + list(relation_emb.parameters()), lr=self.lr)
        train_triples = torch.tensor(train_df.values, dtype=torch.long).to(device)
        
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
            neg_t = torch.randint(0, num_entities, (batch_size,), device=device)
            neg_t_emb = entity_emb(neg_t)
            neg_scores = torch.norm(h_emb + r_emb - neg_t_emb, p=2, dim=1)
            
            # Margin loss
            loss = F.relu(self.margin + pos_scores - neg_scores).mean()
            loss.backward()
            optimizer.step()
            
            # Normalize entity embeddings
            with torch.no_grad():
                entity_emb.weight.data = F.normalize(entity_emb.weight.data, p=2, dim=1)
            
            if verbose and epoch % 20 == 0:
                print_progress(f"  TransE Epoch {epoch}: Loss = {loss.item():.4f}", verbose)
        
        print_progress("TransE training complete", verbose)
        return entity_emb.weight.detach(), relation_emb.weight.detach()
    
    def get_name(self):
        return "TransE"

class DistMultInitializer(EmbeddingInitializer):
    """DistMult initialization"""
    
    def __init__(self, embed_dim=128, epochs=100, lr=0.01, regularization=0.01):
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.lr = lr
        self.regularization = regularization
    
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, verbose=True):
        print_progress(f"Training DistMult embeddings ({self.epochs} epochs)...", verbose)
        
        entity_emb = nn.Embedding(num_entities, self.embed_dim).to(device)
        relation_emb = nn.Embedding(num_relations, self.embed_dim).to(device)
        nn.init.xavier_uniform_(entity_emb.weight)
        nn.init.xavier_uniform_(relation_emb.weight)
        
        optimizer = optim.Adam(list(entity_emb.parameters()) + list(relation_emb.parameters()), lr=self.lr)
        train_triples = torch.tensor(train_df.values, dtype=torch.long).to(device)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            batch_size = min(1024, len(train_triples))
            batch_idx = torch.randperm(len(train_triples))[:batch_size]
            batch_triples = train_triples[batch_idx]
            
            h, r, t = batch_triples[:, 0], batch_triples[:, 1], batch_triples[:, 2]
            h_emb, r_emb, t_emb = entity_emb(h), relation_emb(r), entity_emb(t)
            
            # DistMult scoring
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
            
            if verbose and epoch % 20 == 0:
                print_progress(f"  DistMult Epoch {epoch}: Loss = {total_loss.item():.4f}", verbose)
        
        print_progress("DistMult training complete", verbose)
        return entity_emb.weight.detach(), relation_emb.weight.detach()
    
    def get_name(self):
        return "DistMult"

class ComplExInitializer(EmbeddingInitializer):
    """ComplEx initialization with complex embeddings"""
    
    def __init__(self, embed_dim=128, epochs=100, lr=0.01, regularization=0.01):
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.lr = lr
        self.regularization = regularization
    
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, verbose=True):
        print_progress(f"Training ComplEx embeddings ({self.epochs} epochs)...", verbose)
        
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
            
            if verbose and epoch % 20 == 0:
                print_progress(f"  ComplEx Epoch {epoch}: Loss = {total_loss.item():.4f}", verbose)
        
        # Combine real and imaginary parts for final embeddings
        final_entity_emb = torch.cat([entity_real.weight, entity_imag.weight], dim=1)
        final_relation_emb = torch.cat([relation_real.weight, relation_imag.weight], dim=1)
        
        print_progress("ComplEx training complete", verbose)
        return final_entity_emb.detach(), final_relation_emb.detach()
    
    def get_name(self):
        return "ComplEx"

class RGCNDistMult(nn.Module):
    """R-GCN for structure-aware embeddings"""
    
    def __init__(self, num_entities, num_relations, hidden_dim=128, num_bases=None, num_layers=2):
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        
        if num_bases is None:
            num_bases = min(num_relations, max(25, num_relations // 4))
        
        # Entity and relation embeddings
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim)
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        
        # R-GCN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases)
            self.layers.append(layer)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward_entities(self, edge_index, edge_type):
        """Forward pass through R-GCN"""
        x = self.entity_embedding.weight
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
            x = F.relu(x)
            x = self.dropout(x)
        return x

class RGCNInitializer(EmbeddingInitializer):
    """R-GCN initialization for structure-aware embeddings"""
    
    def __init__(self, embed_dim=128, epochs=50, lr=0.01, layers=2):
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.lr = lr
        self.layers = layers
    
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, verbose=True):
        print_progress(f"Training R-GCN embeddings ({self.epochs} epochs)...", verbose)
        
        # Build graph structure
        all_data = pd.concat([train_df, val_df, test_df])
        all_triples = torch.tensor(all_data.values, dtype=torch.long)
        edge_index = all_triples[:, [0, 2]].t().contiguous().to(device)
        edge_type = all_triples[:, 1].to(device)
        
        # Initialize R-GCN model
        rgcn_model = RGCNDistMult(num_entities, num_relations, self.embed_dim, num_layers=self.layers).to(device)
        optimizer = optim.Adam(rgcn_model.parameters(), lr=self.lr)
        
        train_triples = torch.tensor(train_df.values, dtype=torch.long).to(device)
        
        for epoch in range(self.epochs):
            rgcn_model.train()
            optimizer.zero_grad()
            
            entity_embeddings = rgcn_model.forward_entities(edge_index, edge_type)
            
            # DistMult training on R-GCN embeddings
            batch_size = min(1024, len(train_triples))
            batch_idx = torch.randperm(len(train_triples))[:batch_size]
            batch_triples = train_triples[batch_idx]
            
            h, r, t = batch_triples[:, 0], batch_triples[:, 1], batch_triples[:, 2]
            pos_heads = entity_embeddings[h]
            pos_rels = rgcn_model.relation_embedding(r)
            pos_tails = entity_embeddings[t]
            pos_scores = (pos_heads * pos_rels * pos_tails).sum(dim=1)
            
            # Negative sampling
            neg_tails = torch.randint(0, num_entities, (batch_size,), device=device)
            neg_tail_embs = entity_embeddings[neg_tails]
            neg_scores = (pos_heads * pos_rels * neg_tail_embs).sum(dim=1)
            
            # Margin loss
            loss = F.relu(1.0 + neg_scores - pos_scores).mean()
            loss.backward()
            optimizer.step()
            
            if verbose and epoch % 10 == 0:
                print_progress(f"  R-GCN Epoch {epoch}: Loss = {loss.item():.4f}", verbose)
        
        # Extract final embeddings
        with torch.no_grad():
            final_node_embeddings = rgcn_model.forward_entities(edge_index, edge_type)
            final_rel_embeddings = rgcn_model.relation_embedding.weight
        
        print_progress("R-GCN training complete", verbose)
        return final_node_embeddings, final_rel_embeddings
    
    def get_name(self):
        return "R-GCN"

def create_embedding_initializer(args):
    """Factory function to create embedding initializer"""
    
    method = args.embedding_init.lower()
    
    if method == 'rgcn':
        return RGCNInitializer(args.embed_dim, args.rgcn_epochs, args.rgcn_lr, args.rgcn_layers)
    elif method == 'random':
        return RandomInitializer(args.embed_dim)
    elif method == 'transe':
        return TransEInitializer(args.embed_dim, args.transe_epochs, args.transe_lr, args.transe_margin)
    elif method == 'distmult':
        return DistMultInitializer(args.embed_dim, args.distmult_epochs, args.distmult_lr, args.distmult_regularization)
    elif method in ['complex', 'complEx']:
        return ComplExInitializer(args.embed_dim, args.complex_epochs, args.complex_lr, args.complex_regularization)
    else:
        raise ValueError(f"Unknown embedding initialization method: {method}")

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_root, train_file, val_file, test_file, debug_mode=False, max_samples=None):
    """Load data from CSV files"""
    
    print(f"Loading data from {data_root}...")
    
    train_df = pd.read_csv(os.path.join(data_root, train_file))
    val_df = pd.read_csv(os.path.join(data_root, val_file))
    test_df = pd.read_csv(os.path.join(data_root, test_file))
    
    # Find column names
    possible_head_cols = ['H', 'head', 'subject', 'h']
    possible_rel_cols = ['R', 'relation', 'predicate', 'r']
    possible_tail_cols = ['T', 'tail', 'object', 't']
    
    head_col = next((col for col in possible_head_cols if col in train_df.columns), None)
    rel_col = next((col for col in possible_rel_cols if col in train_df.columns), None)
    tail_col = next((col for col in possible_tail_cols if col in train_df.columns), None)
    
    if not all([head_col, rel_col, tail_col]):
        raise ValueError(f"Could not find H/R/T columns. Found: {list(train_df.columns)}")
    
    # Standardize column names
    for df in [train_df, val_df, test_df]:
        df.rename(columns={head_col: 'H', rel_col: 'R', tail_col: 'T'}, inplace=True)
    
    if debug_mode and max_samples:
        train_df = train_df.sample(min(len(train_df), max_samples), random_state=42)
        val_df = val_df.sample(min(len(val_df), max_samples//5), random_state=42)
        test_df = test_df.sample(min(len(test_df), max_samples//5), random_state=42)
        print(f"DEBUG: Using {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    print(f"Data loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test triples")
    return train_df, val_df, test_df

def create_entity_relation_mappings(train_df, val_df, test_df):
    """Create entity and relation ID mappings"""
    
    all_data = pd.concat([train_df, val_df, test_df])
    all_entities = sorted(set(all_data['H'].unique()) | set(all_data['T'].unique()))
    all_relations = sorted(set(all_data['R'].unique()))
    
    entity_to_id = {entity: idx for idx, entity in enumerate(all_entities)}
    relation_to_id = {relation: idx for idx, relation in enumerate(all_relations)}
    
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
# GAN MODELS
# =============================================================================

class Generator(nn.Module):
    """Generator for knowledge graph completion"""
    
    def __init__(self, embed_dim=128, noise_dim=64):
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
    """Discriminator for knowledge graph completion"""
    
    def __init__(self, embed_dim=128, hidden_dim=1024, dropout=0.3):
        super().__init__()
        in_features = embed_dim * 3
        
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.drop1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.drop2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim, in_features)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        self.drop3 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(in_features, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, head_emb, rel_emb, tail_emb):
        x = torch.cat([head_emb, rel_emb, tail_emb], dim=-1).float()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        res = self.fc2(x)
        res = self.bn2(res)
        res = self.act2(res)
        res = self.drop2(res)
        res = res + x
        
        x = self.fc3(res)
        x = self.act3(x)
        x = self.drop3(x)
        out = self.fc_out(x).view(-1)
        
        return out

# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def print_progress(message, verbose=True):
    """Conditional printing"""
    if verbose:
        print(message)

def compute_metrics(y_true, y_probs, threshold=0.5):
    """Compute classification metrics"""
    try:
        if torch.is_tensor(y_true):
            y_true = y_true.detach().cpu().numpy()
        if torch.is_tensor(y_probs):
            y_probs = y_probs.detach().cpu().numpy()
        
        y_true = np.array(y_true, dtype=np.float64)
        y_probs = np.array(y_probs, dtype=np.float64)
        y_pred = (y_probs >= threshold).astype(int)
        
        if len(set(y_true)) <= 1:
            return {"F1": 0.0, "AUPR": 0.0, "MCC": 0.0, "AUC": 0.0}
        
        return {
            "F1": float(f1_score(y_true, y_pred, zero_division=0)),
            "AUPR": float(average_precision_score(y_true, y_probs)),
            "MCC": float(matthews_corrcoef(y_true, y_pred)),
            "AUC": float(roc_auc_score(y_true, y_probs)) if len(set(y_true)) > 1 else 0.0
        }
    except Exception as e:
        print(f"Warning: Metrics calculation failed: {e}")
        return {"F1": 0.0, "AUPR": 0.0, "MCC": 0.0, "AUC": 0.0}

def check_early_stopping(current_metric, best_metric, patience_counter, patience_limit, verbose=True):
    """Check if early stopping should trigger"""
    if current_metric > best_metric + 1e-5:
        return current_metric, 0, False
    else:
        patience_counter += 1
        should_stop = patience_counter >= patience_limit
        if verbose and should_stop:
            print_progress(f"Early stopping triggered: no improvement for {patience_limit} epochs", verbose)
        return best_metric, patience_counter, should_stop

def evaluate_hit_at_k(data_loader, generator, node_emb, rel_emb, device, hit_at_k_list, score_function):
    """Evaluate Hit@K metrics using method-specific scoring"""
    if generator:
        generator.eval()
    
    hits_dict = {k: 0 for k in hit_at_k_list}
    total_examples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            h, r, t = [b.to(device) for b in batch.T]
            
            if generator:
                # Use generator predictions
                h_emb, r_emb_batch = node_emb[h], rel_emb(r)
                pred = generator(h_emb, r_emb_batch)
                
                # Score all possible tails
                h_exp = h_emb.unsqueeze(1).expand(-1, node_emb.size(0), -1)
                r_exp = r_emb_batch.unsqueeze(1).expand(-1, node_emb.size(0), -1)
                all_t = node_emb.unsqueeze(0).expand(len(h), -1, -1)
                
                scores = score_function(h_exp, r_exp, all_t)
            else:
                # Direct scoring without generator
                h_emb, r_emb_batch = node_emb[h], rel_emb(r)
                
                h_exp = h_emb.unsqueeze(1).expand(-1, node_emb.size(0), -1)
                r_exp = r_emb_batch.unsqueeze(1).expand(-1, node_emb.size(0), -1)
                all_t = node_emb.unsqueeze(0).expand(len(h), -1, -1)
                
                scores = score_function(h_exp, r_exp, all_t)
            
            for k in hit_at_k_list:
                topk = scores.topk(k, dim=1).indices
                for i in range(len(t)):
                    if t[i].item() in topk[i]:
                        hits_dict[k] += 1
            total_examples += len(t)
    
    if generator:
        generator.train()
    
    return {k: hits_dict[k] / total_examples for k in hit_at_k_list} if total_examples > 0 else {k: 0.0 for k in hit_at_k_list}

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

def get_hard_negatives(h_emb, r_emb, t_emb, node_emb, k=5):
    """Generate hard negative samples"""
    with torch.no_grad():
        sim = torch.matmul(F.normalize(t_emb, dim=-1), F.normalize(node_emb, dim=-1).T)
        _, hard_neg_idx = sim.topk(k, dim=1)
        
        batch_size = h_emb.shape[0]
        selected_idx = torch.randint(0, k, (batch_size,), device=h_emb.device)
        final_neg_idx = torch.gather(hard_neg_idx, 1, selected_idx.unsqueeze(1)).squeeze(1)
        
        return node_emb[final_neg_idx]

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_prot_b_gan_clean(args):
    """Main training pipeline with clean, logical architecture"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Prot-B-GAN Clean Architecture Training")
    print(f"Device: {device}")
    print(f"Embedding method: {args.embedding_init}")
    print("=" * 70)
    
    try:
        # Load data
        train_df, val_df, test_df = load_data(
            args.data_root, args.train_file, args.val_file, args.test_file,
            debug_mode=args.debug, max_samples=20000 if args.debug else None
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
        
        # =========================================================================
        # STAGE 1: Embedding Initialization
        # =========================================================================
        
        print_progress(f"\n STAGE 1: {args.embedding_init.upper()} Embedding Initialization", args.verbose)
        
        embedding_initializer = create_embedding_initializer(args)
        node_embeddings, rel_embeddings = embedding_initializer.initialize_embeddings(
            train_df, val_df, test_df, num_entities, num_relations, device, args.verbose
        )
        
        # Set up embeddings
        node_emb = nn.Parameter(node_embeddings.clone().detach()).to(device)
        rel_emb = nn.Embedding(num_relations, args.embed_dim).to(device)
        rel_emb.weight.data.copy_(rel_embeddings)
        
        # Get method-specific scoring function
        score_function = get_scoring_function(args.embedding_init)
        print_progress(f"Using {args.embedding_init} scoring function throughout", args.verbose)
        
        # =========================================================================
        # STAGE 2: Smart Warm-up (Only When Needed)
        # =========================================================================
        
        print_progress(f"\n STAGE 2: Smart Warm-up Analysis", args.verbose)
        
        if needs_warmup(args.embedding_init):
            print_progress(f" Warm-up needed for {args.embedding_init}", args.verbose)
            smart_warmup(node_emb, rel_emb, train_loader, device, args.embedding_init, 
                        epochs=args.warmup_epochs, lr=args.warmup_lr, verbose=args.verbose)
        else:
            print_progress(f" Warm-up skipped for {args.embedding_init} (same objective already trained)", args.verbose)
        
        # Evaluate baseline performance
        print_progress("Evaluating baseline performance...", args.verbose)
        baseline_hit_at_k = evaluate_hit_at_k(val_loader, None, node_emb, rel_emb, device, args.hit_at_k, score_function)
        print_progress(f"Baseline Hit@10: {baseline_hit_at_k[10]:.4f}", args.verbose)
        
        # =========================================================================
        # STAGE 3: GAN-RL Training
        # =========================================================================
        
        print_progress(f"\n STAGE 3: Progressive GAN-RL Training", args.verbose)
        
        # Initialize models
        generator = Generator(args.embed_dim, args.noise_dim).to(device)
        discriminator = Discriminator(args.embed_dim, args.hidden_dim).to(device)
        
        g_opt = optim.Adam(list(generator.parameters()) + [node_emb], lr=args.g_lr)
        d_opt = optim.Adam(discriminator.parameters(), lr=args.d_lr)
        
        # Training history
        training_history = {
            'losses': [], 'cos_sims': [], 
            'train_hitks': {k: [] for k in args.hit_at_k},
            'val_hitks': {k: [] for k in args.hit_at_k},
            'f1_history': [], 'val_f1_history': []
        }
        
        # Training state
        best_val_hit10 = 0.0
        best_epoch = 0
        current_tier = 1
        
        bce_loss = nn.BCEWithLogitsLoss()
        
        print_progress("Starting three-tier progressive training...", args.verbose)
        
        for epoch in range(1, args.epochs + 1):
            
            # =================================================================
            # TIER 1: Generator Pretraining
            # =================================================================
            
            if current_tier == 1:
                generator.train()
                total_loss = 0.0
                
                for step, batch in enumerate(train_loader):
                    if args.debug and step >= 50:
                        break
                    
                    h, r, t = [b.to(device) for b in batch.T]
                    h_emb, r_emb_batch, t_emb = node_emb[h], rel_emb(r), node_emb[t]
                    
                    g_opt.zero_grad()
                    fake = generator(h_emb, r_emb_batch)
                    
                    # Reconstruction loss
                    rec_loss = F.mse_loss(fake, t_emb)
                    
                    # Method-specific scoring loss
                    pos_score = score_function(h_emb, r_emb_batch, fake)
                    target_score = score_function(h_emb, r_emb_batch, t_emb)
                    score_loss = F.mse_loss(pos_score, target_score)
                    
                    # Cosine similarity loss
                    cos_loss = 1 - F.cosine_similarity(fake, t_emb).mean()
                    
                    pretrain_loss = rec_loss + 0.5 * score_loss + 0.3 * cos_loss
                    pretrain_loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_norm)
                    g_opt.step()
                    
                    total_loss += pretrain_loss.item()
                
                avg_loss = total_loss / min(len(train_loader), 50 if args.debug else len(train_loader))
                
                # Evaluate and check for tier transition
                val_hitks = evaluate_hit_at_k(val_loader, generator, node_emb, rel_emb, device, args.hit_at_k, score_function)
                
                if val_hitks[10] > best_val_hit10:
                    best_val_hit10 = val_hitks[10]
                    best_epoch = epoch
                
                print_progress(f"[Tier 1] Epoch {epoch}: Loss {avg_loss:.4f} | Hit@10 {val_hitks[10]:.4f}", args.verbose)
                
                # Transition to Tier 2 after pretraining
                if epoch >= args.pretrain_epochs or val_hitks[10] > baseline_hit_at_k[10] + 0.1:
                    print_progress("Transitioning to Tier 2: RL Training", args.verbose)
                    current_tier = 2
                    node_emb.requires_grad_(False)
                    g_opt = optim.Adam(generator.parameters(), lr=args.g_lr * 0.5)
                
                continue
            
            # =================================================================
            # TIER 2 & 3: RL and Adversarial Training
            # =================================================================
            
            generator.train()
            discriminator.train()
            
            total_loss, total_cos = 0.0, 0.0
            true_labels, pred_probs = [], []
            
            for step, batch in enumerate(train_loader):
                if args.debug and step >= 50:
                    break
                
                h, r, t = [b.to(device) for b in batch.T]
                h_emb, r_emb_batch, t_emb = node_emb[h], rel_emb(r), node_emb[t]
                
                # Discriminator training (Tier 3 only)
                if current_tier == 3 and step % args.d_update_freq == 0:
                    d_opt.zero_grad()
                    
                    with torch.no_grad():
                        fake_samples = generator(h_emb.detach(), r_emb_batch.detach())
                    
                    real_scores = discriminator(h_emb.detach(), r_emb_batch.detach(), t_emb.detach())
                    fake_scores = discriminator(h_emb.detach(), r_emb_batch.detach(), fake_samples)
                    
                    # Hard negatives
                    hard_neg_emb = get_hard_negatives(h_emb.detach(), r_emb_batch.detach(), t_emb.detach(), node_emb)
                    hard_neg_scores = discriminator(h_emb.detach(), r_emb_batch.detach(), hard_neg_emb)
                    
                    # Discriminator loss
                    real_labels = torch.full_like(real_scores, 0.9)
                    fake_labels = torch.full_like(fake_scores, 0.1)
                    hard_neg_labels = torch.full_like(hard_neg_scores, 0.1)
                    
                    d_loss = (bce_loss(real_scores, real_labels) +
                             bce_loss(fake_scores, fake_labels) +
                             bce_loss(hard_neg_scores, hard_neg_labels)) / 3
                    
                    d_loss.backward()
                    d_opt.step()
                    
                    # Track discriminator performance
                    true_labels.extend([1]*len(real_scores) + [0]*len(fake_scores) + [0]*len(hard_neg_scores))
                    pred_probs.extend(torch.sigmoid(torch.cat([real_scores, fake_scores, hard_neg_scores])).cpu().numpy())
                
                # Generator training
                g_opt.zero_grad()
                fake = generator(h_emb, r_emb_batch)
                
                loss_components = []
                
                # Method-specific RL reward
                method_scores = score_function(h_emb, r_emb_batch, fake)
                rl_loss = -torch.tanh(method_scores / 5.0).mean()
                loss_components.append(args.rl_weight * rl_loss)
                
                # Adversarial loss (Tier 3 only)
                if current_tier == 3:
                    adv_scores = discriminator(h_emb, r_emb_batch, fake)
                    adv_loss = -torch.tanh(adv_scores / 5.0).mean()
                    loss_components.append(args.adv_weight * adv_loss)
                
                # Cosine margin loss
                neg_idx = torch.randint(0, node_emb.size(0), (len(h), 5), device=device)
                neg_t_embs = node_emb[neg_idx]
                
                pos_cos = F.cosine_similarity(fake, t_emb, dim=-1)
                neg_cos = F.cosine_similarity(fake.unsqueeze(1), neg_t_embs, dim=-1)
                cos_margin = F.relu(0.3 + neg_cos - pos_cos.unsqueeze(1)).mean()
                
                # L2 regularization
                l2_loss = (fake - t_emb.detach()).pow(2).mean()
                
                # Final generator loss
                final_loss = sum(loss_components) + args.g_guidance_weight * cos_margin + args.l2_reg_weight * l2_loss
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_norm)
                g_opt.step()
                
                total_loss += final_loss.item()
                total_cos += F.cosine_similarity(fake, t_emb).mean().item()
            
            # Epoch evaluation
            effective_steps = min(len(train_loader), 50 if args.debug else len(train_loader))
            avg_loss = total_loss / effective_steps
            avg_cos = total_cos / effective_steps
            
            # Hit@K evaluation
            train_hit_at_k = evaluate_hit_at_k(train_loader, generator, node_emb, rel_emb, device, args.hit_at_k, score_function)
            val_hit_at_k = evaluate_hit_at_k(val_loader, generator, node_emb, rel_emb, device, args.hit_at_k, score_function)
            
            # Validation metrics
            val_metrics, val_cos_avg = validate(val_loader, generator, discriminator, node_emb, rel_emb, device)
            
            # Update history
            training_history['losses'].append(avg_loss)
            training_history['cos_sims'].append(avg_cos)
            for k in args.hit_at_k:
                training_history['train_hitks'][k].append(train_hit_at_k[k])
                training_history['val_hitks'][k].append(val_hit_at_k[k])
            
            if true_labels:
                disc_metrics = compute_metrics(np.array(true_labels), np.array(pred_probs))
                training_history['f1_history'].append(disc_metrics['F1'])
            training_history['val_f1_history'].append(val_metrics['F1'])
            
            # Check for best model
            if val_hit_at_k[10] > best_val_hit10:
                best_val_hit10 = val_hit_at_k[10]
                best_epoch = epoch
            
            # Print progress
            tier_name = f"Tier {current_tier}"
            disc_f1 = disc_metrics['F1'] if true_labels else 0.0
            
            print_progress(f"[{tier_name}] E{epoch:03d} | Loss {avg_loss:.4f} | CosSim {avg_cos:.3f} | "
                          f"D_F1 {disc_f1:.3f} | Hit@10 {val_hit_at_k[10]:.4f}", args.verbose)
            
            # Tier transition
            if current_tier == 2 and epoch >= args.rl_start_epoch:
                print_progress("Transitioning to Tier 3: Full Adversarial Training", args.verbose)
                current_tier = 3
            
            # Early stopping
            if epoch - best_epoch > args.early_stopping_patience:
                print_progress(f"Early stopping at epoch {epoch}", args.verbose)
                break
        
        # =========================================================================
        # Final Evaluation
        # =========================================================================
        
        print(f"\n TRAINING COMPLETED!")
        print(f"Best validation Hit@10: {best_val_hit10:.4f} at epoch {best_epoch}")
        
        # Final test evaluation
        test_hit_at_k = evaluate_hit_at_k(test_loader, generator, node_emb, rel_emb, device, args.hit_at_k, score_function)
        
        print(f"\n FINAL TEST RESULTS:")
        for k in args.hit_at_k:
            print(f"Test Hit@{k}: {test_hit_at_k[k]:.4f} ({test_hit_at_k[k]*100:.1f}%)")
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        
        final_results = {
            "method": args.embedding_init,
            "baseline_hit_at_k": baseline_hit_at_k,
            "test_hit_at_k": test_hit_at_k,
            "best_val_hit10": best_val_hit10,
            "best_epoch": best_epoch,
            "training_history": training_history,
            "improvement": {k: test_hit_at_k[k] - baseline_hit_at_k[k] for k in args.hit_at_k}
        }
        
        results_path = os.path.join(args.output_dir, f"results_{args.embedding_init}.pt")
        torch.save(final_results, results_path)
        
        checkpoint = {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "node_emb": node_emb.detach().cpu(),
            "rel_emb": rel_emb.state_dict(),
            "final_results": final_results
        }
        
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{args.embedding_init}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        print(f"\n💾 Results saved to: {args.output_dir}")
        print(f"   - {results_path}")
        print(f"   - {checkpoint_path}")
        
        return final_results
        
    except Exception as e:
        print(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prot-B-GAN: Clean Architecture for KG Completion')
    
    # Data and I/O
    parser.add_argument('--data_root', type=str, required=True, help='Root directory containing data files')
    parser.add_argument('--train_file', type=str, default='train.csv', help='Training data file')
    parser.add_argument('--val_file', type=str, default='val.csv', help='Validation data file')
    parser.add_argument('--test_file', type=str, default='test.csv', help='Test data file')
    parser.add_argument('--output_dir', type=str, default='./clean_results', help='Output directory')
    
    # Model Architecture
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--noise_dim', type=int, default=64, help='Generator noise dimension')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Discriminator hidden dimension')
    
    # Training Parameters
    parser.add_argument('--epochs', type=int, default=30, help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--g_lr', type=float, default=1e-3, help='Generator learning rate')
    parser.add_argument('--d_lr', type=float, default=5e-5, help='Discriminator learning rate')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Gradient clipping')
    
    # Embedding Initialization
    parser.add_argument('--embedding_init', type=str, default='rgcn', 
                        choices=['rgcn', 'random', 'transe', 'distmult', 'complex'],
                        help='Embedding initialization method')
    
    # Method-specific parameters
    parser.add_argument('--rgcn_epochs', type=int, default=50, help='R-GCN epochs')
    parser.add_argument('--rgcn_lr', type=float, default=0.01, help='R-GCN learning rate')
    parser.add_argument('--rgcn_layers', type=int, default=2, help='R-GCN layers')
    
    parser.add_argument('--transe_epochs', type=int, default=100, help='TransE epochs')
    parser.add_argument('--transe_lr', type=float, default=0.01, help='TransE learning rate')
    parser.add_argument('--transe_margin', type=float, default=1.0, help='TransE margin')
    
    parser.add_argument('--distmult_epochs', type=int, default=100, help='DistMult epochs')
    parser.add_argument('--distmult_lr', type=float, default=0.01, help='DistMult learning rate')
    parser.add_argument('--distmult_regularization', type=float, default=0.01, help='DistMult L2 reg')
    
    parser.add_argument('--complex_epochs', type=int, default=100, help='ComplEx epochs')
    parser.add_argument('--complex_lr', type=float, default=0.01, help='ComplEx learning rate')
    parser.add_argument('--complex_regularization', type=float, default=0.01, help='ComplEx L2 reg')
    
    # Smart warm-up
    parser.add_argument('--warmup_epochs', type=int, default=50, help='Smart warm-up epochs')
    parser.add_argument('--warmup_lr', type=float, default=1e-2, help='Smart warm-up learning rate')
    
    # Progressive training
    parser.add_argument('--pretrain_epochs', type=int, default=10, help='Generator pretraining epochs')
    parser.add_argument('--rl_start_epoch', type=int, default=15, help='RL training start epoch')
    parser.add_argument('--d_update_freq', type=int, default=5, help='Discriminator update frequency')
    
    # Loss weights
    parser.add_argument('--rl_weight', type=float, default=1.0, help='RL loss weight')
    parser.add_argument('--adv_weight', type=float, default=0.5, help='Adversarial loss weight')
    parser.add_argument('--g_guidance_weight', type=float, default=2.0, help='Generator guidance weight')
    parser.add_argument('--l2_reg_weight', type=float, default=0.1, help='L2 regularization weight')
    
    # Evaluation
    parser.add_argument('--hit_at_k', type=int, nargs='+', default=[1, 5, 10], help='Hit@K values')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience')
    
    # Debug and misc
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.debug:
        args.epochs = min(args.epochs, 25)
        args.pretrain_epochs = min(args.pretrain_epochs, 5)
        args.early_stopping_patience = 5
        print("DEBUG MODE: Reduced epochs and patience")
    
    # Verify data files
    data_files = [args.train_file, args.val_file, args.test_file]
    for file in data_files:
        if not os.path.exists(os.path.join(args.data_root, file)):
            print(f"Error: Data file not found: {os.path.join(args.data_root, file)}")
            return 1
    
    print(f" Training {args.embedding_init.upper()} + GAN-RL")
    print(f"Smart warm-up: {'Enabled' if needs_warmup(args.embedding_init) else 'Skipped (not needed)'}")
    
    # Run training
    results = train_prot_b_gan_clean(args)
    
    if results:
        print(f"\n SUCCESS! {args.embedding_init.upper()} + GAN-RL completed")
        print(f"Baseline → Final improvement:")
        for k in args.hit_at_k:
            improvement = results['improvement'][k]
            print(f"  Hit@{k}: +{improvement:.4f} ({improvement*100:+.1f}%)")
        return 0
    else:
        print(f" Training failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
