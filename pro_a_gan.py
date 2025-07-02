"""
Prot-B-GAN Complete - Modular Architecture
==================================================================

UPDATED: Modular design with separated R-GCN structure learning and scoring-specific warm-up
- R-GCN: Pure structure learning (no scoring method bias)
- Warm-up: Dedicated scoring alignment (DistMult/TransE/ComplEx)
- Consistent embedding dimensions throughout pipeline
- Single set of embeddings passed through all phases

Usage Examples:

    # R-GCN structure + DistMult scoring
    python prot_b_gan_modular.py \
        --data_root "/path/to/data" \
        --embedding_init rgcn --reward_scoring_method distmult \
        --embed_dim 500 --epochs 500 \
        --verbose

    # R-GCN structure + TransE scoring  
    python prot_b_gan_modular.py \
        --data_root "/path/to/data" \
        --embedding_init rgcn --reward_scoring_method transe \
        --embed_dim 500 --epochs 500 \
        --verbose
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
# REWARD SCORING FUNCTIONS (METHOD-CONSISTENT + UNIVERSAL)
# =============================================================================

def get_reward_scoring_function(reward_method):
    """Get method-specific reward scoring function with consistent output range [-0.5, 0.5]"""
    
    def transe_reward_function(h, r, t):
        """TransE: Convert distance to reward (lower distance = higher reward)"""
        distance = torch.norm(h + r - t, p=2, dim=-1)
        reward = torch.sigmoid(-distance / 2.0)  # Range: [0, 1]
        return reward - 0.5  # Center around 0: [-0.5, 0.5]
    
    def distmult_reward_function(h, r, t):
        """DistMult: Normalize multiplicative score"""
        raw_score = (h * r * t).sum(dim=-1)
        reward = torch.sigmoid(raw_score / 5.0)  # Range: [0, 1]
        return reward - 0.5  # Center around 0: [-0.5, 0.5]
    
    def complex_reward_function(h, r, t):
        """ComplEx: Handle complex-valued embeddings"""
        dim = h.shape[-1] // 2
        h_real, h_imag = h[..., :dim], h[..., dim:]
        r_real, r_imag = r[..., :dim], r[..., dim:]
        t_real, t_imag = t[..., :dim], t[..., dim:]
        
        score = (h_real * r_real * t_real + 
                h_real * r_imag * t_imag + 
                h_imag * r_real * t_imag - 
                h_imag * r_imag * t_real).sum(dim=-1)
        
        reward = torch.tanh(score / 5.0) * 0.5
        return reward
    
    def universal_distmult_reward_function(h, r, t):
        """Universal DistMult: Use for any embedding type"""
        raw_score = (h * r * t).sum(dim=-1)
        reward = torch.sigmoid(raw_score / 5.0)
        return reward - 0.5
    
    reward_functions = {
        'transe': transe_reward_function,
        'distmult': distmult_reward_function,
        'complex': complex_reward_function,
        'universal_distmult': universal_distmult_reward_function,
        'auto': None  # Will be set based on embedding method
    }
    
    if reward_method.lower() not in reward_functions:
        raise ValueError(f"Unknown reward scoring method: {reward_method}")
    
    return reward_functions[reward_method.lower()]

def get_evaluation_scoring_function(method):
    """Get evaluation scoring function for Hit@K computation"""
    
    def transe_eval_score(h, r, t):
        """TransE: -||h + r - t|| (negative distance, higher is better)"""
        return -torch.norm(h + r - t, p=2, dim=-1)
    
    def distmult_eval_score(h, r, t):
        """DistMult: sum(h * r * t)"""
        return (h * r * t).sum(dim=-1)
    
    def complex_eval_score(h, r, t):
        """ComplEx: Re(<h, r, conj(t)>) with concatenated real/imag embeddings"""
        dim = h.shape[-1] // 2
        h_real, h_imag = h[..., :dim], h[..., dim:]
        r_real, r_imag = r[..., :dim], r[..., dim:]
        t_real, t_imag = t[..., :dim], t[..., dim:]
        
        return (h_real * r_real * t_real + 
                h_real * r_imag * t_imag + 
                h_imag * r_real * t_imag - 
                h_imag * r_imag * t_real).sum(dim=-1)
    
    def rgcn_eval_score(h, r, t):
        """R-GCN uses DistMult scoring (standard practice)"""
        return (h * r * t).sum(dim=-1)
    
    scoring_functions = {
        'transe': transe_eval_score,
        'distmult': distmult_eval_score,
        'complex': complex_eval_score,
        'rgcn': rgcn_eval_score,
        'random': distmult_eval_score  
    }
    
    if method.lower() not in scoring_functions:
        raise ValueError(f"Unknown evaluation scoring method: {method}")
    
    return scoring_functions[method.lower()]

def resolve_reward_method(embedding_method, reward_method):
    """Resolve 'auto' reward method to specific method based on embedding"""
    if reward_method == 'auto':
        if embedding_method in ['transe', 'distmult', 'complex']:
            return embedding_method
        elif embedding_method in ['rgcn', 'random']:
            return 'distmult'
        else:
            return 'distmult'
    return reward_method

# =============================================================================
# UPDATED: MODULAR WARM-UP LOGIC
# =============================================================================

def needs_warmup(embedding_method, reward_method):
    """UPDATED: Always need warm-up for modular architecture"""
    # With modular design, we ALWAYS do warm-up for scoring alignment
    # Even if embedding and reward methods match, dedicated warm-up is beneficial
    return True

def modular_warmup(node_emb, rel_emb, train_loader, device, embedding_method, reward_method, epochs=50, lr=1e-2, verbose=True):
    """UPDATED: Modular warm-up for any embedding + scoring combination"""
    
    print_progress(f"Modular warm-up: Aligning {embedding_method} embeddings with {reward_method} scoring ({epochs} epochs)...", verbose)
    
    # Get the target scoring function for warm-up
    if reward_method in ['distmult', 'universal_distmult']:
        target_score_fn = get_evaluation_scoring_function('distmult')
        score_name = "DistMult"
    elif reward_method == 'transe':
        target_score_fn = get_evaluation_scoring_function('transe')
        score_name = "TransE"
    elif reward_method == 'complex':
        target_score_fn = get_evaluation_scoring_function('complex')
        score_name = "ComplEx"
    else:
        target_score_fn = get_evaluation_scoring_function('distmult')
        score_name = "DistMult"
    
    optimizer = optim.Adam([node_emb, rel_emb.weight], lr=lr)
    start_time = time.time()
    
    print_progress(f"  Using {score_name} scoring function for warm-up alignment", verbose)
    
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
            if reward_method == 'transe':
                # For TransE, pos_scores are negative distances (higher is better)
                loss = F.relu(1.0 + neg_scores - pos_scores).mean()
            else:
                # For DistMult/ComplEx, higher scores are better
                loss = F.relu(1.0 + neg_scores - pos_scores).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            avg_loss = total_loss / num_batches
            print_progress(f"  Warm-up Epoch {epoch}: {score_name} Loss = {avg_loss:.4f}", verbose)
    
    warmup_time = time.time() - start_time
    print_progress(f"Modular warm-up completed in {warmup_time/60:.1f} minutes", verbose)

# =============================================================================
# EMBEDDING INITIALIZATION METHODS (UPDATED)
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
    """UPDATED: R-GCN for PURE structure-aware embeddings (no scoring bias)"""
    
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
    """UPDATED: R-GCN initialization for PURE structure-aware embeddings"""
    
    def __init__(self, embed_dim=128, epochs=100, lr=0.01, layers=2):
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.lr = lr
        self.layers = layers
    
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, verbose=True):
        print_progress(f"Training R-GCN STRUCTURE embeddings ({self.epochs} epochs) - NO scoring bias...", verbose)
        
        # Build graph structure
        all_data = pd.concat([train_df, val_df, test_df])
        all_triples = torch.tensor(all_data.values, dtype=torch.long)
        edge_index = all_triples[:, [0, 2]].t().contiguous().to(device)
        edge_type = all_triples[:, 1].to(device)
        
        # Initialize R-GCN model
        rgcn_model = RGCNDistMult(num_entities, num_relations, self.embed_dim, num_layers=self.layers).to(device)
        optimizer = optim.Adam(rgcn_model.parameters(), lr=self.lr)
        
        train_triples = torch.tensor(train_df.values, dtype=torch.long).to(device)
        
        print_progress("  Using structure-only loss (no DistMult/TransE bias)", verbose)
        
        for epoch in range(self.epochs):
            rgcn_model.train()
            optimizer.zero_grad()
            
            # PURE structure learning: R-GCN forward pass
            entity_embeddings = rgcn_model.forward_entities(edge_index, edge_type)
            
            # STRUCTURE-ONLY LOSS: Simple reconstruction + negative sampling
            # NO scoring method (DistMult/TransE) bias here!
            batch_size = min(1024, len(train_triples))
            batch_idx = torch.randperm(len(train_triples))[:batch_size]
            batch_triples = train_triples[batch_idx]
            
            h, r, t = batch_triples[:, 0], batch_triples[:, 1], batch_triples[:, 2]
            
            # Simple cosine similarity (structure-agnostic)
            h_emb = entity_embeddings[h]
            t_emb = entity_embeddings[t]
            r_emb = rgcn_model.relation_embedding(r)
            
            # Positive similarity (h-r should be similar to t)
            pos_sim = F.cosine_similarity(h_emb + r_emb, t_emb, dim=1)
            
            # Negative sampling
            neg_tails = torch.randint(0, num_entities, (batch_size,), device=device)
            neg_t_emb = entity_embeddings[neg_tails]
            neg_sim = F.cosine_similarity(h_emb + r_emb, neg_t_emb, dim=1)
            
            # Structure-focused loss: maximize pos similarity, minimize neg similarity
            loss = F.relu(0.5 + neg_sim - pos_sim).mean()
            
            loss.backward()
            optimizer.step()
            
            if verbose and epoch % 20 == 0:
                print_progress(f"  R-GCN Structure Epoch {epoch}: Loss = {loss.item():.4f}", verbose)
        
        # Extract final embeddings
        with torch.no_grad():
            final_node_embeddings = rgcn_model.forward_entities(edge_index, edge_type)
            final_rel_embeddings = rgcn_model.relation_embedding.weight
        
        print_progress("R-GCN STRUCTURE training complete (scoring-agnostic)", verbose)
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
# TRAINING UTILITIES AND MISSING FUNCTIONS
# =============================================================================

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

def evaluate_hit_at_k(data_loader, generator, node_emb, rel_emb, device, hit_at_k_list, eval_score_function):
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
                
                scores = eval_score_function(h_exp, r_exp, all_t)
            else:
                # Direct scoring without generator
                h_emb, r_emb_batch = node_emb[h], rel_emb(r)
                
                h_exp = h_emb.unsqueeze(1).expand(-1, node_emb.size(0), -1)
                r_exp = r_emb_batch.unsqueeze(1).expand(-1, node_emb.size(0), -1)
                all_t = node_emb.unsqueeze(0).expand(len(h), -1, -1)
                
                scores = eval_score_function(h_exp, r_exp, all_t)
            
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

# Helper functions
def avg_pairwise_cosine_similarity(tensors):
    norm = F.normalize(tensors, dim=-1)
    sim_matrix = torch.matmul(norm, norm.T)
    upper_tri = sim_matrix.triu(1)
    return upper_tri[upper_tri != 0].mean().item()

def compute_variance(tensors):
    return torch.var(tensors, dim=0).mean().item()

def compute_topk_diversity(fake_embeds, all_node_embeds, k=10):
    sims = torch.matmul(F.normalize(fake_embeds, dim=-1), F.normalize(all_node_embeds, dim=-1).T)
    topk = sims.topk(k, dim=1).indices.view(-1).cpu().numpy()
    unique = len(set(topk))
    return unique / len(topk)

def plot_embedding_space(fake_embeds, real_embeds, epoch, title="Embedding Space Comparison", output_dir="./"):
    """Plot embedding space """
    try:
        combined_embeds = torch.cat([fake_embeds.detach(), real_embeds.detach()], dim=0)
        combined_2d = PCA(n_components=2).fit_transform(combined_embeds.cpu().numpy())
        
        n_fake = fake_embeds.shape[0]
        fake_2d = combined_2d[:n_fake]
        real_2d = combined_2d[n_fake:]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(fake_2d[:, 0], fake_2d[:, 1], alpha=0.5, color='blue', label='Generated Embeddings')
        plt.scatter(real_2d[:, 0], real_2d[:, 1], alpha=0.5, color='green', label='True Embeddings')
        
        plt.title(f"{title} (Epoch {epoch})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/embedding_space_comparison_epoch_{epoch}.png", dpi=300)
        plt.close()  # Close instead of show to prevent display in notebook
    except Exception as e:
        print(f"Warning: Could not create embedding plot: {e}")

# =============================================================================
# [REST OF THE CODE REMAINS THE SAME - DATA LOADING, GAN MODELS, ETC.]
# =============================================================================

def print_progress(message, verbose=True):
    """Conditional printing"""
    if verbose:
        print(message)

# Data loading functions (unchanged)
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

# GAN models (unchanged)
class Generator(nn.Module):
    """Generator architecture"""
    
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
    """Discriminator architecture"""
    
    def __init__(self, embed_dim=128, hidden_dim=1024, dropout=0.3, use_skip=True):
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
        
        x = self.fc1(x_input)
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
        
        if self.use_skip:
            skip = self.skip_linear(x_input).view(-1)
            out = out + skip
        
        return out

# =============================================================================
# UPDATED MAIN TRAINING PIPELINE 
# =============================================================================

def train_prot_b_gan_modular(args):
    """UPDATED: Complete training pipeline with modular architecture"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Prot-B-GAN Modular Training Pipeline")
    print(f"Device: {device}")
    print(f"Embedding method: {args.embedding_init}")
    
    # Resolve reward scoring method
    resolved_reward_method = resolve_reward_method(args.embedding_init, args.reward_scoring_method)
    print(f"Reward scoring method: {resolved_reward_method}")
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
        # STAGE 1: MODULAR EMBEDDING INITIALIZATION
        # =========================================================================
        
        print_progress(f"\nSTAGE 1: {args.embedding_init.upper()} Structure Learning (Scoring-Agnostic)", args.verbose)
        
        embedding_initializer = create_embedding_initializer(args)
        node_embeddings, rel_embeddings = embedding_initializer.initialize_embeddings(
            train_df, val_df, test_df, num_entities, num_relations, device, args.verbose
        )
        
        # Set up embeddings with consistent dimensions
        print_progress(f"Setting up embeddings with dimension: {args.embed_dim}", args.verbose)
        node_emb = nn.Parameter(node_embeddings.clone().detach()).to(device)
        rel_emb = nn.Embedding(num_relations, args.embed_dim).to(device)
        rel_emb.weight.data.copy_(rel_embeddings)
        
        print_progress(f" Embeddings initialized: Nodes {node_emb.shape}, Relations {rel_emb.weight.shape}", args.verbose)
        
        # Get scoring functions
        reward_score_function = get_reward_scoring_function(resolved_reward_method)
        eval_score_function = get_evaluation_scoring_function(args.embedding_init)
        
        # =========================================================================
        # STAGE 2: MODULAR WARM-UP (ALWAYS PERFORMED)
        # =========================================================================
        
        print_progress(f"\nSTAGE 2: Modular Warm-up ({args.embedding_init} → {resolved_reward_method} alignment)", args.verbose)
        
        # ALWAYS do modular warm-up for scoring alignment
        modular_warmup(node_emb, rel_emb, train_loader, device, args.embedding_init, resolved_reward_method,
                      epochs=args.warmup_epochs, lr=args.warmup_lr, verbose=args.verbose)
        
        # Evaluate after warm-up
        print_progress("Evaluating post-warmup performance...", args.verbose)
        baseline_hit_at_k = evaluate_hit_at_k(val_loader, None, node_emb, rel_emb, device, args.hit_at_k, eval_score_function)
        print_progress(f"Post-warmup Hit@10: {baseline_hit_at_k[10]:.4f}", args.verbose)
        
        # =========================================================================
        # STAGE 3: 3-TIER PROGRESSIVE GAN-RL TRAINING
        # =========================================================================
        
        print_progress(f"\nSTAGE 3: 3-Tier Progressive GAN-RL Training (Modular Architecture)", args.verbose)
        
        # Initialize models with consistent dimensions
        generator = Generator(args.embed_dim, args.noise_dim).to(device)
        discriminator = Discriminator(args.embed_dim, args.hidden_dim).to(device)
        
        g_opt = optim.Adam(list(generator.parameters()) + [node_emb], lr=args.g_lr)
        d_opt = optim.Adam(discriminator.parameters(), lr=args.d_lr)
        
        print_progress(f"Model Architecture (embed_dim={args.embed_dim}):", args.verbose)
        print_progress(f"   Generator parameters: {sum(p.numel() for p in generator.parameters()):,}", args.verbose)
        print_progress(f"   Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}", args.verbose)
        
        print_progress(f"Training Schedule:", args.verbose)
        print_progress(f"  Tier 1-A (Epochs 1-{args.pretrain_epochs}): Frozen embedding pretraining", args.verbose)
        print_progress(f"  Tier 1-B (Epochs {args.pretrain_epochs+1}+): Unfrozen basic adversarial", args.verbose)
        print_progress(f"  Tier 2 (Epochs +{args.rl_start_epoch}): + RL system ({resolved_reward_method} scoring)", args.verbose)
        print_progress(f"  Tier 3 (Epochs +{args.full_system_epoch}): + Full adversarial", args.verbose)
        
        # Training state
        best_val_hit10 = 0.0
        best_epoch = 0
        start_epoch = 1
        current_tier = "pretraining"
        
        # TIER EPOCH COUNTERS
        tier_epoch_count = 0
        
        # Pretraining variables
        best_pretrain_state = None
        pretrain_no_improve = 0
        prev_pretrain_hit10 = 0.0
        
        # Dynamic training schedule
        current_d_freq = args.d_update_freq
        current_g_steps = 1
        g_step_counter = 0
        
        # Training history
        training_history = {
            'losses': [], 'cos_sims': [], 'd_losses': [], 'g_losses': [],
            'train_hitks': {k: [] for k in args.hit_at_k},
            'val_hitks': {k: [] for k in args.hit_at_k},
            'f1_history': [], 'val_f1_history': [],
            'aupr_history': [], 'val_aupr_history': [],
            'mcc_history': [], 'val_mcc_history': [],
            'auc_history': [], 'val_auc_history': [],
        }
        
        bce_loss = nn.BCEWithLogitsLoss()
        
        print_progress(f"Starting full 3-tier training loop...", args.verbose)
        
        # =========================================================================
        # FULL TRAINING LOOP WITH ALL MISSING FUNCTIONS
        # =========================================================================
        
        for epoch in range(start_epoch, args.epochs + 1):
            
            # =================================================================
            # PRETRAINING PHASE (Tier 1-A: Frozen Embeddings)
            # =================================================================
            
            if epoch <= args.pretrain_epochs and current_tier == "pretraining":
                generator.train()
                total_pretrain_loss = 0.0
                total_pretrain_steps = 0
                
                for step, batch in enumerate(train_loader):
                    if args.debug and step >= 50:
                        break
                    
                    h, r, t = [b.to(device) for b in batch.T]
                    h_emb, r_emb_batch, t_emb = node_emb[h], rel_emb(r), node_emb[t]
                    
                    # Sample negatives 
                    batch_size = h_emb.size(0)
                    neg_idx = torch.randint(0, node_emb.size(0), (batch_size, args.n_pre_neg), device=device)
                    neg_t_embs = node_emb[neg_idx]
                    
                    # Generator forward with pretraining losses
                    g_opt.zero_grad()
                    fake = generator(h_emb, r_emb_batch)
                    
                    # Reconstruction + cosine loss
                    rec_loss = F.mse_loss(fake, t_emb) + 0.1 * (1 - F.cosine_similarity(fake, t_emb).mean())
                    
                    # Margin loss with multiple negatives
                    fake_exp = fake.unsqueeze(1).expand(-1, args.n_pre_neg, -1)
                    pos_cos = F.cosine_similarity(fake, t_emb, dim=-1)
                    neg_cos = F.cosine_similarity(fake_exp, neg_t_embs, dim=-1)
                    margin_term = 0.35 + neg_cos - pos_cos.unsqueeze(1)
                    margin_loss = F.relu(margin_term).mean()
                    
                    # Pretraining loss composition
                    pretrain_loss = rec_loss + args.alpha_pretrain * margin_loss
                    pretrain_loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_norm)
                    g_opt.step()
                    
                    total_pretrain_loss += pretrain_loss.item()
                    total_pretrain_steps += 1
                
                # Validate pretraining progress 
                val_hitks = evaluate_hit_at_k(val_loader, generator, node_emb, rel_emb, device, args.hit_at_k, eval_score_function)
                this_hit10 = val_hitks[10]
                
                if this_hit10 > prev_pretrain_hit10 + 1e-3:
                    prev_pretrain_hit10 = this_hit10
                    best_pretrain_state = {
                        "generator": generator.state_dict(),
                        "node_emb": node_emb.detach().cpu().clone()
                    }
                    pretrain_no_improve = 0
                    print_progress(f"[Pretraining] Epoch {epoch}: Hit@10 improved to {this_hit10:.4f}, saving snapshot", args.verbose)
                else:
                    pretrain_no_improve += 1
                    print_progress(f"[Pretraining] Epoch {epoch}: no improvement ({this_hit10:.4f} <= {prev_pretrain_hit10:.4f}), patience {pretrain_no_improve}/{args.pretrain_patience}", args.verbose)
                
                # Restore best pretrain state 
                if best_pretrain_state:
                    generator.load_state_dict(best_pretrain_state["generator"])
                    node_emb.data.copy_(best_pretrain_state["node_emb"].to(device))
                
                # Early stopping check
                if pretrain_no_improve >= args.pretrain_patience:
                    print_progress(f"[Pretraining] Early stopping at epoch {epoch}", args.verbose)
                    current_tier = 1
                    tier_epoch_count = 0
                    node_emb.requires_grad_(False)
                    g_opt = optim.Adam(generator.parameters(), lr=1e-4)
                    continue
                
                continue
            
            # Transition to Tier 1-B after pretraining
            if epoch == args.pretrain_epochs + 1 and current_tier == "pretraining":
                current_tier = 1
                tier_epoch_count = 0
                node_emb.requires_grad_(False)
                g_opt = optim.Adam(generator.parameters(), lr=1e-4)
                print_progress("Transitioning to Tier 1-B: Basic Adversarial Training", args.verbose)
            
            # =================================================================
            # TIER 1-B, 2 & 3: PROGRESSIVE ADVERSARIAL TRAINING
            # =================================================================
            
            # Update tier based on epoch count
            tier_epoch_count += 1
            
            if tier_epoch_count >= args.rl_start_epoch and current_tier == 1:
                current_tier = 2
                print_progress(f"Transitioning to Tier 2: RL Training with {resolved_reward_method} scoring (Epoch {epoch})", args.verbose)
            elif tier_epoch_count >= args.full_system_epoch and current_tier == 2:
                current_tier = 3
                print_progress(f"Transitioning to Tier 3: Full Adversarial System (Epoch {epoch})", args.verbose)
            
            generator.train()
            discriminator.train()
            
            total_loss, total_cos = 0.0, 0.0
            total_d_loss, total_g_loss = 0.0, 0.0
            true_labels, pred_probs = [], []
            
            for step, batch in enumerate(train_loader):
                if args.debug and step >= 50:
                    break
                
                h, r, t = [b.to(device) for b in batch.T]
                h_emb, r_emb_batch, t_emb = node_emb[h], rel_emb(r), node_emb[t]
                
                # =============================================================
                # DISCRIMINATOR TRAINING (Tier 2 & 3) 
                # =============================================================
                
                if current_tier >= 2 and step % current_d_freq == 0 and g_step_counter >= current_g_steps:
                    d_opt.zero_grad()
                    
                    # Real samples
                    real_scores = discriminator(h_emb.detach(), r_emb_batch.detach(), t_emb.detach())
                    
                    # Generator fake samples
                    with torch.no_grad():
                        fake_samples = generator(h_emb.detach(), r_emb_batch.detach())
                    fake_scores = discriminator(h_emb.detach(), r_emb_batch.detach(), fake_samples)
                    
                    # Hard negative samples
                    batch_size = h_emb.shape[0]
                    neg_indices = torch.randint(0, node_emb.size(0), (batch_size, 25), device=device)
                    selected_neg_idx = torch.randint(0, 25, (batch_size,), device=h_emb.device)
                    final_neg_idx = torch.gather(neg_indices, 1, selected_neg_idx.unsqueeze(1)).squeeze(1)
                    hard_neg_samples = node_emb[final_neg_idx]
                    hard_neg_scores = discriminator(h_emb.detach(), r_emb_batch.detach(), hard_neg_samples)
                    
                    # Soft labels 
                    real_labels = torch.full_like(real_scores, 0.8)
                    fake_labels = torch.full_like(fake_scores, 0.2)
                    hard_neg_labels = torch.full_like(hard_neg_scores, 0.1)
                    
                    # Balanced loss computation
                    d_loss = (bce_loss(real_scores, real_labels) +
                              bce_loss(fake_scores, fake_labels) +
                              bce_loss(hard_neg_scores, hard_neg_labels)) / 3
                    
                    d_loss.backward()
                    d_opt.step()
                    total_d_loss += d_loss.item()
                    
                    # Reset generator counter
                    g_step_counter = 0
                    
                    # Track discriminator performance
                    with torch.no_grad():
                        all_real_scores = real_scores
                        all_fake_scores = torch.cat([fake_scores, hard_neg_scores])
                        
                        true_labels.extend([1]*len(all_real_scores) + [0]*len(all_fake_scores))
                        pred_probs.extend(torch.sigmoid(torch.cat([all_real_scores, all_fake_scores])).cpu().numpy())
                
                # =============================================================
                # GENERATOR TRAINING with method-specific reward scoring
                # =============================================================
                
                g_opt.zero_grad()
                fake = generator(h_emb, r_emb_batch)
                
                loss_components = []
                
                # Refinement loss 
                refinement_loss = F.mse_loss(fake, t_emb)
                loss_components.append(args.refinement_weight * refinement_loss)
                
                # Method-specific component loss
                eval_scores = eval_score_function(h_emb, r_emb_batch, fake)
                eval_loss = -torch.tanh(eval_scores / 10.0).mean()
                loss_components.append(args.distmult_weight * eval_loss)
                
                # RL loss using method-specific reward scoring
                if current_tier >= 2:
                    method_rewards = reward_score_function(h_emb, r_emb_batch, fake)
                    if current_tier == 3:
                        # Include discriminator rewards in Tier 3
                        with torch.no_grad():
                            disc_scores = discriminator(h_emb.detach(), r_emb_batch.detach(), fake.detach())
                            disc_rewards = torch.sigmoid(disc_scores) - 0.5
                        composite_rewards = 0.7 * method_rewards + 0.3 * disc_rewards
                        rl_loss = -composite_rewards.mean()
                    else:
                        rl_loss = -method_rewards.mean()
                    
                    loss_components.append(args.rl_weight * rl_loss)
                
                # Bias mitigation (hard negative mining)
                with torch.no_grad():
                    sim = torch.matmul(F.normalize(t_emb, dim=-1), F.normalize(node_emb, dim=-1).T)
                    _, hard_neg_idx = sim.topk(3, dim=1)
                    selected_idx = torch.randint(0, 3, (h_emb.shape[0],), device=h_emb.device)
                    final_neg_idx = torch.gather(hard_neg_idx, 1, selected_idx.unsqueeze(1)).squeeze(1)
                    hard_neg_emb = node_emb[final_neg_idx]
                
                pos_cos = F.cosine_similarity(fake, t_emb, dim=-1)
                hard_neg_cos = F.cosine_similarity(fake, hard_neg_emb, dim=-1)
                margin_loss = F.relu(0.2 + hard_neg_cos - pos_cos).mean()
                bias_loss = margin_loss + 0.1 * torch.exp(-2 * torch.var(fake, dim=0).mean())
                loss_components.append(args.bias_weight * bias_loss)
                
                # Adversarial loss (Tier 3 only)
                if current_tier == 3:
                    adv_scores = discriminator(h_emb, r_emb_batch, fake)
                    adv_loss = -torch.tanh(adv_scores / 5.0).mean()
                    loss_components.append(args.adv_weight * adv_loss)
                
                # Cosine margin loss with hard negatives
                batch_size = h_emb.shape[0]
                rand_k = args.n_neg - args.hard_neg_k
                rand_idxs = torch.randint(0, node_emb.size(0), (batch_size, rand_k), device=device)
                hard_idxs = torch.randint(0, node_emb.size(0), (batch_size, args.hard_neg_k), device=device)
                neg_indices_gen = torch.cat([hard_idxs, rand_idxs], dim=1)
                neg_t_embs = node_emb[neg_indices_gen]
                
                fake_expanded = fake.unsqueeze(1).expand(-1, args.n_neg, -1)
                pos_cos = F.cosine_similarity(fake, t_emb, dim=-1)
                neg_cos = F.cosine_similarity(fake_expanded, neg_t_embs, dim=-1)
                margin_term = 0.30 + neg_cos - pos_cos.unsqueeze(1)
                cos_margin = F.relu(margin_term).mean()
                
                # L2 regularization and diversity losses 
                l2_loss = (fake - t_emb.detach()).pow(2).mean()
                fake_std = fake.std(dim=0).mean()
                diversity_loss = 0.2 * torch.exp(-5 * fake_std)
                
                # Final generator loss
                final_loss = sum(loss_components) + args.g_guidance_weight * cos_margin + args.l2_reg_weight * l2_loss + diversity_loss
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_norm)
                g_opt.step()
                total_g_loss += final_loss.item()
                
                # Track generator steps
                g_step_counter += 1
                
                # Track training metrics
                with torch.no_grad():
                    cos_sim = F.cosine_similarity(fake, t_emb).mean().item()
                    total_loss += final_loss.item()
                    total_cos += cos_sim
            
            # =================================================================
            # EPOCH-LEVEL EVALUATION 
            # =================================================================
            
            effective_steps = min(len(train_loader), 50 if args.debug else len(train_loader))
            avg_loss = total_loss / effective_steps if effective_steps > 0 else 0.0
            avg_cos = total_cos / effective_steps if effective_steps > 0 else 0.0
            avg_d_loss = total_d_loss / max(1, effective_steps // current_d_freq) if current_tier >= 2 else 0.0
            avg_g_loss = total_g_loss / effective_steps if effective_steps > 0 else 0.0
            
            # Update training history
            training_history['losses'].append(avg_loss)
            training_history['cos_sims'].append(avg_cos)
            training_history['d_losses'].append(avg_d_loss)
            training_history['g_losses'].append(avg_g_loss)
            
            # Compute discriminator metrics 
            if true_labels:
                enhanced_metrics = compute_metrics(np.array(true_labels), np.array(pred_probs))
                training_history['f1_history'].append(enhanced_metrics['F1'])
                training_history['aupr_history'].append(enhanced_metrics['AUPR'])
                training_history['mcc_history'].append(enhanced_metrics['MCC'])
                training_history['auc_history'].append(enhanced_metrics['AUC'])
            else:
                enhanced_metrics = {'F1': 0, 'AUPR': 0, 'MCC': 0, 'AUC': 0}
            
            # Hit@K evaluation
            train_hit_at_k = evaluate_hit_at_k(train_loader, generator, node_emb, rel_emb, device, args.hit_at_k, eval_score_function)
            for k in args.hit_at_k:
                training_history['train_hitks'][k].append(train_hit_at_k[k])
            
            # Validation
            val_metrics, val_cos_avg = validate(val_loader, generator, discriminator, node_emb, rel_emb, device)
            training_history['val_f1_history'].append(val_metrics['F1'])
            training_history['val_aupr_history'].append(val_metrics['AUPR'])
            training_history['val_mcc_history'].append(val_metrics['MCC'])
            training_history['val_auc_history'].append(val_metrics['AUC'])
            
            val_hit_at_k = evaluate_hit_at_k(val_loader, generator, node_emb, rel_emb, device, args.hit_at_k, eval_score_function)
            for k in args.hit_at_k:
                training_history['val_hitks'][k].append(val_hit_at_k[k])
            
            current_hit10 = val_hit_at_k[10]
            
            # Update best model
            if current_hit10 > best_val_hit10:
                best_val_hit10 = current_hit10
                best_epoch = epoch
                
                # Save checkpoint (simplified)
                os.makedirs(args.output_dir, exist_ok=True)
                checkpoint_path = os.path.join(args.output_dir, "checkpoint.pt")
                torch.save({
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "node_emb": node_emb.detach().cpu(),
                    "rel_emb": rel_emb.state_dict(),
                    "epoch": epoch,
                    "best_val_hit10": best_val_hit10,
                    "best_epoch": best_epoch,
                    "training_history": training_history
                }, checkpoint_path)
            
            # Progress logging 
            if current_tier == "pretraining":
                tier_name = "Pretraining"
            else:
                tier_name = f"Tier {current_tier}"
            
            print_progress(f"[{tier_name}] E{epoch:03d} | Loss {avg_loss:.4f} | CosSim {avg_cos:.3f} | "
                          f"F1 {enhanced_metrics['F1']:.4f}", args.verbose)
            
            print_progress("Train: " + " ".join([f"Hit@{k}: {train_hit_at_k[k]:.4f}" for k in args.hit_at_k]), args.verbose)
            print_progress("Val:   " + " ".join([f"Hit@{k}: {val_hit_at_k[k]:.4f}" for k in args.hit_at_k]) + f" | F1 {val_metrics['F1']:.4f}", args.verbose)
            
            # Early stopping
            if epoch - best_epoch > args.early_stopping_patience:
                print_progress(f"Early stopping at epoch {epoch}", args.verbose)
                break
        
        # =========================================================================
        # FINAL EVALUATION
        # =========================================================================
        
        print(f"\nTRAINING COMPLETED!")
        print(f"Best validation Hit@10: {best_val_hit10:.4f} achieved at epoch {best_epoch}")
        
        # Final test evaluation
        test_hit_at_k = evaluate_hit_at_k(test_loader, generator, node_emb, rel_emb, device, args.hit_at_k, eval_score_function)
        final_val_metrics, _ = validate(val_loader, generator, discriminator, node_emb, rel_emb, device)
        
        print(f"\nFINAL TEST RESULTS:")
        for k in args.hit_at_k:
            print(f"Test Hit@{k}: {test_hit_at_k[k]:.4f} ({test_hit_at_k[k]*100:.1f}%)")
        
        # Final results
        final_results = {
            "method": args.embedding_init,
            "reward_method": resolved_reward_method,
            "baseline_hit_at_k": baseline_hit_at_k,
            "test_hit_at_k": test_hit_at_k,
            "best_val_hit10": best_val_hit10,
            "best_epoch": best_epoch,
            "final_discriminator_health": final_val_metrics,
            "training_history": training_history,
            "improvement": {k: test_hit_at_k[k] - baseline_hit_at_k[k] for k in args.hit_at_k}
        }
        
        return final_results
        
    except Exception as e:
        print(f"Training failed: {e}")
        return None

# =============================================================================
# COMMAND LINE INTERFACE (UPDATED)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prot-B-GAN: Modular Architecture with Separated Structure and Scoring')
    
    # Data and I/O
    parser.add_argument('--data_root', type=str, required=True, help='Root directory containing data files')
    parser.add_argument('--train_file', type=str, default='prothgt-train-graph_triplets.csv', help='Training data file')
    parser.add_argument('--val_file', type=str, default='prothgt-val-graph_triplets.csv', help='Validation data file')
    parser.add_argument('--test_file', type=str, default='prothgt-test-graph_triplets.csv', help='Test data file')
    parser.add_argument('--output_dir', type=str, default='./modular_results', help='Output directory')
    
    # Model Architecture 
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension (consistent throughout)')
    parser.add_argument('--noise_dim', type=int, default=64, help='Generator noise dimension')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Discriminator hidden dimension')
    
    # Training Parameters 
    parser.add_argument('--epochs', type=int, default=500, help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--g_lr', type=float, default=1e-3, help='Generator learning rate')
    parser.add_argument('--d_lr', type=float, default=1e-4, help='Discriminator learning rate')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Gradient clipping')
    
    # MODULAR: Embedding + Reward Methods
    parser.add_argument('--embedding_init', type=str, default='rgcn', 
                        choices=['rgcn', 'random', 'transe', 'distmult', 'complex'],
                        help='Embedding initialization method (structure learning)')
    parser.add_argument('--reward_scoring_method', type=str, default='distmult',
                        choices=['auto', 'transe', 'distmult', 'complex', 'universal_distmult'],
                        help='Reward scoring method (for warm-up and RL training)')
    
    # Method-specific parameters (only relevant ones based on embedding_init)
    parser.add_argument('--rgcn_epochs', type=int, default=100, help='R-GCN structure epochs')
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
    
    # Modular warm-up (ALWAYS performed)
    parser.add_argument('--warmup_epochs', type=int, default=50, help='Modular warm-up epochs')
    parser.add_argument('--warmup_lr', type=float, default=1e-2, help='Modular warm-up learning rate')
    
    # Progressive training 
    parser.add_argument('--pretrain_epochs', type=int, default=90, help='Generator pretraining epochs')
    parser.add_argument('--pretrain_patience', type=int, default=20, help='Pretraining early stopping patience')
    parser.add_argument('--rl_start_epoch', type=int, default=20, help='RL training start epoch (Tier 2)')
    parser.add_argument('--full_system_epoch', type=int, default=25, help='Full adversarial start epoch (Tier 3)')
    parser.add_argument('--d_update_freq', type=int, default=10, help='Discriminator update frequency')
    
    # Loss weights (ALL components)
    parser.add_argument('--rl_weight', type=float, default=0.1, help='RL loss weight')
    parser.add_argument('--adv_weight', type=float, default=0.05, help='Adversarial loss weight')
    parser.add_argument('--refinement_weight', type=float, default=0.7, help='Refinement loss weight')
    parser.add_argument('--distmult_weight', type=float, default=0.2, help='Method-specific loss weight')
    parser.add_argument('--bias_weight', type=float, default=0.1, help='Bias mitigation weight')
    parser.add_argument('--g_guidance_weight', type=float, default=2.0, help='Generator guidance weight')
    parser.add_argument('--l2_reg_weight', type=float, default=0.1, help='L2 regularization weight')
    
    # Hard negative mining 
    parser.add_argument('--n_neg', type=int, default=50, help='Number of negative samples')
    parser.add_argument('--hard_neg_k', type=int, default=50, help='Number of hard negatives')
    parser.add_argument('--n_pre_neg', type=int, default=30, help='Number of pretraining negatives')
    parser.add_argument('--alpha_pretrain', type=float, default=1.5, help='Pretraining alpha weight')
    
    # Evaluation
    parser.add_argument('--hit_at_k', type=int, nargs='+', default=[1, 5, 10], help='Hit@K values')
    parser.add_argument('--early_stopping_patience', type=int, default=30, help='Early stopping patience')
    
    # Debug and misc
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Resolve reward method
    resolved_reward_method = resolve_reward_method(args.embedding_init, args.reward_scoring_method)
    
    print(f"MODULAR ARCHITECTURE:")
    print(f"  Structure Learning: {args.embedding_init.upper()} (scoring-agnostic)")
    print(f"  Scoring Alignment: {resolved_reward_method.upper()} warm-up")
    print(f"  Reward Function: {resolved_reward_method.upper()} RL")
    print(f"  Consistent Embedding Dimension: {args.embed_dim}")
    
    # Run training
    results = train_prot_b_gan_modular(args)
    
    if results:
        print(f"\nSUCCESS! Modular {args.embedding_init.upper()} + {resolved_reward_method.upper()} setup complete")
        if 'baseline_hit10' in results:
            print(f"Baseline Hit@10 after warm-up: {results['baseline_hit10']:.4f}")
        return 0
    else:
        print(f"Setup failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
