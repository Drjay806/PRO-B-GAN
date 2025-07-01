"""
Prot-B-GAN
==================================================================

A complete pipeline that includes training functions:
- Load pre-trained embeddings OR train from scratch
- Resume from checkpoint capability
- 3-tier progressive training system
-  loss functions and hard negative mining
- Method-consistent scoring throughout

Installation (Google Colab/CUDA 11.8):
    pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install torch-geometric scikit-learn pandas matplotlib tqdm

Usage Examples:

    # Mode 1: Load your pre-trained embeddings (RECOMMENDED)
    python prot_b_gan_complete.py \
        --data_root "/path/to/FB15k-237/converted" \
        --load_embeddings \
        --node_emb_path "/path/to/FB15k-237_final_node_embeddings.pt" \
        --rel_emb_path "/path/to/FB15k-237_final_distmult_rel_emb.pt" \
        --rel_map_path "/path/to/FB15k-237_relation_map.pkl" \
        --embed_dim 500 --epochs 500 --batch_size 64 \
        --pretrain_epochs 200 --rl_start_epoch 20 --full_system_epoch 25 \
        --verbose

    # Mode 2: Train from scratch
    python prot_b_gan_complete.py \
        --data_root "/path/to/data" \
        --embedding_init rgcn --embed_dim 500 --epochs 500 \
        --verbose

    # Mode 3: Resume from checkpoint
    python prot_b_gan_complete.py \
        --data_root "/path/to/data" \
        --resume_checkpoint "/path/to/checkpoint.pt" \
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
# SCORING FUNCTIONS 
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
        'random': distmult_score  
    }
    
    if method.lower() not in scoring_functions:
        raise ValueError(f"Unknown scoring method: {method}")
    
    return scoring_functions[method.lower()]

# =============================================================================
# FUNCTIONS
# =============================================================================

def compute_composite_rl_loss(tier2_epoch, tier3_epoch, current_tier, h_emb, r_emb, fake, t_emb, discriminator, rl_start_epoch, full_system_epoch, score_function):
    """three-tier RL loss progression"""
    
    if current_tier == 1:
        return torch.tensor(0.0, device=h_emb.device), {}
    
    elif current_tier == 2:
        # Tier 2: Method-specific scoring only
        if tier2_epoch < rl_start_epoch:
            return torch.tensor(0.0, device=h_emb.device), {}
        
        method_scores = score_function(h_emb, r_emb, fake)
        method_component = torch.sigmoid(method_scores / 5.0)
        simple_reward = method_component - 0.5
        rl_loss = -simple_reward.mean()
        return rl_loss, {'method_component': method_component.mean().item()}
    
    else:  # current_tier == 3
        # Tier 3: Full composite RL
        if tier3_epoch < full_system_epoch:
            # Still in transition, use method scoring only
            method_scores = score_function(h_emb, r_emb, fake)
            method_component = torch.sigmoid(method_scores / 5.0)
            simple_reward = method_component - 0.5
            rl_loss = -simple_reward.mean()
            return rl_loss, {'method_component': method_component.mean().item()}
        else:
            # Full composite system
            method_scores = score_function(h_emb, r_emb, fake)
            method_component = torch.sigmoid(method_scores / 5.0)
            
            with torch.no_grad():
                disc_scores = discriminator(h_emb.detach(), r_emb.detach(), fake.detach())
                disc_component = torch.sigmoid(disc_scores)
            
            composite_reward = 0.3 * disc_component + 0.7 * method_component - 0.5
            rl_loss = -composite_reward.mean()
            
            return rl_loss, {
                'disc_component': disc_component.mean().item(),
                'method_component': method_component.mean().item(),
                'composite_reward': composite_reward.mean().item()
            }

def compute_bias_mitigation_loss(fake, t_emb, node_emb, h_emb, r_emb):
    """ bias mitigation through hard negative mining"""
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

def generate_balanced_hard_negatives(h_emb, r_emb, node_emb, score_function, num_hard=10, num_medium=8, num_easy=7):
    """ sophisticated hard negative generation strategy"""
    batch_size = h_emb.shape[0]
    
    with torch.no_grad():
        # Calculate scores for all entities using the appropriate method
        h_expand = h_emb.unsqueeze(1).expand(-1, node_emb.size(0), -1)
        r_expand = r_emb.unsqueeze(1).expand(-1, node_emb.size(0), -1)
        all_nodes = node_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        all_scores = score_function(h_expand, r_expand, all_nodes)
        
        # Get different difficulty levels
        hard_vals, hard_idxs = all_scores.topk(num_hard, dim=1)
        
        sorted_scores, sorted_idxs = all_scores.sort(dim=1, descending=True)
        start_idx = node_emb.size(0) // 3
        end_idx = start_idx + num_medium
        medium_idxs = sorted_idxs[:, start_idx:end_idx]
        
        easy_idxs = torch.randint(0, node_emb.size(0), (batch_size, num_easy), device=h_emb.device)
        
        all_neg_idxs = torch.cat([hard_idxs, medium_idxs, easy_idxs], dim=1)
    
    return all_neg_idxs

def print_enhanced_discriminator_metrics(true_labels, pred_probs, epoch, verbose=True):
    """Enhanced discriminator metrics"""
    if not true_labels:
        return {"F1": 0, "AUPR": 0, "MCC": 0, "AUC": 0, "Precision": 0, "Recall": 0, "Gap": 0}

    y_true = np.array(true_labels)
    y_probs = np.array(pred_probs)

    # Calculate discriminator gap
    real_probs = [p for i, p in enumerate(y_probs) if y_true[i] == 1]
    fake_probs = [p for i, p in enumerate(y_probs) if y_true[i] == 0]
    gap = np.mean(real_probs) - np.mean(fake_probs) if real_probs and fake_probs else 0

    # Standard metrics
    y_pred = (y_probs >= 0.5).astype(int)
    
    try:
        metrics = {
            "F1": float(f1_score(y_true, y_pred, zero_division=0)),
            "AUPR": float(average_precision_score(y_true, y_probs)),
            "MCC": float(matthews_corrcoef(y_true, y_pred)),
            "AUC": float(roc_auc_score(y_true, y_probs)) if len(set(y_true)) > 1 else 0.0,
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "Gap": float(gap)
        }
    except Exception as e:
        if verbose:
            print(f"Warning: Metrics calculation failed: {e}")
        return {"F1": 0, "AUPR": 0, "MCC": 0, "AUC": 0, "Precision": 0, "Recall": 0, "Gap": 0}

    # Health assessment
    if gap > 0.05 and metrics['F1'] > 0.60:
        status = " HEALTHY "
    elif gap > 0.03 and metrics['F1'] > 0.50:
        status = " WEAK but learning"
    elif gap < 0.01:
        status = " BROKEN (like before the fix)"
    else:
        status = " LEARNING"

    if verbose:
        print(f"   Discriminator Health: Gap={gap:.3f} F1={metrics['F1']:.3f} Precision={metrics['Precision']:.3f} Recall={metrics['Recall']:.3f} AUC={metrics['AUC']:.3f}")
        print(f"     Status: {status}")

    return metrics

def rebalance_training_schedule(epoch, disc_f1):
    """Adjust training frequency based on discriminator dominance"""
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

# =============================================================================
# SMART WARM-UP LOGIC 
# =============================================================================

def needs_warmup(embedding_method):
    """Determine if warm-up is needed - only when changing objectives"""
    warmup_needed = {
        'rgcn': True,     # R-GCN → DistMult (structure → multiplicative)
        'random': True,   # Random → Method (no training → trained)
        'transe': False,  # TransE 
        'distmult': False, # DistMult 
        'complex': False  # ComplEx 
    }
    return warmup_needed.get(embedding_method.lower(), False)

def smart_warmup(node_emb, rel_emb, train_loader, device, method, epochs=50, lr=1e-2, verbose=True):
    """Smart warm-up: only train when changing objectives"""
    
    if not needs_warmup(method):
        print_progress(f" Skipping warm-up for {method} (already trained with same objective)", verbose)
        return
    
    if method.lower() == 'rgcn':
        # R-GCN → DistMult alignment
        print_progress(f" R-GCN → DistMult alignment warm-up ({epochs} epochs)...", verbose)
        target_score_fn = get_scoring_function('distmult')
    elif method.lower() == 'random':
        # Random → DistMult training (could be any method, defaulting to DistMult)
        print_progress(f" Random → DistMult training ({epochs} epochs)...", verbose)
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
# EMBEDDING LOADING AND CHECKPOINT UTILITIES
# =============================================================================

def load_pretrained_embeddings(args, num_entities, num_relations, device, verbose=True):
    """Load pre-trained embeddings from files """
    
    if not all([args.node_emb_path, args.rel_emb_path]):
        raise ValueError("Both --node_emb_path and --rel_emb_path must be provided when using --load_embeddings")
    
    print_progress(f" Loading pre-trained embeddings...", verbose)
    print_progress(f"   Node embeddings: {args.node_emb_path}", verbose)
    print_progress(f"   Relation embeddings: {args.rel_emb_path}", verbose)
    
    try:
        # Load node embeddings
        node_embeddings = torch.load(args.node_emb_path, map_location=device)
        print_progress(f"   Loaded node embeddings: {node_embeddings.shape}", verbose)
        
        # Load relation embeddings
        if args.rel_map_path:
            # Load relation map and create embedding layer
            with open(args.rel_map_path, "rb") as f:
                relation_map = pickle.load(f)
            rel_emb_layer = nn.Embedding(len(relation_map), args.embed_dim).to(device)
            rel_emb_layer.load_state_dict(torch.load(args.rel_emb_path, map_location=device))
            rel_embeddings = rel_emb_layer.weight.detach()
            print_progress(f"   Loaded relation embeddings: {rel_embeddings.shape}", verbose)
        else:
            # Direct relation embedding file
            rel_embeddings = torch.load(args.rel_emb_path, map_location=device)
            print_progress(f"   Loaded relation embeddings: {rel_embeddings.shape}", verbose)
        
        # Verify dimensions
        if node_embeddings.shape[0] != num_entities:
            print_progress(f"     Node embedding count mismatch. Expected {num_entities}, got {node_embeddings.shape[0]}", verbose)
        if rel_embeddings.shape[0] != num_relations:
            print_progress(f"     Relation embedding count mismatch. Expected {num_relations}, got {rel_embeddings.shape[0]}", verbose)
        if node_embeddings.shape[1] != args.embed_dim or rel_embeddings.shape[1] != args.embed_dim:
            print_progress(f"     Embedding dimension mismatch. Expected {args.embed_dim}", verbose)
        
        print_progress(f" Pre-trained embeddings loaded successfully", verbose)
        return node_embeddings, rel_embeddings
        
    except Exception as e:
        print_progress(f"  Failed to load pre-trained embeddings: {e}", verbose)
        print_progress(f"   Falling back to training from scratch...", verbose)
        return None, None

def save_checkpoint(generator, discriminator, node_emb, rel_emb, g_opt, d_opt, epoch, 
                   training_history, args, best_val_hit10, best_epoch, additional_data=None):
    """Save training checkpoint with all necessary state"""
    
    checkpoint = {
        # Model states
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "node_emb": node_emb.detach().cpu(),
        "rel_emb": rel_emb.state_dict(),
        
        # Optimizer states
        "g_opt_state": g_opt.state_dict(),
        "d_opt_state": d_opt.state_dict(),
        
        # Training state
        "epoch": epoch,
        "best_val_hit10": best_val_hit10,
        "best_epoch": best_epoch,
        
        # Configuration
        "args": vars(args),
        "embed_dim": args.embed_dim,
        "num_entities": node_emb.shape[0],
        "num_relations": rel_emb.num_embeddings,
        
        # Training history
        "training_history": training_history,
        
        # Additional data
        "additional_data": additional_data or {}
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pt")
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, generator, discriminator, node_emb, rel_emb, g_opt, d_opt, device, force_resume=False, verbose=True):
    """Load training checkpoint and restore all state"""
    
    if not os.path.exists(checkpoint_path):
        print_progress(f" Checkpoint not found: {checkpoint_path}", verbose)
        return None
    
    try:
        print_progress(f" Loading checkpoint: {checkpoint_path}", verbose)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model states
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        
        # Load embeddings
        saved_node_emb = checkpoint["node_emb"].to(device)
        node_emb.data.copy_(saved_node_emb)
        rel_emb.load_state_dict(checkpoint["rel_emb"])
        
        # Load optimizer states
        if "g_opt_state" in checkpoint and "d_opt_state" in checkpoint:
            g_opt.load_state_dict(checkpoint["g_opt_state"])
            d_opt.load_state_dict(checkpoint["d_opt_state"])
            
            # Move optimizer states to device
            for state in g_opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            for state in d_opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        
        # Extract checkpoint info
        resume_epoch = checkpoint.get("epoch", 0) + 1
        best_val_hit10 = checkpoint.get("best_val_hit10", 0.0)
        best_epoch = checkpoint.get("best_epoch", 0)
        training_history = checkpoint.get("training_history", {})
        
        print_progress(f"   Checkpoint loaded successfully", verbose)
        print_progress(f"   Resuming from epoch {resume_epoch}", verbose)
        print_progress(f"   Best validation Hit@10: {best_val_hit10:.4f} (epoch {best_epoch})", verbose)
        
        return {
            "resume_epoch": resume_epoch,
            "best_val_hit10": best_val_hit10,
            "best_epoch": best_epoch,
            "training_history": training_history
        }
        
    except Exception as e:
        print_progress(f" Failed to load checkpoint: {e}", verbose)
        return None

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
    """ generator architecture"""
    
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
    """discriminator architecture"""
    
    def __init__(self, embed_dim=128, hidden_dim=1024, dropout=0.3, use_skip=True):
        super().__init__()
        in_features = embed_dim * 3
        
        # First linear + batch‐norm + activation
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.drop1 = nn.Dropout(dropout)
        
        # Residual block: Linear → BN → activation, with skip
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.drop2 = nn.Dropout(dropout)
        
        # Project back up to input‐dim size for final scoring
        self.fc3 = nn.Linear(hidden_dim, in_features)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        self.drop3 = nn.Dropout(dropout)
        
        # Final output layer to a single scalar
        self.fc_out = nn.Linear(in_features, 1)
        
        # Optional skip from input directly to output
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
        # 1) flatten and concatenate
        x_input = torch.cat([head_emb, rel_emb, tail_emb], dim=-1).float()
        if x_input.dim() > 2:
            x_input = x_input.view(x_input.size(0), -1)
        
        # 2) first layer
        x = self.fc1(x_input)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        # 3) residual block
        res = self.fc2(x)
        res = self.bn2(res)
        res = self.act2(res)
        res = self.drop2(res)
        res = res + x
        
        # 4) project back to input‐dim 
        x = self.fc3(res)
        x = self.act3(x)
        x = self.drop3(x)
        
        out = self.fc_out(x).view(-1)
        
        if self.use_skip:
            skip = self.skip_linear(x_input).view(-1)
            out = out + skip
        
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

# HELPER FUNCTIONS
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
# MAIN TRAINING PIPELINE WITH ALL FUNCTIONS
# =============================================================================

def train_prot_b_gan_complete(args):
    """Complete training pipeline"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Prot-B-GAN Complete Training Pipeline")
    print(f"Device: {device}")
    print(f"Embedding method: {args.embedding_init}")
    print("=" * 70)
    
    try:
        # =========================================================================
        # DATA LOADING AND PREPROCESSING
        # =========================================================================
        
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
        # STAGE 1: EMBEDDING INITIALIZATION (3 MODES)
        # =========================================================================
        
        if args.load_embeddings:
            # MODE 1: Load pre-trained embeddings
            print_progress(f"\n MODE 1: Loading Pre-trained Embeddings", args.verbose)
            node_embeddings, rel_embeddings = load_pretrained_embeddings(args, num_entities, num_relations, device, args.verbose)
            
            if node_embeddings is None or rel_embeddings is None:
                print_progress(" Failed to load embeddings, falling back to training from scratch", args.verbose)
                args.load_embeddings = False  # Fallback
        
        if not args.load_embeddings:
            # MODE 2: Train embeddings from scratch
            print_progress(f"\n🏗 MODE 2: Training {args.embedding_init.upper()} Embeddings From Scratch", args.verbose)
            
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
        # STAGE 2: SMART WARM-UP (Skip if loading pre-trained)
        # =========================================================================
        
        if not args.load_embeddings and not args.resume_checkpoint:
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
        else:
            if args.load_embeddings:
                print_progress(f"\n STAGE 2: Warm-up skipped (using pre-trained embeddings)", args.verbose)
            else:
                print_progress(f"\n STAGE 2: Warm-up skipped (resuming from checkpoint)", args.verbose)
            
            # Quick baseline evaluation for reference
            baseline_hit_at_k = evaluate_hit_at_k(val_loader, None, node_emb, rel_emb, device, args.hit_at_k, score_function)
            print_progress(f"Current embedding Hit@10: {baseline_hit_at_k[10]:.4f}", args.verbose)
        
        # =========================================================================
        # STAGE 3: GAN-RL TRAINING WITH
        # =========================================================================
        
        print_progress(f"\n STAGE 3:  3-Tier Progressive GAN-RL Training", args.verbose)
        
        # Initialize models
        generator = Generator(args.embed_dim, args.noise_dim).to(device)
        discriminator = Discriminator(args.embed_dim, args.hidden_dim).to(device)
        
        g_opt = optim.Adam(list(generator.parameters()) + [node_emb], lr=args.g_lr)
        d_opt = optim.Adam(discriminator.parameters(), lr=args.d_lr)
        
        print_progress(f" Model Architecture:", args.verbose)
        print_progress(f"   Generator parameters: {sum(p.numel() for p in generator.parameters()):,}", args.verbose)
        print_progress(f"   Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}", args.verbose)
        
        # Training history
        training_history = {
            'losses': [], 'cos_sims': [], 'd_losses': [], 'g_losses': [],
            'train_hitks': {k: [] for k in args.hit_at_k},
            'val_hitks': {k: [] for k in args.hit_at_k},
            'f1_history': [], 'val_f1_history': [],
            'aupr_history': [], 'val_aupr_history': [],
            'mcc_history': [], 'val_mcc_history': [],
            'auc_history': [], 'val_auc_history': [],
            'real_acc_list': [], 'fake_acc_list': [],
            'collapse_hist': deque(maxlen=5), 'diversity_hist': deque(maxlen=5)
        }
        
        # Training state
        best_val_hit10 = 0.0
        best_epoch = 0
        start_epoch = 1
        current_tier = 1
        
        # TIER EPOCH COUNTERS
        tier2_epoch_count = 0
        tier3_epoch_count = 0
        
        # Pretraining variables
        best_pretrain_state = None
        pretrain_no_improve = 0
        prev_pretrain_hit10 = 0.0
        
        # Dynamic training schedule
        current_d_freq = args.d_update_freq
        current_g_steps = 1
        g_step_counter = 0
        
        bce_loss = nn.BCEWithLogitsLoss()
        
        # MODE 3: Resume from checkpoint if specified
        if args.resume_checkpoint:
            print_progress(f"\n MODE 3: Resuming from Checkpoint", args.verbose)
            checkpoint_data = load_checkpoint(args.resume_checkpoint, generator, discriminator, node_emb, rel_emb, g_opt, d_opt, device, args.force_resume, args.verbose)
            if checkpoint_data:
                start_epoch = checkpoint_data["resume_epoch"]
                best_val_hit10 = checkpoint_data["best_val_hit10"]
                best_epoch = checkpoint_data["best_epoch"]
                training_history = checkpoint_data["training_history"]
                current_tier = 2 if start_epoch > args.pretrain_epochs else 1
                print_progress(f" Resuming from epoch {start_epoch}, current tier: {current_tier}", args.verbose)
        
        print_progress(f"Starting  3-tier progressive training...", args.verbose)
        print_progress(f"  Tier 1 (Epochs 1-{args.pretrain_epochs}): Generator pretraining", args.verbose)
        print_progress(f"  Tier 2 (Epochs {args.rl_start_epoch}+): + RL system", args.verbose)
        print_progress(f"  Tier 3 (Epochs {args.full_system_epoch}+): + Full adversarial", args.verbose)
        
        # =========================================================================
        #  3-TIER TRAINING LOOP
        # =========================================================================
        
        for epoch in range(start_epoch, args.epochs + 1):
            
            # =================================================================
            # TIER 1:  PRETRAINING 
            # =================================================================
            
            if epoch <= args.pretrain_epochs:
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
                    
                    #reconstruction + cosine loss
                    rec_loss = F.mse_loss(fake, t_emb) + 0.1 * (1 - F.cosine_similarity(fake, t_emb).mean())
                    
                    # margin loss with multiple negatives
                    fake_exp = fake.unsqueeze(1).expand(-1, args.n_pre_neg, -1)
                    pos_cos = F.cosine_similarity(fake, t_emb, dim=-1)
                    neg_cos = F.cosine_similarity(fake_exp, neg_t_embs, dim=-1)
                    margin_term = 0.35 + neg_cos - pos_cos.unsqueeze(1)
                    margin_loss = F.relu(margin_term).mean()
                    
                    # pretraining loss composition
                    pretrain_loss = rec_loss + args.alpha_pretrain * margin_loss
                    pretrain_loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_norm)
                    g_opt.step()
                    
                    total_pretrain_loss += pretrain_loss.item()
                    total_pretrain_steps += 1
                
                # Validate pretraining progress 
                val_hitks = evaluate_hit_at_k(val_loader, generator, node_emb, rel_emb, device, args.hit_at_k, score_function)
                this_hit10 = val_hitks[10]
                
                if this_hit10 > prev_pretrain_hit10 + 1e-3:
                    prev_pretrain_hit10 = this_hit10
                    best_pretrain_state = {
                        "generator": generator.state_dict(),
                        "node_emb": node_emb.detach().cpu().clone()
                    }
                    pretrain_no_improve = 0
                    print_progress(f"[Tier 1] Epoch {epoch}: Hit@10 improved to {this_hit10:.4f}, saving snapshot", args.verbose)
                else:
                    pretrain_no_improve += 1
                    print_progress(f"[Tier 1] Epoch {epoch}: no improvement ({this_hit10:.4f} ≤ {prev_pretrain_hit10:.4f}), patience {pretrain_no_improve}/{args.pretrain_patience}", args.verbose)
                
                # Restore best pretrain state 
                if best_pretrain_state:
                    generator.load_state_dict(best_pretrain_state["generator"])
                    node_emb.data.copy_(best_pretrain_state["node_emb"].to(device))
                
                # Early stopping check
                if pretrain_no_improve >= args.pretrain_patience:
                    print_progress(f"[Tier 1] Early stopping at epoch {epoch}", args.verbose)
                    current_tier = 2
                    node_emb.requires_grad_(False)
                    g_opt = optim.Adam(generator.parameters(), lr=1e-4)
                    continue
                
                continue
            
            # Transition to Tier 2 after pretraining
            if epoch == args.pretrain_epochs + 1:
                current_tier = 2
                node_emb.requires_grad_(False)
                g_opt = optim.Adam(generator.parameters(), lr=1e-4)
                print_progress(" Transitioning to Tier 2: RL Training", args.verbose)
            
            # =================================================================
            # TIER 2 & 3: RL + ADVERSARIAL TRAINING
            # =================================================================
            
            # Update tier epoch counters
            if current_tier == 2:
                tier2_epoch_count += 1
            elif current_tier == 3:
                tier3_epoch_count += 1
            
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
                # DISCRIMINATOR TRAINING (Tier 3 only) 
                # =============================================================
                
                if current_tier == 3 and step % current_d_freq == 0 and g_step_counter >= current_g_steps:
                    d_opt.zero_grad()
                    
                    # Real samples
                    real_scores = discriminator(h_emb.detach(), r_emb_batch.detach(), t_emb.detach())
                    
                    # Generator fake samples
                    with torch.no_grad():
                        fake_samples = generator(h_emb.detach(), r_emb_batch.detach())
                    fake_scores = discriminator(h_emb.detach(), r_emb_batch.detach(), fake_samples)
                    
                    #  hard negative samples
                    neg_indices = generate_balanced_hard_negatives(
                        h_emb.detach(), r_emb_batch.detach(), node_emb, score_function,
                        num_hard=10, num_medium=8, num_easy=7
                    )
                    
                    batch_size = h_emb.shape[0]
                    selected_neg_idx = torch.randint(0, neg_indices.shape[1], (batch_size,), device=h_emb.device)
                    final_neg_idx = torch.gather(neg_indices, 1, selected_neg_idx.unsqueeze(1)).squeeze(1)
                    hard_neg_samples = node_emb[final_neg_idx]
                    hard_neg_scores = discriminator(h_emb.detach(), r_emb_batch.detach(), hard_neg_samples)
                    
                    # Soft labels 
                    real_labels = torch.full_like(real_scores, 0.8)     # Soft real
                    fake_labels = torch.full_like(fake_scores, 0.2)     # Soft fake
                    hard_neg_labels = torch.full_like(hard_neg_scores, 0.1)  # Very fake
                    
                    # Balanced loss computation
                    d_loss = (bce_loss(real_scores, real_labels) +
                              bce_loss(fake_scores, fake_labels) +
                              bce_loss(hard_neg_scores, hard_neg_labels)) / 3
                    
                    d_loss.backward()
                    d_opt.step()
                    total_d_loss += d_loss.item()
                    
                    # Reset generator counter
                    g_step_counter = 0
                    
                    # Track discriminator performance with hard negatives
                    with torch.no_grad():
                        all_real_scores = real_scores
                        all_fake_scores = torch.cat([fake_scores, hard_neg_scores])
                        
                        true_labels.extend([1]*len(all_real_scores) + [0]*len(all_fake_scores))
                        pred_probs.extend(torch.sigmoid(torch.cat([all_real_scores, all_fake_scores])).cpu().numpy())
                
                # =============================================================
                # GENERATOR TRAINING loss composition
                # =============================================================
                
                g_opt.zero_grad()
                fake = generator(h_emb, r_emb_batch)
                
                loss_components = []
                
                # Refinement loss 
                refinement_loss = F.mse_loss(fake, t_emb)
                loss_components.append(args.refinement_weight * refinement_loss)
                
                # DistMult component loss
                distmult_scores = score_function(h_emb, r_emb_batch, fake)
                distmult_loss = -torch.tanh(distmult_scores / 10.0).mean()
                loss_components.append(args.distmult_weight * distmult_loss)
                
                # RL loss 
                rl_loss, rl_metrics = compute_composite_rl_loss(
                    tier2_epoch_count, tier3_epoch_count, current_tier,
                    h_emb, r_emb_batch, fake, t_emb, discriminator,
                    args.rl_start_epoch, args.full_system_epoch, score_function
                )
                if rl_loss.item() != 0:
                    loss_components.append(args.rl_weight * rl_loss)
                
                # Bias mitigation 
                bias_loss = compute_bias_mitigation_loss(fake, t_emb, node_emb, h_emb, r_emb_batch)
                loss_components.append(args.bias_weight * bias_loss)
                
                # Adversarial loss (Tier 3 only)
                if current_tier == 3:
                    adv_scores = discriminator(h_emb, r_emb_batch, fake)
                    adv_loss = -torch.tanh(adv_scores / 5.0).mean()
                    loss_components.append(args.adv_weight * adv_loss)
                
                # Cosine margin loss with  hard negative approach
                batch_size = h_emb.shape[0]
                with torch.no_grad():
                    # Generate hard negatives for generator training
                    h_expand = h_emb.unsqueeze(1).expand(-1, node_emb.size(0), -1)
                    r_expand = r_emb_batch.unsqueeze(1).expand(-1, node_emb.size(0), -1)
                    all_nodes = node_emb.unsqueeze(0).expand(batch_size, -1, -1)
                    all_scores = score_function(h_expand, r_expand, all_nodes)
                    hard_vals, hard_idxs = all_scores.topk(args.hard_neg_k, dim=1)
                
                rand_k = args.n_neg - args.hard_neg_k
                rand_idxs = torch.randint(0, node_emb.size(0), (batch_size, rand_k), device=device)
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
                
                # Track generator steps for dynamic scheduling
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
            avg_d_loss = total_d_loss / max(1, effective_steps // current_d_freq) if current_tier == 3 else 0.0
            avg_g_loss = total_g_loss / effective_steps if effective_steps > 0 else 0.0
            
            # Update training history
            training_history['losses'].append(avg_loss)
            training_history['cos_sims'].append(avg_cos)
            training_history['d_losses'].append(avg_d_loss)
            training_history['g_losses'].append(avg_g_loss)
            
            # Compute enhanced discriminator metrics 
            enhanced_metrics = print_enhanced_discriminator_metrics(true_labels, pred_probs, epoch, args.verbose)
            training_history['f1_history'].append(enhanced_metrics['F1'])
            training_history['aupr_history'].append(enhanced_metrics['AUPR'])
            training_history['mcc_history'].append(enhanced_metrics['MCC'])
            training_history['auc_history'].append(enhanced_metrics['AUC'])
            
            # Rebalance training schedule based on discriminator performance
            if current_tier == 3:
                current_d_freq, current_g_steps = rebalance_training_schedule(epoch, enhanced_metrics['F1'])
            
            # Hit@K evaluation
            train_hit_at_k = evaluate_hit_at_k(train_loader, generator, node_emb, rel_emb, device, args.hit_at_k, score_function)
            for k in args.hit_at_k:
                training_history['train_hitks'][k].append(train_hit_at_k[k])
            
            # Validation
            val_metrics, val_cos_avg = validate(val_loader, generator, discriminator, node_emb, rel_emb, device)
            training_history['val_f1_history'].append(val_metrics['F1'])
            training_history['val_aupr_history'].append(val_metrics['AUPR'])
            training_history['val_mcc_history'].append(val_metrics['MCC'])
            training_history['val_auc_history'].append(val_metrics['AUC'])
            
            val_hit_at_k = evaluate_hit_at_k(val_loader, generator, node_emb, rel_emb, device, args.hit_at_k, score_function)
            for k in args.hit_at_k:
                training_history['val_hitks'][k].append(val_hit_at_k[k])
            
            current_hit10 = val_hit_at_k[10]
            
            # Diversity tracking 
            with torch.no_grad():
                fake_eval = generator(h_emb, r_emb_batch).detach()
                collapse_score = avg_pairwise_cosine_similarity(fake_eval)
                variance_score = compute_variance(fake_eval)
                diversity_score = compute_topk_diversity(fake_eval, node_emb)
            
            training_history['collapse_hist'].append(collapse_score)
            training_history['diversity_hist'].append(diversity_score)
            
            # Update best model
            if current_hit10 > best_val_hit10:
                best_val_hit10 = current_hit10
                best_epoch = epoch
                
                # Save checkpoint
                save_checkpoint(generator, discriminator, node_emb, rel_emb, g_opt, d_opt, epoch, 
                               training_history, args, best_val_hit10, best_epoch, enhanced_metrics)
            
            # Progress logging 
            tier_name = f"Tier {current_tier}"
            rl_info = ""
            if current_tier >= 2 and rl_metrics:
                if current_tier == 2:
                    rl_info = f" | RL_Method: {rl_metrics.get('method_component', 0):.3f}"
                else:
                    rl_info = f" | RL_Disc: {rl_metrics.get('disc_component', 0):.3f} RL_Method: {rl_metrics.get('method_component', 0):.3f}"
            
            schedule_info = f" | D_freq:{current_d_freq} G_steps:{current_g_steps}" if current_tier == 3 else ""
            
            print_progress(f"[{tier_name}] E{epoch:03d} | Loss {avg_loss:.4f} | CosSim {avg_cos:.3f} | "
                          f"F1 {enhanced_metrics['F1']:.4f}{rl_info}{schedule_info}", args.verbose)
            
            print_progress("Train: " + " ".join([f"Hit@{k}: {train_hit_at_k[k]:.4f}" for k in args.hit_at_k]), args.verbose)
            print_progress("Val:   " + " ".join([f"Hit@{k}: {val_hit_at_k[k]:.4f}" for k in args.hit_at_k]) + f" | F1 {val_metrics['F1']:.4f}", args.verbose)
            print_progress(f"Collapse: {collapse_score:.3f} | Diversity: {diversity_score:.3f} | Variance: {variance_score:.5f}", args.verbose)
            
            # Tier transition notifications 
            if epoch == args.rl_start_epoch:
                print_progress(f" TIER 2: RL system activated (method-specific scoring)", args.verbose)
                current_tier = 2
            elif epoch == args.full_system_epoch:
                print_progress(f" TIER 3: Full composite RL + Adversarial system activated", args.verbose)
                current_tier = 3
            
            # Visualization (every 5 epochs)
            if epoch % 5 == 0 or epoch == 1:
                with torch.no_grad():
                    fake_vis = generator(h_emb[:min(100, len(h_emb))], r_emb_batch[:min(100, len(r_emb_batch))]).detach()
                    real_vis = t_emb[:min(100, len(t_emb))].detach()
                plot_embedding_space(fake_vis, real_vis, epoch, output_dir=args.output_dir)
            
            # Early stopping
            if epoch - best_epoch > args.early_stopping_patience:
                print_progress(f"Early stopping at epoch {epoch}", args.verbose)
                break
        
        # =========================================================================
        # FINAL EVALUATION
        # =========================================================================
        
        print(f"\n TRAINING COMPLETED!")
        print(f"Best validation Hit@10: {best_val_hit10:.4f} achieved at epoch {best_epoch}")
        print(f"Final tier reached: {current_tier}")
        
        # Load best model for final evaluation
        checkpoint_path = os.path.join(args.output_dir, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            generator.load_state_dict(checkpoint['generator'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            node_emb.data.copy_(checkpoint['node_emb'].to(device))
            rel_emb.load_state_dict(checkpoint['rel_emb'])
        
        # Final test evaluation
        test_hit_at_k = evaluate_hit_at_k(test_loader, generator, node_emb, rel_emb, device, args.hit_at_k, score_function)
        final_val_metrics, _ = validate(val_loader, generator, discriminator, node_emb, rel_emb, device)
        
        print(f"\n FINAL TEST RESULTS:")
        for k in args.hit_at_k:
            print(f"Test Hit@{k}: {test_hit_at_k[k]:.4f} ({test_hit_at_k[k]*100:.1f}%)")
        
        print(f"\nDISCRIMINATOR HEALTH:")
        print(f"   F1 Score:    {final_val_metrics['F1']:.3f}")
        print(f"   AUPR:        {final_val_metrics['AUPR']:.3f}")
        print(f"   AUC:         {final_val_metrics['AUC']:.3f}")
        
        # Health assessment 
        if 0.60 <= final_val_metrics['F1'] <= 0.75:
            print(f"  Status: PROPERLY BALANCED! ")
        elif final_val_metrics['F1'] > 0.80:
            print(f" Status:  Still too strong (need more hard negatives)")
        else:
            print(f" Status:  Learning (better than broken F1=0.28)")
        
        # Save final results
        final_results = {
            "method": args.embedding_init,
            "baseline_hit_at_k": baseline_hit_at_k,
            "test_hit_at_k": test_hit_at_k,
            "best_val_hit10": best_val_hit10,
            "best_epoch": best_epoch,
            "final_discriminator_health": final_val_metrics,
            "training_history": training_history,
            "improvement": {k: test_hit_at_k[k] - baseline_hit_at_k[k] for k in args.hit_at_k},
            "training_summary": {
                "total_epochs_trained": epoch,
                "final_tier_reached": current_tier,
                "training_mode": "pre-trained" if args.load_embeddings else ("resumed" if args.resume_checkpoint else "from_scratch"),
                "start_epoch": start_epoch,
                "tier_transitions": [
                    ("Tier 1", f"Epochs 1-{args.pretrain_epochs}"),
                    ("Tier 2", f"Epochs {args.rl_start_epoch}+"),
                    ("Tier 3", f"Epochs {args.full_system_epoch}+")
                ]
            }
        }
        
        results_path = os.path.join(args.output_dir, f"final_results_{args.embedding_init}.pt")
        torch.save(final_results, results_path)
        
        print(f"\n All results saved to: {args.output_dir}")
        print(f"   - checkpoint.pt: Complete model checkpoint")
        print(f"   - final_results_{args.embedding_init}.pt: Final evaluation metrics")
        
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
    parser = argparse.ArgumentParser(description='Prot-B-GAN: Complete Implementation')
    
    # Data and I/O
    parser.add_argument('--data_root', type=str, required=True, help='Root directory containing data files')
    parser.add_argument('--train_file', type=str, default='FB15k-237-train-graph_triplets.csv', help='Training data file')
    parser.add_argument('--val_file', type=str, default='FB15k-237-val-graph_triplets.csv', help='Validation data file')
    parser.add_argument('--test_file', type=str, default='FB15k-237-test-graph_triplets.csv', help='Test data file')
    parser.add_argument('--output_dir', type=str, default='./exact_results', help='Output directory')
    
    # Pre-trained embeddings (MODE 1 - RECOMMENDED)
    parser.add_argument('--load_embeddings', action='store_true', help='Load pre-trained embeddings instead of training from scratch')
    parser.add_argument('--node_emb_path', type=str, help='Path to pre-trained node embeddings (.pt file)')
    parser.add_argument('--rel_emb_path', type=str, help='Path to pre-trained relation embeddings (.pt file)')
    parser.add_argument('--rel_map_path', type=str, help='Path to relation mapping (.pkl file)')
    
    # Checkpoint resuming (MODE 3)
    parser.add_argument('--resume_checkpoint', type=str, help='Path to checkpoint file to resume training from')
    parser.add_argument('--force_resume', action='store_true', help='Force resume even if epoch/config mismatch')
    
    # Model Architecture 
    parser.add_argument('--embed_dim', type=int, default=500, help='Embedding dimension')
    parser.add_argument('--noise_dim', type=int, default=64, help='Generator noise dimension')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Discriminator hidden dimension')
    
    # Training Parameters 
    parser.add_argument('--epochs', type=int, default=500, help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--g_lr', type=float, default=1e-3, help='Generator learning rate')
    parser.add_argument('--d_lr', type=float, default=5e-5, help='Discriminator learning rate')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Gradient clipping')
    
    # Embedding Initialization (MODE 2)
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
    parser.add_argument('--pretrain_epochs', type=int, default=200, help='Generator pretraining epochs')
    parser.add_argument('--pretrain_patience', type=int, default=20, help='Pretraining early stopping patience')
    parser.add_argument('--rl_start_epoch', type=int, default=20, help='RL training start epoch')
    parser.add_argument('--full_system_epoch', type=int, default=25, help='Full adversarial system start epoch')
    parser.add_argument('--d_update_freq', type=int, default=10, help='Discriminator update frequency')
    
    # Loss weights
    parser.add_argument('--rl_weight', type=float, default=0.1, help='RL loss weight')
    parser.add_argument('--adv_weight', type=float, default=0.05, help='Adversarial loss weight')
    parser.add_argument('--refinement_weight', type=float, default=0.7, help='Refinement loss weight')
    parser.add_argument('--distmult_weight', type=float, default=0.2, help='DistMult loss weight')
    parser.add_argument('--bias_weight', type=float, default=0.1, help='Bias mitigation weight')
    parser.add_argument('--g_guidance_weight', type=float, default=1.0, help='Generator guidance weight')
    parser.add_argument('--l2_reg_weight', type=float, default=0.1, help='L2 regularization weight')
    
    # Hard negative mining 
    parser.add_argument('--n_neg', type=int, default=50, help='Number of negative samples')
    parser.add_argument('--hard_neg_k', type=int, default=50, help='Number of hard negatives')
    parser.add_argument('--n_pre_neg', type=int, default=30, help='Number of pretraining negatives')
    parser.add_argument('--alpha_pretrain', type=float, default=1.5, help='Pretraining alpha weight')
    
    # Evaluation
    parser.add_argument('--hit_at_k', type=int, nargs='+', default=[1, 5, 10], help='Hit@K values')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Early stopping patience')
    
    # Debug and misc
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.debug:
        args.epochs = min(args.epochs, 40)
        args.pretrain_epochs = min(args.pretrain_epochs, 20)
        args.early_stopping_patience = 5
        print("DEBUG MODE: Reduced epochs and patience")
    
    # Verify data files
    data_files = [args.train_file, args.val_file, args.test_file]
    for file in data_files:
        if not os.path.exists(os.path.join(args.data_root, file)):
            print(f"Error: Data file not found: {os.path.join(args.data_root, file)}")
            return 1
    
    # Training mode summary
    if args.load_embeddings:
        mode = "MODE 1: Loading pre-trained embeddings"
    elif args.resume_checkpoint:
        mode = "MODE 3: Resuming from checkpoint"
    else:
        mode = "MODE 2: Training from scratch"
    
    print(f" {mode}")
    print(f"Embedding method: {args.embedding_init.upper()}")
    print(f"Smart warm-up: {'Enabled' if needs_warmup(args.embedding_init) else 'Skipped (not needed)'}")
    
    # Run training
    results = train_prot_b_gan_complete(args)
    
    if results:
        training_mode = results['training_summary']['training_mode'].upper()
        print(f"\n SUCCESS! {args.embedding_init.upper()} + GAN-RL completed ({training_mode})")
        print(f"Training: Epochs {results['training_summary']['start_epoch']} → {results['training_summary']['total_epochs_trained']}")
        print(f"Final tier: {results['training_summary']['final_tier_reached']}")
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
