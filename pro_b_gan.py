"""
Prot-B-GAN
================================================================

 tiered training system:
- Multiple embedding initialization methods
- Configurable reward scoring
- generator/discriminator for inference
- Full parameter configuration
- Simple vs detailed display options
- Complete saving system for reproducibility

Usage:
    python modular_prot_b_gan.py \
        --data_root "/path/to/data" \
        --embedding_init rgcn \
        --reward_scoring_method distmult \
        --embed_dim 500 --epochs 500 \
        --display_mode detailed --verbose
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
import json
import time
try:
    from torch_geometric.nn import RGCNConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: torch_geometric not available: {e}")
    print("R-GCN embedding initialization will not work.")
    print("Install with: pip install torch-geometric")
    TORCH_GEOMETRIC_AVAILABLE = False
    RGCNConv = None  # Placeholder
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score, matthews_corrcoef, roc_auc_score, precision_score, recall_score
from sklearn.decomposition import PCA
import numpy as np
from collections import deque, defaultdict
import random
from tqdm import tqdm
from abc import ABC, abstractmethod

# =============================================================================
# GENERATOR AND DISCRIMINATOR
# =============================================================================

class Generator(nn.Module):    
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
    
    def predict_tails(self, node_emb, rel_emb, head_ids, relation_ids, top_k=10):
        """Inference method for predicting top-k tails"""
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            head_ids = torch.tensor(head_ids, device=device) if not torch.is_tensor(head_ids) else head_ids.to(device)
            relation_ids = torch.tensor(relation_ids, device=device) if not torch.is_tensor(relation_ids) else relation_ids.to(device)
            
            h_emb = node_emb[head_ids]
            r_emb = rel_emb(relation_ids)
            
            # Generate predictions
            pred_emb = self.forward(h_emb, r_emb)
            
            # Find most similar nodes
            similarities = torch.matmul(F.normalize(pred_emb, dim=-1), 
                                      F.normalize(node_emb, dim=-1).T)
            
            top_k_scores, top_k_indices = similarities.topk(top_k, dim=1)
            
            return top_k_indices.cpu().numpy(), top_k_scores.cpu().numpy()

class Discriminator(nn.Module):
    
    def __init__(self, embed_dim=128, hidden_dim=1024, dropout=0.3, use_skip=True, final_activation=False):
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
        
        self.final_activation = final_activation
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
        
        if self.final_activation:
            out = torch.sigmoid(out)
        
        return out
    
    def score_triplets(self, node_emb, rel_emb, triplets):
        """Inference method for scoring triplets"""
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            if not torch.is_tensor(triplets):
                triplets = torch.tensor(triplets, device=device)
            else:
                triplets = triplets.to(device)
            
            h_emb = node_emb[triplets[:, 0]]
            r_emb = rel_emb(triplets[:, 1])
            t_emb = node_emb[triplets[:, 2]]
            
            scores = self.forward(h_emb, r_emb, t_emb)
            probabilities = torch.sigmoid(scores)
            
            return scores.cpu().numpy(), probabilities.cpu().numpy()

# =============================================================================
# EMBEDDING INITIALIZATION METHODS
# =============================================================================

class EmbeddingInitializer(ABC):
    @abstractmethod
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, verbose=True):
        pass
    
    @abstractmethod
    def get_name(self):
        pass

# =============================================================================
# EXACT RGCN + DISTMULT IMPLEMENTATION
# =============================================================================

class RGCNDistMult(nn.Module):
    """
    R-GCN + DistMult as described in Schlichtkrull et al. 2017
    "Modeling Relational Data with Graph Convolutional Networks"
    EXACT COPY FROM YOUR STANDALONE CODE
    """
    def __init__(self, num_entities, num_relations, hidden_dim=500, num_bases=None, num_layers=2):
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for R-GCN. Install with: pip install torch-geometric")

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim

        # Set num_bases as in paper (typically num_relations // 2 for FB15k-237)
        if num_bases is None:
            num_bases = min(num_relations, max(25, num_relations // 4))

        if TORCH_GEOMETRIC_AVAILABLE:
            print(f"R-GCN Configuration:")
            print(f"  Entities: {num_entities:,}")
            print(f"  Relations: {num_relations:,}")
            print(f"  Hidden dim: {hidden_dim}")
            print(f"  Num bases: {num_bases}")
            print(f"  Layers: {num_layers}")

        # Entity embeddings (input to R-GCN)
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim)

        # R-GCN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if RGCNConv is None:
                raise ImportError("RGCNConv not available. Install torch-geometric.")
            self.layers.append(
                RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases)
            )

        # DistMult relation embeddings (separate from R-GCN relations)
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)

        # Initialize as in paper
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

        self.dropout = nn.Dropout(0.2)

    def apply_edge_dropout(self, edge_index, edge_type, dropout_rate_self=0.2, dropout_rate_other=0.4):
        """
        Apply edge dropout as in paper: 0.2 for self-loops, 0.4 for others
        EXACT COPY FROM YOUR STANDALONE CODE
        """
        if not self.training:
            return edge_index, edge_type

        num_edges = edge_index.shape[1]
        mask = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)

        # Identify self-loops
        self_loops = edge_index[0] == edge_index[1]

        # Apply different dropout rates
        for i in range(num_edges):
            if self_loops[i]:
                if random.random() < dropout_rate_self:
                    mask[i] = False
            else:
                if random.random() < dropout_rate_other:
                    mask[i] = False

        return edge_index[:, mask], edge_type[mask]

    def forward_entities(self, edge_index, edge_type):
        """
        Forward pass through R-GCN to get entity representations
        EXACT COPY FROM YOUR STANDALONE CODE
        """
        # Apply edge dropout before message passing
        edge_index_dropped, edge_type_dropped = self.apply_edge_dropout(edge_index, edge_type)

        # Get initial entity embeddings
        x = self.entity_embedding.weight  # [num_entities, hidden_dim]

        # Apply R-GCN layers
        for layer in self.layers:
            x = layer(x, edge_index_dropped, edge_type_dropped)
            x = F.relu(x)
            x = self.dropout(x)

        return x

    def distmult_score(self, head_emb, rel_emb, tail_emb):
        """
        DistMult scoring function: <h, r, t> = Σ(h ⊙ r ⊙ t)
        EXACT COPY FROM YOUR STANDALONE CODE
        """
        return torch.sum(head_emb * rel_emb * tail_emb, dim=-1)

class RGCNTrainer:
    """
    Trainer following the exact paper methodology
    EXACT COPY FROM YOUR STANDALONE CODE
    """

    def __init__(self, model, device='cuda', lr=0.01, l2_penalty=0.01):
        self.model = model.to(device)
        self.device = device
        self.l2_penalty = l2_penalty

        # Optimizer as in paper
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def negative_sampling_paper(self, positive_triples, num_entities):
        """
        Paper negative sampling: ω = 1 (one negative per positive)
        Corrupt either head or tail randomly
        EXACT COPY FROM YOUR STANDALONE CODE
        """
        batch_size = positive_triples.shape[0]
        negative_triples = positive_triples.clone()

        # Random corruption (head or tail)
        for i in range(batch_size):
            if random.random() < 0.5:
                # Corrupt head
                negative_triples[i, 0] = random.randint(0, num_entities - 1)
            else:
                # Corrupt tail
                negative_triples[i, 2] = random.randint(0, num_entities - 1)

        return negative_triples

    def train_epoch_fullbatch(self, edge_index, edge_type, train_triples):
        """
        Full-batch training as in paper (not mini-batch!)
        EXACT COPY FROM YOUR STANDALONE CODE
        """
        self.model.train()

        # Get entity embeddings from R-GCN
        entity_embeddings = self.model.forward_entities(edge_index, edge_type)

        # Positive triples
        pos_heads = entity_embeddings[train_triples[:, 0]]
        pos_rels = self.model.relation_embedding(train_triples[:, 1])
        pos_tails = entity_embeddings[train_triples[:, 2]]
        positive_scores = self.model.distmult_score(pos_heads, pos_rels, pos_tails)

        # Generate negatives (ω = 1)
        negative_triples = self.negative_sampling_paper(train_triples, self.model.num_entities)

        neg_heads = entity_embeddings[negative_triples[:, 0]]
        neg_rels = self.model.relation_embedding(negative_triples[:, 1])
        neg_tails = entity_embeddings[negative_triples[:, 2]]
        negative_scores = self.model.distmult_score(neg_heads, neg_rels, neg_tails)

        # Margin ranking loss (paper default)
        margin = 1.0
        ranking_loss = F.relu(margin + negative_scores - positive_scores).mean()

        # L2 regularization on decoder (relation embeddings) - paper: 0.01
        l2_loss = self.l2_penalty * self.model.relation_embedding.weight.norm(2).pow(2)

        total_loss = ranking_loss + l2_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping (standard practice)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return total_loss.item(), ranking_loss.item(), l2_loss.item()

    def evaluate_hit_at_k_simple(self, edge_index, edge_type, eval_triples, k_values=[1, 5, 10]):
        """
        Simplified Hit@K evaluation for embedding initialization (faster than full filtered eval)
        """
        self.model.eval()
        hits = {k: 0 for k in k_values}
        total = 0

        with torch.no_grad():
            # Get all entity embeddings
            entity_embeddings = self.model.forward_entities(edge_index, edge_type)

            # Sample a subset for faster evaluation during initialization
            max_eval_samples = min(1000, len(eval_triples))
            eval_indices = torch.randperm(len(eval_triples))[:max_eval_samples]
            sample_triples = eval_triples[eval_indices]

            for triple in sample_triples:
                head, rel, tail = triple.cpu().numpy()

                # Get embeddings
                head_emb = entity_embeddings[head:head+1]
                rel_emb = self.model.relation_embedding(torch.tensor([rel], device=self.device))

                # Score against all possible tails
                all_scores = self.model.distmult_score(
                    head_emb.expand(self.model.num_entities, -1),
                    rel_emb.expand(self.model.num_entities, -1),
                    entity_embeddings
                )

                # Calculate rank (no filtering for speed)
                target_score = all_scores[tail].item()
                rank = (all_scores > target_score).sum().item() + 1

                # Update metrics
                for k in k_values:
                    if rank <= k:
                        hits[k] += 1

                total += 1

        # Calculate final metrics
        hit_ratios = {k: hits[k] / total for k in k_values}
        self.model.train()
        return hit_ratios

class RGCNInitializer(EmbeddingInitializer):
    def __init__(self, embed_dim=128, epochs=100, lr=0.01, layers=2, l2_penalty=0.01, early_stopping_patience=50):
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.lr = lr
        self.layers = layers
        self.l2_penalty = l2_penalty
        self.early_stopping_patience = early_stopping_patience
    
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, verbose=True):
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for R-GCN initialization. Install with: pip install torch-geometric")
        
        if verbose:
            print(f"Training R-GCN + DistMult embeddings (YOUR EXACT METHODOLOGY)")
            print(f"  Epochs: {self.epochs} (with early stopping patience: {self.early_stopping_patience})")
            print(f"  Training method: Full-batch (paper-exact)")
            print(f"  Edge dropout: 0.2 self-loops, 0.4 others")
            print(f"  L2 penalty: {self.l2_penalty} on decoder")
            print(f"  Learning rate: {self.lr}")
        
        # Build graph from all data (YOUR EXACT METHOD)
        all_data = pd.concat([train_df, val_df, test_df])
        all_triples = torch.tensor(all_data.values, dtype=torch.long)
        edge_index = all_triples[:, [0, 2]].t().contiguous().to(device)
        edge_type = all_triples[:, 1].to(device)
        
        # Initialize YOUR EXACT R-GCN model
        model = RGCNDistMult(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=self.embed_dim,
            num_bases=None, 
            num_layers=self.layers
        )

        trainer = RGCNTrainer(
            model,
            device=device,
            lr=self.lr,
            l2_penalty=self.l2_penalty
        )

        # Convert data to tensors
        train_triples = torch.tensor(train_df.values, dtype=torch.long).to(device)
        val_triples = torch.tensor(val_df.values, dtype=torch.long).to(device)

        # Training with YOUR EXACT methodology + early stopping
        best_val_hit10 = 0.0
        patience_counter = 0
        training_history = []

        if verbose:
            print(f"\nStarting R-GCN training (YOUR EXACT METHOD)...")
            print("="*60)

        for epoch in range(self.epochs):
            # YOUR EXACT training method
            epoch_start = time.time()
            total_loss, ranking_loss, l2_loss = trainer.train_epoch_fullbatch(
                edge_index, edge_type, train_triples
            )
            epoch_time = time.time() - epoch_start

            # Save training info
            training_history.append({
                'epoch': epoch,
                'total_loss': total_loss,
                'ranking_loss': ranking_loss,
                'l2_loss': l2_loss,
                'epoch_time': epoch_time
            })

            # Evaluation for early stopping (every 10 epochs or at key points)
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                if verbose:
                    print(f"Epoch {epoch}: Loss = {total_loss:.4f} (Ranking: {ranking_loss:.4f}, L2: {l2_loss:.6f}) - {epoch_time:.1f}s")

                # Quick evaluation for early stopping
                val_hits = trainer.evaluate_hit_at_k_simple(edge_index, edge_type, val_triples)
                
                if verbose:
                    print(f"  Validation: Hit@1: {val_hits[1]:.3f}, Hit@5: {val_hits[5]:.3f}, Hit@10: {val_hits[10]:.3f}")

                # Check for improvement
                if val_hits[10] > best_val_hit10:
                    best_val_hit10 = val_hits[10]
                    patience_counter = 0
                    
                    # Save best embeddings
                    with torch.no_grad():
                        best_node_embeddings = model.forward_entities(edge_index, edge_type).detach().clone()
                        best_rel_embeddings = model.relation_embedding.weight.detach().clone()
                    
                    if verbose:
                        print(f"  ✅ New best: Hit@10 {best_val_hit10:.3f}")
                else:
                    patience_counter += 1
                    if verbose:
                        print(f"  No improvement. Patience: {patience_counter}/{self.early_stopping_patience}")

                if patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f"\n Early stopping triggered after {self.early_stopping_patience} epochs without improvement")
                        print(f"Best validation Hit@10: {best_val_hit10:.3f}")
                    break

            elif epoch % 5 == 0:
                if verbose:
                    print(f"Epoch {epoch}: Loss = {total_loss:.4f} - {epoch_time:.1f}s")

        # Use best embeddings or final if no early stopping occurred
        try:
            final_node_embeddings = best_node_embeddings
            final_rel_embeddings = best_rel_embeddings
        except:
            with torch.no_grad():
                final_node_embeddings = model.forward_entities(edge_index, edge_type)
                final_rel_embeddings = model.relation_embedding.weight

        if verbose:
            print(f"R-GCN + DistMult training complete (YOUR EXACT METHOD)")
            print(f"Final embeddings: {final_node_embeddings.shape} nodes, {final_rel_embeddings.shape} relations")

        return final_node_embeddings, final_rel_embeddings
    
    def get_name(self):
        return "R-GCN + DistMult (Paper-Exact)"

class RandomInitializer(EmbeddingInitializer):
    def __init__(self, embed_dim=128):
        self.embed_dim = embed_dim
    
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, verbose=True):
        if verbose:
            print(f"Initializing random embeddings (dim={self.embed_dim})...")
        
        node_embeddings = torch.randn(num_entities, self.embed_dim, device=device)
        relation_embeddings = torch.randn(num_relations, self.embed_dim, device=device)
        nn.init.xavier_uniform_(node_embeddings)
        nn.init.xavier_uniform_(relation_embeddings)
        
        if verbose:
            print("Random initialization complete")
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
        if verbose:
            print(f"Training TransE embeddings ({self.epochs} epochs)...")
        
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
                print(f"  TransE Epoch {epoch}: Loss = {loss.item():.4f}")
        
        if verbose:
            print("TransE training complete")
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
        if verbose:
            print(f"Training DistMult embeddings ({self.epochs} epochs)...")
        
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
                print(f"  DistMult Epoch {epoch}: Loss = {total_loss.item():.4f}")
        
        if verbose:
            print("DistMult training complete")
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
        if verbose:
            print(f"Training ComplEx embeddings ({self.epochs} epochs)...")
        
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
                print(f"  ComplEx Epoch {epoch}: Loss = {total_loss.item():.4f}")
        
        # Combine real and imaginary parts for final embeddings
        final_entity_emb = torch.cat([entity_real.weight, entity_imag.weight], dim=1)
        final_relation_emb = torch.cat([relation_real.weight, relation_imag.weight], dim=1)
        
        if verbose:
            print("ComplEx training complete")
        return final_entity_emb.detach(), final_relation_emb.detach()
    
    def get_name(self):
        return "ComplEx"

class PreloadedInitializer(EmbeddingInitializer):
    """Load pre-saved embeddings"""
    
    def __init__(self, embed_dim=128, data_root=None):
        self.embed_dim = embed_dim
        self.data_root = data_root
    
    def initialize_embeddings(self, train_df, val_df, test_df, num_entities, num_relations, device, verbose=True):
        if verbose:
            print(f"Loading pre-saved embeddings from {self.data_root}...")
        
        # Auto-detect dataset name
        dataset_name = os.path.basename(self.data_root.rstrip('/'))
        if 'FB15k' in dataset_name or 'fb15k' in dataset_name:
            dataset_prefix = "FB15k-237"
        elif 'prothgt' in dataset_name.lower():
            dataset_prefix = "prothgt"
        else:
            dataset_prefix = dataset_name
        
        node_emb_path = os.path.join(self.data_root, f"{dataset_prefix}_final_node_embeddings.pt")
        rel_emb_path = os.path.join(self.data_root, f"{dataset_prefix}_final_distmult_rel_emb.pt")
        
        if not os.path.exists(node_emb_path):
            raise FileNotFoundError(f"Node embeddings not found: {node_emb_path}")
        if not os.path.exists(rel_emb_path):
            raise FileNotFoundError(f"Relation embeddings not found: {rel_emb_path}")
        
        # Load embeddings
        node_embeddings = torch.load(node_emb_path, map_location=device)
        
        # Load relation embeddings (they're saved as state_dict)
        rel_state_dict = torch.load(rel_emb_path, map_location=device)
        relation_embeddings = rel_state_dict['weight']
        
        if verbose:
            print(f"Loaded embeddings: {node_embeddings.shape} nodes, {relation_embeddings.shape} relations")
        
        return node_embeddings, relation_embeddings
    
    def get_name(self):
        return "Preloaded"

def create_embedding_initializer(args):
    """Factory function to create embedding initializer"""
    method = args.embedding_init.lower()
    if method == 'rgcn':
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for R-GCN initialization. Install with: pip install torch-geometric")
        return RGCNInitializer(args.embed_dim, args.rgcn_epochs, args.rgcn_lr, args.rgcn_layers, args.rgcn_l2_penalty, args.rgcn_early_stopping_patience)
    elif method == 'preloaded':
        return PreloadedInitializer(args.embed_dim, args.data_root)
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
# REWARD SCORING FUNCTIONS
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
    
    reward_functions = {
        'transe': transe_reward_function,
        'distmult': distmult_reward_function,
        'complex': complex_reward_function,
        'auto': distmult_reward_function  # Default
    }
    
    return reward_functions.get(reward_method.lower(), distmult_reward_function)

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
# UTILITY FUNCTIONS (UPDATED WITH YOUR NON-MODULAR METRICS)
# =============================================================================

def print_progress(message, verbose=True, display_mode='detailed'):
    if verbose:
        if display_mode == 'simple':
            # Simple one-line updates
            print(f"\r{message}", end='', flush=True)
        else:
            # Detailed multi-line output
            print(message)

def compute_metrics(y_true, y_probs, threshold=0.5):
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

def evaluate_hit_at_k(data_loader, generator, node_emb, rel_emb, device, hit_at_k_list):
    if generator:
        generator.eval()
    
    hits_dict = {k: 0 for k in hit_at_k_list}
    total_examples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            h, r, t = [b.to(device) for b in batch.T]
            h_emb, r_emb_batch = node_emb[h], rel_emb(r)
            
            if generator:
                pred = generator(h_emb, r_emb_batch)
                norm_pred = F.normalize(pred, dim=-1)
                norm_nodes = F.normalize(node_emb, dim=-1)
                sims = torch.matmul(norm_pred, norm_nodes.T)
            else:
                # Direct evaluation without generator
                sims = torch.matmul(F.normalize(h_emb, dim=-1), F.normalize(node_emb, dim=-1).T)
            
            for k in hit_at_k_list:
                topk = sims.topk(k, dim=1).indices
                for i in range(len(t)):
                    if t[i].item() in topk[i]:
                        hits_dict[k] += 1
            total_examples += len(t)
    
    if generator:
        generator.train()
    
    return {k: hits_dict[k] / total_examples for k in hit_at_k_list} if total_examples > 0 else {k: 0.0 for k in hit_at_k_list}

def validate(val_loader, generator, discriminator, node_emb, rel_emb, device):
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

def generate_balanced_hard_negatives(h_emb, r_emb, node_emb, num_hard=10, num_medium=8, num_easy=7):
    batch_size = h_emb.shape[0]
    
    with torch.no_grad():
        h_expand = h_emb.unsqueeze(1).expand(-1, node_emb.size(0), -1)
        r_expand = r_emb.unsqueeze(1).expand(-1, node_emb.size(0), -1)
        all_nodes = node_emb.unsqueeze(0).expand(batch_size, -1, -1)
        dm_scores = (h_expand * r_expand * all_nodes).sum(dim=-1)
        
        hard_vals, hard_idxs = dm_scores.topk(num_hard, dim=1)
        
        sorted_scores, sorted_idxs = dm_scores.sort(dim=1, descending=True)
        start_idx = node_emb.size(0) // 3
        end_idx = start_idx + num_medium
        medium_idxs = sorted_idxs[:, start_idx:end_idx]
        
        easy_idxs = torch.randint(0, node_emb.size(0), (batch_size, num_easy), device=h_emb.device)
        all_neg_idxs = torch.cat([hard_idxs, medium_idxs, easy_idxs], dim=1)
    
    return all_neg_idxs

# UPDATED METRICS FUNCTIONS FROM YOUR NON-MODULAR CODE
def avg_pairwise_cosine_similarity(tensors):
    norm = F.normalize(tensors, dim=-1)
    sim_matrix = torch.matmul(norm, norm.T)
    upper_tri = sim_matrix.triu(1)
    return upper_tri[upper_tri != 0].mean().item()

def compute_variance(tensors):
    return torch.var(tensors, dim=0).mean().item()

def compute_topk_diversity(fake_embeds, all_node_embeds, k=10):
    sims = torch.matmul(F.normalize(fake_embeds, dim=-1),
                        F.normalize(all_node_embeds, dim=-1).T)
    topk = sims.topk(k, dim=1).indices.view(-1).cpu().numpy()
    unique = len(set(topk))
    return unique / len(topk)

def print_enhanced_discriminator_metrics(true_labels, pred_probs, epoch, display_mode='detailed', detailed_metrics=True):
    """Enhanced discriminator metrics with health assessment (from your non-modular code)"""
    if not true_labels:
        return {"F1": 0, "AUPR": 0, "MCC": 0, "AUC": 0, "Precision": 0, "Recall": 0, "Gap": 0}
    
    y_true = np.array(true_labels)
    y_probs = np.array(pred_probs)
    
    # Calculate discriminator gap
    real_probs = [p for i, p in enumerate(y_probs) if y_true[i] == 1]
    fake_probs = [p for i, p in enumerate(y_probs) if y_true[i] == 0]
    gap = np.mean(real_probs) - np.mean(fake_probs) if real_probs and fake_probs else 0
    
    metrics = compute_metrics(y_true, y_probs)
    
    # Additional metrics
    y_pred = (y_probs >= 0.5).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    enhanced_metrics = {
        **metrics,
        "Precision": precision,
        "Recall": recall,
        "Gap": gap
    }
    
    # Health assessment (only if detailed metrics enabled)
    if detailed_metrics and display_mode == 'detailed':
        if gap > 0.05 and metrics['F1'] > 0.60:
            status = "✅ HEALTHY (like your original F1=0.67!)"
        elif gap > 0.03 and metrics['F1'] > 0.50:
            status = "⚠️ WEAK but learning"
        elif gap < 0.01:
            status = "❌ BROKEN (like before the fix)"
        else:
            status = "🔄 LEARNING"
        
        print(f" Discriminator Health: Gap={gap:.3f} F1={metrics['F1']:.3f} Precision={precision:.3f} Recall={recall:.3f} AUC={metrics['AUC']:.3f}")
        print(f"   Status: {status}")
    
    return enhanced_metrics

# =============================================================================
#  COMPOSITE RL LOSS FUNCTION (FROM YOUR NON-MODULAR CODE)
# =============================================================================

def compute_composite_rl_loss(epoch, h_emb, r_emb, fake, t_emb, discriminator, args, 
                                    true_labels=None, pred_probs=None):
    """
    Your exact composite RL loss with discriminator health monitoring and gradual transition
    """
    
    device = h_emb.device
    
    if epoch < args.rl_start_epoch:
        return torch.tensor(0.0, device=device), {}
    
    elif epoch < args.full_system_epoch:
        # TIER 2: DistMult-only RL with ultra-aggressive temperature scaling
        dm_scores = (h_emb * r_emb * fake).sum(dim=-1)
        # Ultra-aggressive temperature scaling to prevent saturation
        dm_component = torch.sigmoid(dm_scores / 100.0)  # Was /20.0, now /100.0 for maximum responsiveness
        simple_reward = dm_component - 0.5
        rl_loss = -simple_reward.mean()
        
        return rl_loss, {
            'dm_component': dm_component.mean().item(),
            'tier': 2,
            'status': f'Tier 2 - DistMult only (Epoch {epoch}/{args.full_system_epoch})'
        }
    
    else:
        # TIER 3: Full composite RL with discriminator health monitoring
        dm_scores = (h_emb * r_emb * fake).sum(dim=-1)
        dm_component = torch.sigmoid(dm_scores / 100.0)  # Keep ultra-aggressive scaling
        
        with torch.no_grad():
            disc_scores = discriminator(h_emb.detach(), r_emb.detach(), fake.detach())
            disc_component = torch.sigmoid(disc_scores)
        
        # Discriminator health assessment
        if true_labels is not None and pred_probs is not None and len(true_labels) > 0:
            current_disc_f1 = compute_metrics(np.array(true_labels), np.array(pred_probs))['F1']
        else:
            current_disc_f1 = 0.3  # Conservative fallback
        
        # Gradual transition based on discriminator health
        if current_disc_f1 < 0.3:  # Discriminator too weak
            disc_weight = 0.1
            dm_weight = 0.9
            protection_status = f"WEAK (F1={current_disc_f1:.3f}), using 0.1/0.9"
        elif current_disc_f1 < 0.5:  # Discriminator learning
            disc_weight = 0.3
            dm_weight = 0.7
            protection_status = f"LEARNING (F1={current_disc_f1:.3f}), using 0.3/0.7"
        else:  # Discriminator healthy
            disc_weight = 0.5
            dm_weight = 0.5
            protection_status = f"HEALTHY (F1={current_disc_f1:.3f}), using 0.5/0.5"
        
        composite_reward = disc_weight * disc_component + dm_weight * dm_component - 0.5
        rl_loss = -composite_reward.mean()
        
        return rl_loss, {
            'disc_component': disc_component.mean().item(),
            'dm_component': dm_component.mean().item(),
            'composite_reward': composite_reward.mean().item(),
            'disc_weight': disc_weight,
            'dm_weight': dm_weight,
            'disc_f1': current_disc_f1,
            'tier': 3,
            'protection_status': protection_status
        }

def bias_mitigation_loss(fake, t_emb, node_emb, h_emb, r_emb):
    """Your exact bias mitigation approach"""
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

# =============================================================================
# COMPREHENSIVE SAVING FUNCTIONS
# =============================================================================

def save_initial_embeddings_and_mappings(args, train_df, val_df, test_df, node_embeddings, rel_embeddings, num_entities, num_relations, verbose=True):
    """Save initial embeddings and all necessary mappings for reusability"""
    
    # Auto-detect dataset prefix
    dataset_name = os.path.basename(args.data_root.rstrip('/'))
    if 'FB15k' in dataset_name or 'fb15k' in dataset_name:
        dataset_prefix = "FB15k-237"
    elif 'prothgt' in dataset_name.lower():
        dataset_prefix = "prothgt"
    else:
        dataset_prefix = dataset_name
    
    print_progress(f"Saving initial embeddings and mappings for reuse...", verbose, args.display_mode)
    
    # 1. Save node embeddings
    node_emb_path = os.path.join(args.data_root, f"{dataset_prefix}_final_node_embeddings.pt")
    torch.save(node_embeddings.cpu(), node_emb_path)
    print_progress(f" Node embeddings saved: {node_emb_path}", verbose, args.display_mode)
    
    # 2. Save relation embeddings (as state_dict for consistency)
    rel_emb_path = os.path.join(args.data_root, f"{dataset_prefix}_final_distmult_rel_emb.pt")
    rel_state_dict = {'weight': rel_embeddings.cpu()}
    torch.save(rel_state_dict, rel_emb_path)
    print_progress(f" Relation embeddings saved: {rel_emb_path}", verbose, args.display_mode)
    
    # 3. Create and save entity mapping
    entity_map_path = os.path.join(args.data_root, f"{dataset_prefix}_entity_map.pkl")
    if not os.path.exists(entity_map_path):
        all_data = pd.concat([train_df, val_df, test_df])
        all_entities = sorted(set(all_data['H'].unique()) | set(all_data['T'].unique()))
        entity_map = {entity_id: idx for idx, entity_id in enumerate(all_entities)}
        
        with open(entity_map_path, 'wb') as f:
            pickle.dump(entity_map, f)
        print_progress(f" Entity mapping saved: {entity_map_path}", verbose, args.display_mode)
    else:
        print_progress(f" Entity mapping already exists: {entity_map_path}", verbose, args.display_mode)
    
    # 4. Create and save relation mapping
    relation_map_path = os.path.join(args.data_root, f"{dataset_prefix}_relation_map.pkl")
    if not os.path.exists(relation_map_path):
        all_data = pd.concat([train_df, val_df, test_df])
        unique_relations = sorted(all_data['R'].unique())
        relation_map = {rel_id: idx for idx, rel_id in enumerate(unique_relations)}
        
        with open(relation_map_path, 'wb') as f:
            pickle.dump(relation_map, f)
        print_progress(f" Relation mapping saved: {relation_map_path}", verbose, args.display_mode)
    else:
        print_progress(f" Relation mapping already exists: {relation_map_path}", verbose, args.display_mode)
    
    # 5. Save reverse mappings for inference
    id_to_entity_path = os.path.join(args.data_root, f"{dataset_prefix}_id_to_entity_map.pkl")
    id_to_relation_path = os.path.join(args.data_root, f"{dataset_prefix}_id_to_relation_map.pkl")
    
    if not os.path.exists(id_to_entity_path):
        with open(entity_map_path, 'rb') as f:
            entity_map = pickle.load(f)
        id_to_entity = {idx: entity_id for entity_id, idx in entity_map.items()}
        with open(id_to_entity_path, 'wb') as f:
            pickle.dump(id_to_entity, f)
        print_progress(f" Reverse entity mapping saved: {id_to_entity_path}", verbose, args.display_mode)
    
    if not os.path.exists(id_to_relation_path):
        with open(relation_map_path, 'rb') as f:
            relation_map = pickle.load(f)
        id_to_relation = {idx: rel_id for rel_id, idx in relation_map.items()}
        with open(id_to_relation_path, 'wb') as f:
            pickle.dump(id_to_relation, f)
        print_progress(f" Reverse relation mapping saved: {id_to_relation_path}", verbose, args.display_mode)
    
    # 6. Save training configuration
    config_path = os.path.join(args.data_root, f"{dataset_prefix}_training_config.json")
    config_data = {
        'args': vars(args),
        'dataset_info': {
            'num_entities': num_entities,
            'num_relations': num_relations,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df)
        },
        'embedding_method': args.embedding_init,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'torch_version': torch.__version__,
        'device': str(torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu')
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    print_progress(f" Training config saved: {config_path}", verbose, args.display_mode)
    
    # 7. Save dataset statistics
    stats_path = os.path.join(args.data_root, f"{dataset_prefix}_dataset_stats.json")
    all_data = pd.concat([train_df, val_df, test_df])
    stats = {
        'entity_count': num_entities,
        'relation_count': num_relations,
        'split_sizes': {
            'train': len(train_df),
            'val': len(val_df), 
            'test': len(test_df)
        },
        'relation_frequency': dict(all_data['R'].value_counts().head(20)),
        'entity_frequency_head': dict(all_data['H'].value_counts().head(20)),
        'entity_frequency_tail': dict(all_data['T'].value_counts().head(20)),
        'avg_degree': {
            'in_degree': float(all_data['T'].value_counts().mean()),
            'out_degree': float(all_data['H'].value_counts().mean())
        }
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print_progress(f" Dataset statistics saved: {stats_path}", verbose, args.display_mode)
    
    print_progress(f" Initial embeddings and mappings saved successfully!", verbose, args.display_mode)
    print_progress(f"   Files can be reused with: --embedding_init preloaded", verbose, args.display_mode)
    
    return dataset_prefix

def save_phase_checkpoint(phase_name, epoch, generator, discriminator, node_emb, rel_emb, additional_data, args, dataset_prefix, verbose=True):
    """Save checkpoint for specific training phase"""
    
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f"{dataset_prefix}_{phase_name}_checkpoint.pt")
    
    checkpoint_data = {
        "phase": phase_name,
        "epoch": epoch,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        **additional_data
    }
    
    # Add model states if available
    if generator is not None:
        checkpoint_data["generator"] = generator.state_dict()
    if discriminator is not None:
        checkpoint_data["discriminator"] = discriminator.state_dict()
    if node_emb is not None:
        checkpoint_data["node_emb"] = node_emb.detach().cpu()
    if rel_emb is not None:
        checkpoint_data["rel_emb"] = rel_emb.state_dict()
    
    torch.save(checkpoint_data, checkpoint_path)
    print_progress(f"  {phase_name.title()} checkpoint saved: {checkpoint_path}", verbose, args.display_mode)

# =============================================================================
# DATA LOADING
# =============================================================================

class TripletDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples
    def __len__(self):
        return len(self.triples)
    def __getitem__(self, idx):
        return torch.tensor(self.triples[idx], dtype=torch.long)

def load_data(args):
    """Load and prepare data"""
    print_progress(f"Loading data from {args.data_root}...", args.verbose, args.display_mode)
    
    train_df = pd.read_csv(os.path.join(args.data_root, args.train_file))
    val_df = pd.read_csv(os.path.join(args.data_root, args.val_file))
    test_df = pd.read_csv(os.path.join(args.data_root, args.test_file))
    
    # Find column names flexibly
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
    
    if args.debug:
        train_df = train_df.sample(min(len(train_df), args.max_train_samples), random_state=42)
        val_df = val_df.sample(min(len(val_df), args.max_val_samples), random_state=42)
        test_df = test_df.sample(min(len(test_df), args.max_test_samples), random_state=42)
        print_progress(f"DEBUG: Using {len(train_df)} train, {len(val_df)} val, {len(test_df)} test", args.verbose, args.display_mode)
    
    # Create entity and relation mappings
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
    
    print_progress(f"Data loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test", args.verbose, args.display_mode)
    print_progress(f"Entities: {num_entities:,}, Relations: {num_relations:,}", args.verbose, args.display_mode)
    
    return train_df, val_df, test_df, num_entities, num_relations

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_modular_prot_b_gan(args):
    """Main training function with tiered system and complete saving system"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_progress(f"Modular Prot-B-GAN Training", args.verbose, args.display_mode)
    print_progress(f"Device: {device}", args.verbose, args.display_mode)
    print_progress(f"Embedding: {args.embedding_init}, Reward: {args.reward_scoring_method}", args.verbose, args.display_mode)
    print_progress("=" * 70, args.verbose, args.display_mode)
    
    try:
        # Load data
        train_df, val_df, test_df, num_entities, num_relations = load_data(args)
        
        # Create data loaders
        train_triples = list(zip(train_df["H"], train_df["R"], train_df["T"]))
        val_triples = list(zip(val_df["H"], val_df["R"], val_df["T"]))
        test_triples = list(zip(test_df["H"], test_df["R"], test_df["T"]))
        
        train_loader = DataLoader(TripletDataset(train_triples), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TripletDataset(val_triples), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(TripletDataset(test_triples), batch_size=args.batch_size, shuffle=False)
        
        # Initialize embeddings
        print_progress(f"STAGE 1: {args.embedding_init.upper()} Embedding Initialization", args.verbose, args.display_mode)
        
        embedding_initializer = create_embedding_initializer(args)
        node_embeddings, rel_embeddings = embedding_initializer.initialize_embeddings(
            train_df, val_df, test_df, num_entities, num_relations, device, args.verbose
        )
        
        node_emb = nn.Parameter(node_embeddings.clone().detach()).to(device)
        rel_emb = nn.Embedding(num_relations, args.embed_dim).to(device)
        rel_emb.weight.data.copy_(rel_embeddings)
        
        print_progress(f"Embeddings ready: Nodes {node_emb.shape}, Relations {rel_emb.weight.shape}", args.verbose, args.display_mode)
        
        # SAVE INITIAL EMBEDDINGS AND MAPPINGS
        dataset_prefix = save_initial_embeddings_and_mappings(
            args, train_df, val_df, test_df, node_embeddings, rel_embeddings, 
            num_entities, num_relations, args.verbose
        )
        
        # BASELINE EVALUATION AFTER EMBEDDING INITIALIZATION
        print_progress(f"\nSTAGE 1.5: Baseline Evaluation (Post-Embedding)", args.verbose, args.display_mode)
        
        baseline_train_hit_at_k = evaluate_hit_at_k(train_loader, None, node_emb, rel_emb, device, args.hit_at_k)
        baseline_val_hit_at_k = evaluate_hit_at_k(val_loader, None, node_emb, rel_emb, device, args.hit_at_k)
        baseline_test_hit_at_k = evaluate_hit_at_k(test_loader, None, node_emb, rel_emb, device, args.hit_at_k)
        
        print_progress(f"BASELINE RESULTS (Post R-GCN + DistMult Initialization):", args.verbose, args.display_mode)
        print_progress("Train: " + " ".join([f"Hit@{k}: {baseline_train_hit_at_k[k]:.4f}" for k in args.hit_at_k]), args.verbose, args.display_mode)
        print_progress("Val:   " + " ".join([f"Hit@{k}: {baseline_val_hit_at_k[k]:.4f}" for k in args.hit_at_k]), args.verbose, args.display_mode)
        print_progress("Test:  " + " ".join([f"Hit@{k}: {baseline_test_hit_at_k[k]:.4f}" for k in args.hit_at_k]), args.verbose, args.display_mode)
        print_progress("=" * 70, args.verbose, args.display_mode)
        
        # SAVE R-GCN CHECKPOINT
        save_phase_checkpoint("rgcn", 0, None, None, node_emb, rel_emb, {
            "baseline_results": {
                "train": baseline_train_hit_at_k,
                "val": baseline_val_hit_at_k,
                "test": baseline_test_hit_at_k
            },
            "config": vars(args)
        }, args, dataset_prefix, args.verbose)
        
        # STAGE 2.5: DISTMULT WARM-UP
        print_progress(f"STAGE 2.5: DistMult Warm-up (Your Exact Method)", args.verbose, args.display_mode)
        
        # DistMult warm-up optimizer (your exact settings)
        dm_opt = optim.Adam([node_emb, rel_emb.weight], lr=args.distmult_warmup_lr)
        
        print_progress(f"🚀 Starting DistMult warm-up ({args.distmult_warmup_epochs} epochs for proper initialization)...", args.verbose, args.display_mode)
        if device.type == 'cuda':
            print_progress("   GPU acceleration: Expected completion in 2-3 minutes", args.verbose, args.display_mode)
        else:
            print_progress("   CPU mode: Will take longer, but thorough initialization", args.verbose, args.display_mode)

        for epoch in range(args.distmult_warmup_epochs):  # Configurable epochs
            for batch in train_loader:
                h_batch, r_batch, t_batch = [b.to(device) for b in batch.T]

                # positive score: batch_size-vector 
                pos_scores = ( node_emb[h_batch]
                             * rel_emb(r_batch)
                             * node_emb[t_batch]
                             ).sum(dim=1)

                # sample a random tail for each example 
                neg_t = torch.randint(0, node_emb.size(0), t_batch.shape, device=device)
                neg_scores = ( node_emb[h_batch]
                             * rel_emb(r_batch)
                             * node_emb[neg_t]
                             ).sum(dim=1)

                # margin loss, averaged over the batch
                loss = F.relu(1 + neg_scores - pos_scores).mean()

                dm_opt.zero_grad()
                loss.backward()
                dm_opt.step()

            if epoch % 10 == 0 or epoch == args.distmult_warmup_epochs - 1:
                print_progress(f"   Warm-up epoch {epoch+1}/{args.distmult_warmup_epochs} completed", args.verbose, args.display_mode)

        # Start-of-training evaluation (your exact method)
        warmup_hit_at_k = evaluate_hit_at_k(val_loader, None, node_emb, rel_emb, device, args.hit_at_k)  # None = no generator yet
        print_progress(f"Warm-up Hit@1: {warmup_hit_at_k[1]:.4f}, Hit@5: {warmup_hit_at_k[5]:.4f}, Hit@10: {warmup_hit_at_k[10]:.4f}", args.verbose, args.display_mode)
        print_progress("Warm-up evaluation complete.", args.verbose, args.display_mode)
        
        # SAVE WARMUP CHECKPOINT
        save_phase_checkpoint("warmup", args.distmult_warmup_epochs, None, None, node_emb, rel_emb, {
            "warmup_results": warmup_hit_at_k
        }, args, dataset_prefix, args.verbose)
        
        print_progress("=" * 70, args.verbose, args.display_mode)

        # Initialize models
        print_progress(f"STAGE 2: Model Architecture Setup", args.verbose, args.display_mode)
        
        generator = Generator(args.embed_dim, args.noise_dim).to(device)
        discriminator = Discriminator(args.embed_dim, args.hidden_dim).to(device)
        
        g_opt = optim.Adam(list(generator.parameters()) + [node_emb], lr=args.g_lr)
        d_opt = optim.Adam(discriminator.parameters(), lr=args.d_lr)
        
        print_progress(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}", args.verbose, args.display_mode)
        print_progress(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}", args.verbose, args.display_mode)
        
        # Training state
        best_val_hit10 = 0.0
        best_epoch = 0
        
        # Training schedule
        print_progress(f"STAGE 3: Complete Training Pipeline (Your Exact Method)", args.verbose, args.display_mode)
        print_progress(f"  Phase 1: R-GCN + DistMult Initialization", args.verbose, args.display_mode)
        print_progress(f"  Phase 2: DistMult Warm-up ({args.distmult_warmup_epochs} epochs)", args.verbose, args.display_mode)
        print_progress(f"  Phase 3: Pretraining (Epochs 1-{args.pretrain_epochs})", args.verbose, args.display_mode)
        print_progress(f"  Phase 4: Tier 2 - RL System (Epoch {args.rl_start_epoch}+)", args.verbose, args.display_mode)
        print_progress(f"  Phase 5: Tier 3 - Full Adversarial (Epoch {args.full_system_epoch}+)", args.verbose, args.display_mode)
        
        # Training history (from your non-modular code)
        training_history = {
            'losses': [], 'd_losses': [], 'g_losses': [], 'cossims': [],
            'train_hitks': {k: [] for k in args.hit_at_k},
            'val_hitks': {k: [] for k in args.hit_at_k},
            'f1_history': [], 'val_f1_history': [],
            'aupr_history': [], 'val_aupr_history': [],
            'mcc_history': [], 'val_mcc_history': [],
            'auc_history': [], 'val_auc_history': [],
            'real_acc_list': [], 'fake_acc_list': [],
            'collapse_hist': deque(maxlen=5), 'diversity_hist': deque(maxlen=5)
        }
        
        bce_loss = nn.BCEWithLogitsLoss()
        
        # MAIN TRAINING LOOP
        for epoch in range(1, args.epochs + 1):
            
            # PRETRAINING PHASE
            if epoch <= args.pretrain_epochs:
                generator.train()
                
                for step, batch in enumerate(train_loader):
                    if args.debug and step >= args.max_debug_steps:
                        break
                    
                    h, r, t = [b.to(device) for b in batch.T]
                    h_emb, r_emb_batch, t_emb = node_emb[h], rel_emb(r), node_emb[t]
                    
                    batch_size = h_emb.size(0)
                    neg_idx = torch.randint(0, node_emb.size(0), (batch_size, args.n_pre_neg), device=device)
                    neg_t_embs = node_emb[neg_idx]
                    
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
                
                # Skip advanced evaluation during pretraining
                if args.display_mode == 'simple':
                    print_progress(f"Epoch {epoch}/{args.epochs} [Pretraining]", args.verbose, args.display_mode)
                else:
                    print_progress(f"[Pretraining] Epoch {epoch}: Basic generator training", args.verbose, args.display_mode)
                
                # SAVE PRETRAINING CHECKPOINT (at the end of pretraining)
                if epoch == args.pretrain_epochs:
                    save_phase_checkpoint("pretrain", epoch, generator, discriminator, node_emb, rel_emb, {
                        "training_history": training_history
                    }, args, dataset_prefix, args.verbose)
                
                continue
            
            # ADVANCED TRAINING (Tiers 2 & 3)
            if epoch == args.pretrain_epochs + 1:
                node_emb.requires_grad_(False)
                g_opt = optim.Adam(generator.parameters(), lr=1e-4)
                print_progress("Transitioning to advanced adversarial training", args.verbose, args.display_mode)
            
            generator.train()
            discriminator.train()
            
            total_loss, total_cos = 0.0, 0.0
            total_d_loss, total_g_loss = 0.0, 0.0
            true_labels, pred_probs = [], []
            
            for step, batch in enumerate(train_loader):
                if args.debug and step >= args.max_debug_steps:
                    break
                
                h, r, t = [b.to(device) for b in batch.T]
                h_emb, r_emb_batch, t_emb = node_emb[h], rel_emb(r), node_emb[t]
                
                # DISCRIMINATOR TRAINING
                if step % args.d_update_freq == 0:
                    d_opt.zero_grad()
                    
                    real_scores = discriminator(h_emb.detach(), r_emb_batch.detach(), t_emb.detach())
                    
                    with torch.no_grad():
                        fake_samples = generator(h_emb.detach(), r_emb_batch.detach())
                    fake_scores = discriminator(h_emb.detach(), r_emb_batch.detach(), fake_samples)
                    
                    neg_indices = generate_balanced_hard_negatives(h_emb.detach(), r_emb_batch.detach(), node_emb,
                                                                  num_hard=30, num_medium=8, num_easy=7)  # Using your values
                    
                    batch_size = h_emb.shape[0]
                    selected_neg_idx = torch.randint(0, neg_indices.shape[1], (batch_size,), device=h_emb.device)
                    final_neg_idx = torch.gather(neg_indices, 1, selected_neg_idx.unsqueeze(1)).squeeze(1)
                    hard_neg_samples = node_emb[final_neg_idx]
                    hard_neg_scores = discriminator(h_emb.detach(), r_emb_batch.detach(), hard_neg_samples)
                    
                    real_labels = torch.full_like(real_scores, 0.8)
                    fake_labels = torch.full_like(fake_scores, 0.2)
                    hard_neg_labels = torch.full_like(hard_neg_scores, 0.1)
                    
                    d_loss = (bce_loss(real_scores, real_labels) +
                              bce_loss(fake_scores, fake_labels) +
                              bce_loss(hard_neg_scores, hard_neg_labels)) / 3
                    
                    d_loss.backward()
                    d_opt.step()
                    total_d_loss += d_loss.item()
                    
                    # Track discriminator performance (your metrics)
                    with torch.no_grad():
                        real_preds = (torch.sigmoid(real_scores) > 0.5).float()
                        fake_preds = (torch.sigmoid(torch.cat([fake_scores, hard_neg_scores])) <= 0.5).float()
                        real_acc = real_preds.mean().item()
                        fake_acc = fake_preds.mean().item()
                        
                        training_history['real_acc_list'].append(real_acc)
                        training_history['fake_acc_list'].append(fake_acc)
                        
                        true_labels.extend([1]*len(real_scores) + [0]*(len(fake_scores) + len(hard_neg_scores)))
                        pred_probs.extend(torch.sigmoid(torch.cat([real_scores, fake_scores, hard_neg_scores])).cpu().numpy())
                
                # GENERATOR TRAINING
                g_opt.zero_grad()
                fake = generator(h_emb, r_emb_batch)
                
                loss_components = []
                
                # Refinement loss
                refinement_loss = F.mse_loss(fake, t_emb)
                loss_components.append(args.refinement_weight * refinement_loss)
                
                # DistMult loss
                dm_scores = (h_emb * r_emb_batch * fake).sum(dim=-1)
                dm_loss = -torch.tanh(dm_scores / 10.0).mean()
                loss_components.append(args.distmult_weight * dm_loss)
                
                # Your RL loss
                rl_loss, rl_metrics = compute_composite_rl_loss(
                    epoch, h_emb, r_emb_batch, fake, t_emb, discriminator, args,
                    true_labels=true_labels,
                    pred_probs=pred_probs
                )
                if rl_loss.item() != 0:
                    loss_components.append(args.rl_weight * rl_loss)
                
                # Bias mitigation
                bias_loss = bias_mitigation_loss(fake, t_emb, node_emb, h_emb, r_emb_batch)
                loss_components.append(args.bias_weight * bias_loss)
                
                # Adversarial loss (Tier 3 only)
                if epoch >= args.pretrain_epochs + args.full_system_epoch:
                    adv_scores = discriminator(h_emb, r_emb_batch, fake)
                    adv_loss = -torch.tanh(adv_scores / 5.0).mean()
                    loss_components.append(args.adv_weight * adv_loss)
                
                # Cosine margin and diversity losses
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
                
                l2_loss = (fake - t_emb.detach()).pow(2).mean()
                fake_std = fake.std(dim=0).mean()
                diversity_loss = 0.2 * torch.exp(-5 * fake_std)
                
                final_loss = sum(loss_components) + args.g_guidance_weight * cos_margin + args.l2_reg_weight * l2_loss + diversity_loss
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_norm)
                g_opt.step()
                total_g_loss += final_loss.item()
                
                with torch.no_grad():
                    cos_sim = F.cosine_similarity(fake, t_emb).mean().item()
                    total_loss += final_loss.item()
                    total_cos += cos_sim
            
            # EPOCH EVALUATION (YOUR STYLE)
            effective_steps = min(len(train_loader), args.max_debug_steps if args.debug else len(train_loader))
            avg_loss = total_loss / effective_steps if effective_steps > 0 else 0.0
            avg_cos = total_cos / effective_steps if effective_steps > 0 else 0.0
            avg_d_loss = total_d_loss / max(1, effective_steps // args.d_update_freq)
            avg_g_loss = total_g_loss / effective_steps
            
            # Update history
            training_history['losses'].append(avg_loss)
            training_history['cossims'].append(avg_cos)
            training_history['d_losses'].append(avg_d_loss)
            training_history['g_losses'].append(avg_g_loss)
            
            # Discriminator metrics (your enhanced version)
            enhanced_metrics = print_enhanced_discriminator_metrics(true_labels, pred_probs, epoch, args.display_mode, args.detailed_metrics)
            training_history['f1_history'].append(enhanced_metrics['F1'])
            training_history['aupr_history'].append(enhanced_metrics['AUPR'])
            training_history['mcc_history'].append(enhanced_metrics['MCC'])
            training_history['auc_history'].append(enhanced_metrics['AUC'])
            
            # Hit@K evaluation
            train_hit_at_k = evaluate_hit_at_k(train_loader, generator, node_emb, rel_emb, device, args.hit_at_k)
            for k in args.hit_at_k:
                training_history['train_hitks'][k].append(train_hit_at_k[k])
            
            val_metrics, val_cos_avg = validate(val_loader, generator, discriminator, node_emb, rel_emb, device)
            training_history['val_f1_history'].append(val_metrics['F1'])
            training_history['val_aupr_history'].append(val_metrics['AUPR'])
            training_history['val_mcc_history'].append(val_metrics['MCC'])
            training_history['val_auc_history'].append(val_metrics['AUC'])
            
            val_hit_at_k = evaluate_hit_at_k(val_loader, generator, node_emb, rel_emb, device, args.hit_at_k)
            for k in args.hit_at_k:
                training_history['val_hitks'][k].append(val_hit_at_k[k])
            
            # Calculate collapse and diversity metrics (your additional metrics)
            if args.detailed_metrics:
                with torch.no_grad():
                    fake_eval = generator(h_emb, r_emb_batch).detach()
                    collapse_score = avg_pairwise_cosine_similarity(fake_eval)
                    variance_score = compute_variance(fake_eval)
                    diversity_score = compute_topk_diversity(fake_eval, node_emb)
                    
                    training_history['collapse_hist'].append(collapse_score)
                    training_history['diversity_hist'].append(diversity_score)
            
            # Save best model
            if val_hit_at_k[10] > best_val_hit10:
                best_val_hit10 = val_hit_at_k[10]
                best_epoch = epoch
                
                # ENHANCED FINAL CHECKPOINT SAVE
                os.makedirs(args.output_dir, exist_ok=True)
                checkpoint_path = os.path.join(args.output_dir, "best_checkpoint.pt")
                torch.save({
                    "phase": "training_complete",
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "node_emb": node_emb.detach().cpu(),
                    "rel_emb": rel_emb.state_dict(),
                    "epoch": epoch,
                    "best_val_hit10": best_val_hit10,
                    "best_epoch": best_epoch,
                    
                    # Full configuration
                    "args": vars(args),
                    "model_config": {
                        "generator_params": sum(p.numel() for p in generator.parameters()),
                        "discriminator_params": sum(p.numel() for p in discriminator.parameters()),
                        "embedding_dims": {
                            "nodes": tuple(node_emb.shape),
                            "relations": tuple(rel_emb.weight.shape)
                        }
                    },
                    
                    # Complete training history
                    "training_history": training_history,
                    
                    # All baseline results
                    "baseline_results": {
                        "rgcn_init": {"train": baseline_train_hit_at_k, "val": baseline_val_hit_at_k, "test": baseline_test_hit_at_k},
                        "post_warmup": {"val": warmup_hit_at_k}
                    },
                    
                    # Current results
                    "current_results": {
                        "train_hit_at_k": train_hit_at_k,
                        "val_hit_at_k": val_hit_at_k,
                        "val_metrics": val_metrics
                    },
                    
                    # Reproducibility info
                    "reproducibility": {
                        "torch_version": torch.__version__,
                        "torch_geometric_available": TORCH_GEOMETRIC_AVAILABLE,
                        "device": str(device),
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "dataset_prefix": dataset_prefix
                    }
                }, checkpoint_path)
            
            # Progress display 
            tier_name = "Pretraining" if epoch <= args.pretrain_epochs else f"Tier {2 if epoch < args.pretrain_epochs + args.full_system_epoch else 3}"
            
            # Enhanced RL information display 
            rl_info = ""
            if epoch >= args.rl_start_epoch:
                if epoch < args.pretrain_epochs + args.full_system_epoch:
                    rl_info = f" | RL_DM: {rl_metrics.get('dm_component', 0):.3f} (Tier 2)"
                else:
                    rl_disc = rl_metrics.get('disc_component', 0)
                    rl_dm = rl_metrics.get('dm_component', 0)
                    protection = rl_metrics.get('protection_status', 'Unknown')
                    rl_info = f" | RL_Disc: {rl_disc:.3f} RL_DM: {rl_dm:.3f} (Tier 3 - {protection})"
            
            if args.display_mode == 'simple':
                print_progress(f"E{epoch:03d} [{tier_name}] Loss {avg_loss:.4f} F1 {enhanced_metrics['F1']:.3f} Hit@10 {val_hit_at_k[10]:.4f}{rl_info}", 
                              args.verbose, args.display_mode)
            else:
                print_progress(f"[{tier_name}] E{epoch:03d} | Loss {avg_loss:.4f} | CosSim {avg_cos:.3f} | "
                              f"F1 {enhanced_metrics['F1']:.4f} AUPR {enhanced_metrics['AUPR']:.4f} "
                              f"MCC {enhanced_metrics['MCC']:.4f} AUC {enhanced_metrics['AUC']:.4f}{rl_info}", args.verbose, args.display_mode)
                
                print_progress("Train: " + " ".join([f"Hit@{k}: {train_hit_at_k[k]:.4f}" for k in args.hit_at_k]), 
                              args.verbose, args.display_mode)
                print_progress("Val:   " + " ".join([f"Hit@{k}: {val_hit_at_k[k]:.4f}" for k in args.hit_at_k]) + 
                              f" | VAL_F1 {val_metrics['F1']:.4f} VAL_AUPR {val_metrics['AUPR']:.4f}", args.verbose, args.display_mode)
                
                if args.detailed_metrics:
                    print_progress(f"Collapse: {collapse_score:.3f} | Diversity: {diversity_score:.3f} | Variance: {variance_score:.5f}", 
                                  args.verbose, args.display_mode)
            
            # Tier transition announcements
            if epoch == args.rl_start_epoch:
                print_progress(f" 🎯 TIER 2: RL system activated (DistMult only) - Ultra-aggressive temp scaling (÷100)!", args.verbose, args.display_mode)
            elif epoch == args.pretrain_epochs + args.full_system_epoch:
                print_progress(f" 🎯 TIER 3: Full composite RL + Adversarial system - Adaptive weights based on discriminator health!", args.verbose, args.display_mode)
            
            # Early stopping
            if epoch - best_epoch > args.early_stopping_patience:
                print_progress(f"Early stopping at epoch {epoch}", args.verbose, args.display_mode)
                break
        
        # FINAL EVALUATION
        print_progress(f"\nTRAINING COMPLETED!", args.verbose, args.display_mode)
        print_progress(f"Best validation Hit@10: {best_val_hit10:.4f} at epoch {best_epoch}", args.verbose, args.display_mode)
        
        # Test evaluation
        test_hit_at_k = evaluate_hit_at_k(test_loader, generator, node_emb, rel_emb, device, args.hit_at_k)
        final_val_metrics, _ = validate(val_loader, generator, discriminator, node_emb, rel_emb, device)
        
        print_progress(f"\nFINAL TEST RESULTS:", args.verbose, args.display_mode)
        for k in args.hit_at_k:
            print_progress(f"Test Hit@{k}: {test_hit_at_k[k]:.4f} ({test_hit_at_k[k]*100:.1f}%)", args.verbose, args.display_mode)
        
        print_progress(f"IMPROVEMENT OVER BASELINES:", args.verbose, args.display_mode)
        print_progress(f"Over R-GCN Init:", args.verbose, args.display_mode)
        for k in args.hit_at_k:
            improvement = test_hit_at_k[k] - baseline_test_hit_at_k[k]
            print_progress(f"  Hit@{k}: {improvement:+.4f} ({improvement*100:+.1f}%)", args.verbose, args.display_mode)
        print_progress(f"Over Post-Warmup:", args.verbose, args.display_mode)
        for k in args.hit_at_k:
            improvement = test_hit_at_k[k] - warmup_hit_at_k[k]
            print_progress(f"  Hit@{k}: {improvement:+.4f} ({improvement*100:+.1f}%)", args.verbose, args.display_mode)
        
        if args.display_mode == 'detailed':
            print_progress(f"\nDISCRIMINATOR FINAL HEALTH:", args.verbose, args.display_mode)
            print_progress(f"   F1 Score: {final_val_metrics['F1']:.3f}", args.verbose, args.display_mode)
            print_progress(f"   AUPR:     {final_val_metrics['AUPR']:.3f}", args.verbose, args.display_mode)
            print_progress(f"   AUC:      {final_val_metrics['AUC']:.3f}", args.verbose, args.display_mode)
        
        # Final comprehensive checkpoint save
        final_checkpoint_path = os.path.join(args.output_dir, "final_complete_checkpoint.pt")
        torch.save({
            "phase": "final_complete",
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "node_emb": node_emb.detach().cpu(),
            "rel_emb": rel_emb.state_dict(),
            "epoch": epoch,
            "best_val_hit10": best_val_hit10,
            "best_epoch": best_epoch,
            "args": vars(args),
            "training_history": training_history,
            "final_results": {
                "test_hit_at_k": test_hit_at_k,
                "final_val_metrics": final_val_metrics
            },
            "all_baselines": {
                "rgcn_init": {"train": baseline_train_hit_at_k, "val": baseline_val_hit_at_k, "test": baseline_test_hit_at_k},
                "post_warmup": {"val": warmup_hit_at_k}
            },
            "reproducibility": {
                "torch_version": torch.__version__,
                "device": str(device),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "dataset_prefix": dataset_prefix
            }
        }, final_checkpoint_path)
        
        print_progress(f"\n SAVED FILES SUMMARY:", args.verbose, args.display_mode)
        print_progress(f"Initial Embeddings (Reusable):", args.verbose, args.display_mode)
        print_progress(f"  • {dataset_prefix}_final_node_embeddings.pt", args.verbose, args.display_mode)
        print_progress(f"  • {dataset_prefix}_final_distmult_rel_emb.pt", args.verbose, args.display_mode)
        print_progress(f"  • {dataset_prefix}_*_map.pkl files (4 files)", args.verbose, args.display_mode)
        print_progress(f"Phase Checkpoints:", args.verbose, args.display_mode)
        print_progress(f"  • {dataset_prefix}_rgcn_checkpoint.pt", args.verbose, args.display_mode)
        print_progress(f"  • {dataset_prefix}_warmup_checkpoint.pt", args.verbose, args.display_mode)
        print_progress(f"  • {dataset_prefix}_pretrain_checkpoint.pt", args.verbose, args.display_mode)
        print_progress(f"Final Models:", args.verbose, args.display_mode)
        print_progress(f"  • best_checkpoint.pt (best validation)", args.verbose, args.display_mode)
        print_progress(f"  • final_complete_checkpoint.pt (complete final)", args.verbose, args.display_mode)
        
        # Return results for inference
        return {
            "generator": generator,
            "discriminator": discriminator,
            "node_emb": node_emb,
            "rel_emb": rel_emb,
            "test_hit_at_k": test_hit_at_k,
            "baseline_hit_at_k": baseline_test_hit_at_k,
            "warmup_hit_at_k": warmup_hit_at_k,
            "best_val_hit10": best_val_hit10,
            "training_history": training_history,
            "device": device,
            "dataset_prefix": dataset_prefix
        }
        
    except Exception as e:
        print_progress(f"Training failed: {e}", args.verbose, args.display_mode)
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Modular Prot-B-GAN with Complete Saving System')
    
    # Data paths
    parser.add_argument('--data_root', type=str, required=True, help='Root directory containing data files')
    parser.add_argument('--train_file', type=str, default='prothgt-train-graph_triplets.csv', help='Training data file')
    parser.add_argument('--val_file', type=str, default='prothgt-val-graph_triplets.csv', help='Validation data file')
    parser.add_argument('--test_file', type=str, default='prothgt-test-graph_triplets.csv', help='Test data file')
    parser.add_argument('--output_dir', type=str, default='./modular_results', help='Output directory')
    
    # Model architecture
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--noise_dim', type=int, default=64, help='Generator noise dimension')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Discriminator hidden dimension')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=500, help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--g_lr', type=float, default=1e-3, help='Generator learning rate')
    parser.add_argument('--d_lr', type=float, default=1e-4, help='Discriminator learning rate')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Gradient clipping')
    
    # Embedding initialization
    parser.add_argument('--embedding_init', type=str, default='rgcn', 
                     choices=['rgcn', 'preloaded', 'random', 'transe', 'distmult', 'complex'], 
                     help='Embedding initialization method')
    parser.add_argument('--reward_scoring_method', type=str, default='distmult',
                        choices=['distmult', 'transe', 'auto'],
                        help='Reward scoring method')
    
    # R-GCN specific (updated)
    parser.add_argument('--rgcn_epochs', type=int, default=100, help='R-GCN training epochs')
    parser.add_argument('--rgcn_lr', type=float, default=0.01, help='R-GCN learning rate')
    parser.add_argument('--rgcn_layers', type=int, default=2, help='R-GCN layers')
    parser.add_argument('--rgcn_l2_penalty', type=float, default=0.01, help='R-GCN L2 penalty')
    parser.add_argument('--rgcn_early_stopping_patience', type=int, default=50, help='R-GCN early stopping patience')

    # TransE specific parameters
    parser.add_argument('--transe_epochs', type=int, default=100, help='TransE training epochs')
    parser.add_argument('--transe_lr', type=float, default=0.01, help='TransE learning rate')
    parser.add_argument('--transe_margin', type=float, default=1.0, help='TransE margin')

    # DistMult specific parameters  
    parser.add_argument('--distmult_epochs', type=int, default=100, help='DistMult training epochs')
    parser.add_argument('--distmult_lr', type=float, default=0.01, help='DistMult learning rate')
    parser.add_argument('--distmult_regularization', type=float, default=0.01, help='DistMult L2 regularization')

    # ComplEx specific parameters
    parser.add_argument('--complex_epochs', type=int, default=100, help='ComplEx training epochs')
    parser.add_argument('--complex_lr', type=float, default=0.01, help='ComplEx learning rate')
    parser.add_argument('--complex_regularization', type=float, default=0.01, help='ComplEx L2 regularization')
    
    # DistMult warm-up specific
    parser.add_argument('--distmult_warmup_epochs', type=int, default=50, help='DistMult warm-up epochs')
    parser.add_argument('--distmult_warmup_lr', type=float, default=1e-2, help='DistMult warm-up learning rate')
    
    # Tiered training schedule
    parser.add_argument('--pretrain_epochs', type=int, default=90, help='Pretraining epochs')
    parser.add_argument('--rl_start_epoch', type=int, default=20, help='RL start epoch (Tier 2)')
    parser.add_argument('--full_system_epoch', type=int, default=25, help='Full system epoch (Tier 3)')
    parser.add_argument('--d_update_freq', type=int, default=10, help='Discriminator update frequency')
    
    # Loss weights
    parser.add_argument('--rl_weight', type=float, default=0.1, help='RL loss weight')
    parser.add_argument('--adv_weight', type=float, default=0.05, help='Adversarial loss weight')
    parser.add_argument('--refinement_weight', type=float, default=0.7, help='Refinement loss weight')
    parser.add_argument('--distmult_weight', type=float, default=0.2, help='DistMult loss weight')
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
    
    # Display and debug options (UPDATED)
    parser.add_argument('--display_mode', type=str, default='detailed', choices=['simple', 'detailed'],
                        help='Display mode: simple (one-line) or detailed (multi-line)')
    parser.add_argument('--detailed_metrics', action='store_true', default=True, 
                        help='Enable detailed metric tracking (collapse, diversity, enhanced discriminator metrics)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Debug mode settings
    parser.add_argument('--max_train_samples', type=int, default=20000, help='Max training samples in debug mode')
    parser.add_argument('--max_val_samples', type=int, default=4000, help='Max validation samples in debug mode')
    parser.add_argument('--max_test_samples', type=int, default=4000, help='Max test samples in debug mode')
    parser.add_argument('--max_debug_steps', type=int, default=200, help='Max steps per epoch in debug mode')
    
    # Utility options
    parser.add_argument('--save_embeddings_only', action='store_true', help='Only create and save initial embeddings, then exit')
    parser.add_argument('--inference_mode', action='store_true', help='Load existing model for inference only')
    
    args = parser.parse_args()
    
    print(f"MODULAR PROT-B-GAN WITH COMPLETE SAVING SYSTEM")
    print(f"  Embedding Init: {args.embedding_init.upper()}")
    print(f"  Reward Method: {args.reward_scoring_method.upper()}")
    print(f"  Display Mode: {args.display_mode}")
    print(f"  Detailed Metrics: {args.detailed_metrics}")
    print(f"  Debug Mode: {args.debug}")
    print(f"  Save Only Mode: {args.save_embeddings_only}")
    
    # Run training
    results = train_modular_prot_b_gan(args)
    
    if results:
        print(f"\n SUCCESS! Complete Pipeline Training Completed")
        print(f"Best validation Hit@10: {results['best_val_hit10']:.4f}")
        print(f"R-GCN Baseline Test Hit@10: {results['baseline_hit_at_k'][10]:.4f}")
        print(f"Post-Warmup Test Hit@10: {results['warmup_hit_at_k'][10]:.4f}")
        print(f"Final Test Hit@10: {results['test_hit_at_k'][10]:.4f}")
        print(f"Total Improvement: {results['test_hit_at_k'][10] - results['baseline_hit_at_k'][10]:+.4f}")
        print(f"All models and embeddings saved to: {args.output_dir}")
        print(f"Dataset files saved with prefix: {results['dataset_prefix']}")
        return 0
    else:
        print(f" Training failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
