"""
Prot-B-GAN Inference Script
============================

Standalone inference script for Prot-B-GAN system.
Loads trained models and perform various inference tasks.

Usage Examples:

    # Predict top-k tails for given head-relation pairs
    python inference.py \
        --checkpoint_path "./modular_results/best_checkpoint.pt" \
        --task predict_tails \
        --input_triplets "[[0, 1], [2, 3]]" \
        --top_k 10

    # Score existing triplets
    python inference.py \
        --checkpoint_path "./modular_results/best_checkpoint.pt" \
        --task score_triplets \
        --input_triplets "[[0, 1, 2], [3, 4, 5]]"

    # Interactive mode
    python inference.py \
        --checkpoint_path "./modular_results/best_checkpoint.pt" \
        --task interactive
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import json
import argparse
import os
from typing import List, Tuple, Dict, Any, Optional

# Import the modular components (assumes they're in the same directory or module)
from modular_prot_b_gan import ModularGenerator, ModularDiscriminator

class ProtBGANInference:
    """Main inference class for Prot-B-GAN"""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Initialize inference system from checkpoint
        
        Args:
            checkpoint_path: Path to saved checkpoint
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        self.checkpoint_path = checkpoint_path
        
        print(f"Loading Prot-B-GAN inference system...")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device: {self.device}")
        
        # Load checkpoint
        self._load_checkpoint()
        
        print(f"Inference ready!")
        print(f"   - Entities: {self.num_entities:,}")
        print(f"   - Relations: {self.num_relations:,}")
        print(f"   - Embedding dim: {self.embed_dim}")
    
    def _load_checkpoint(self):
        """Load the trained models from checkpoint"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Get model dimensions from saved args
        saved_args = checkpoint.get('args', {})
        self.embed_dim = saved_args.get('embed_dim', 128)
        self.noise_dim = saved_args.get('noise_dim', 64)
        self.hidden_dim = saved_args.get('hidden_dim', 1024)
        
        # Get dataset info
        node_emb = checkpoint['node_emb'].to(self.device)
        self.num_entities = node_emb.shape[0]
        self.num_relations = checkpoint['rel_emb']['weight'].shape[0]
        
        print(f"Model dimensions from checkpoint:")
        print(f"  - Embed dim: {self.embed_dim}")
        print(f"  - Entities: {self.num_entities:,}")
        print(f"  - Relations: {self.num_relations:,}")
        
        # Initialize models
        self.generator = Generator(self.embed_dim, self.noise_dim).to(self.device)
        self.discriminator = Discriminator(self.embed_dim, self.hidden_dim).to(self.device)
        
        # Load model states
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        
        # Load embeddings
        self.node_emb = nn.Parameter(node_emb, requires_grad=False)
        self.rel_emb = nn.Embedding(self.num_relations, self.embed_dim).to(self.device)
        self.rel_emb.load_state_dict(checkpoint['rel_emb'])
        
        # Set to eval mode
        self.generator.eval()
        self.discriminator.eval()
        
        # Store training info
        self.best_val_hit10 = checkpoint.get('best_val_hit10', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.training_history = checkpoint.get('training_history', {})
        
        print(f"Model performance:")
        print(f"  - Best validation Hit@10: {self.best_val_hit10:.4f}")
        print(f"  - Achieved at epoch: {self.best_epoch}")
    
    def predict_tails(self, head_relation_pairs: List[Tuple[int, int]], top_k: int = 10, 
                     return_scores: bool = False) -> Dict[str, Any]:
        """
        Predict top-k most likely tail entities for given head-relation pairs
        
        Args:
            head_relation_pairs: List of (head_id, relation_id) pairs
            top_k: Number of top predictions to return
            return_scores: Whether to return prediction scores
            
        Returns:
            Dictionary with predictions and metadata
        """
        print(f"Predicting top-{top_k} tails for {len(head_relation_pairs)} head-relation pairs...")
        
        with torch.no_grad():
            # Convert input to tensors
            heads = torch.tensor([pair[0] for pair in head_relation_pairs], device=self.device)
            relations = torch.tensor([pair[1] for pair in head_relation_pairs], device=self.device)
            
            # Get embeddings
            h_emb = self.node_emb[heads]
            r_emb = self.rel_emb(relations)
            
            # Generate predictions
            pred_emb = self.generator(h_emb, r_emb)
            
            # Calculate similarities to all entities
            pred_norm = F.normalize(pred_emb, dim=-1)
            entity_norm = F.normalize(self.node_emb, dim=-1)
            similarities = torch.matmul(pred_norm, entity_norm.T)
            
            # Get top-k predictions
            top_scores, top_indices = similarities.topk(top_k, dim=1)
            
            results = {
                'predictions': top_indices.cpu().numpy().tolist(),
                'metadata': {
                    'num_queries': len(head_relation_pairs),
                    'top_k': top_k,
                    'model_hit10': self.best_val_hit10
                }
            }
            
            if return_scores:
                results['scores'] = top_scores.cpu().numpy().tolist()
            
            return results
    
    def score_triplets(self, triplets: List[Tuple[int, int, int]], method: str = 'both') -> Dict[str, Any]:
        """
        Score existing triplets using generator and/or discriminator
        
        Args:
            triplets: List of (head_id, relation_id, tail_id) triplets
            method: Scoring method ('generator', 'discriminator', 'both')
            
        Returns:
            Dictionary with scores and metadata
        """
        print(f"Scoring {len(triplets)} triplets using {method}...")
        
        with torch.no_grad():
            # Convert to tensors
            triplet_tensor = torch.tensor(triplets, device=self.device)
            heads, relations, tails = triplet_tensor[:, 0], triplet_tensor[:, 1], triplet_tensor[:, 2]
            
            # Get embeddings
            h_emb = self.node_emb[heads]
            r_emb = self.rel_emb(relations)
            t_emb = self.node_emb[tails]
            
            results = {
                'triplets': triplets,
                'metadata': {
                    'num_triplets': len(triplets),
                    'method': method,
                    'model_hit10': self.best_val_hit10
                }
            }
            
            if method in ['generator', 'both']:
                # Generator-based scoring (similarity between generated and actual tail)
                pred_emb = self.generator(h_emb, r_emb)
                gen_similarities = F.cosine_similarity(pred_emb, t_emb, dim=1)
                results['generator_scores'] = gen_similarities.cpu().numpy().tolist()
            
            if method in ['discriminator', 'both']:
                # Discriminator-based scoring (probability of being real)
                disc_logits, disc_probs = self.discriminator.score_triplets(self.node_emb, self.rel_emb, triplet_tensor)
                results['discriminator_logits'] = disc_logits.tolist()
                results['discriminator_probabilities'] = disc_probs.tolist()
            
            return results
    
    def find_similar_entities(self, entity_ids: List[int], top_k: int = 10) -> Dict[str, Any]:
        """
        Find entities most similar to given entities in the embedding space
        
        Args:
            entity_ids: List of entity IDs to find similarities for
            top_k: Number of similar entities to return
            
        Returns:
            Dictionary with similar entities and scores
        """
        print(f"Finding top-{top_k} similar entities for {len(entity_ids)} query entities...")
        
        with torch.no_grad():
            query_entities = torch.tensor(entity_ids, device=self.device)
            query_emb = self.node_emb[query_entities]
            
            # Calculate similarities
            query_norm = F.normalize(query_emb, dim=-1)
            all_norm = F.normalize(self.node_emb, dim=-1)
            similarities = torch.matmul(query_norm, all_norm.T)
            
            # Get top-k (excluding self)
            top_scores, top_indices = similarities.topk(top_k + 1, dim=1)
            
            results = {
                'similar_entities': [],
                'metadata': {
                    'num_queries': len(entity_ids),
                    'top_k': top_k,
                    'model_hit10': self.best_val_hit10
                }
            }
            
            for i, query_id in enumerate(entity_ids):
                # Remove self from results
                query_top_indices = top_indices[i].cpu().numpy()
                query_top_scores = top_scores[i].cpu().numpy()
                
                # Filter out the query entity itself
                mask = query_top_indices != query_id
                filtered_indices = query_top_indices[mask][:top_k]
                filtered_scores = query_top_scores[mask][:top_k]
                
                results['similar_entities'].append({
                    'query_entity': query_id,
                    'similar_entities': filtered_indices.tolist(),
                    'similarity_scores': filtered_scores.tolist()
                })
            
            return results
    
    def analyze_relations(self, head_ids: List[int], tail_ids: List[int], top_k: int = 5) -> Dict[str, Any]:
        """
        Analyze which relations are most likely between given head and tail entities
        
        Args:
            head_ids: List of head entity IDs
            tail_ids: List of tail entity IDs  
            top_k: Number of top relations to return
            
        Returns:
            Dictionary with relation predictions
        """
        print(f"Analyzing relations between {len(head_ids)} heads and {len(tail_ids)} tails...")
        
        results = {
            'relation_analysis': [],
            'metadata': {
                'num_head_entities': len(head_ids),
                'num_tail_entities': len(tail_ids),
                'top_k': top_k,
                'model_hit10': self.best_val_hit10
            }
        }
        
        with torch.no_grad():
            for head_id in head_ids:
                for tail_id in tail_ids:
                    h_emb = self.node_emb[head_id:head_id+1]
                    t_emb = self.node_emb[tail_id:tail_id+1]
                    
                    relation_scores = []
                    
                    for rel_id in range(self.num_relations):
                        r_emb = self.rel_emb(torch.tensor([rel_id], device=self.device))
                        
                        # Score using discriminator
                        disc_score = self.discriminator(h_emb, r_emb, t_emb).item()
                        disc_prob = torch.sigmoid(torch.tensor(disc_score)).item()
                        
                        relation_scores.append({
                            'relation_id': rel_id,
                            'discriminator_score': disc_score,
                            'probability': disc_prob
                        })
                    
                    # Sort by probability and take top-k
                    relation_scores.sort(key=lambda x: x['probability'], reverse=True)
                    top_relations = relation_scores[:top_k]
                    
                    results['relation_analysis'].append({
                        'head_entity': head_id,
                        'tail_entity': tail_id,
                        'top_relations': top_relations
                    })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_architecture': {
                'embedding_dim': self.embed_dim,
                'noise_dim': self.noise_dim,
                'hidden_dim': self.hidden_dim,
                'num_entities': self.num_entities,
                'num_relations': self.num_relations
            },
            'training_performance': {
                'best_validation_hit10': self.best_val_hit10,
                'best_epoch': self.best_epoch
            },
            'checkpoint_path': self.checkpoint_path,
            'device': str(self.device)
        }

def interactive_mode(inference_system: ProtBGANInference):
    """Interactive mode for exploration"""
    print("\n Prot-B-GAN Interactive Mode")
    print("=" * 50)
    print("Available commands:")
    print("1. predict <head_id> <relation_id> <top_k>  - Predict tails")
    print("2. score <head_id> <relation_id> <tail_id>  - Score triplet") 
    print("3. similar <entity_id> <top_k>              - Find similar entities")
    print("4. info                                     - Model information")
    print("5. help                                     - Show this help")
    print("6. quit                                     - Exit")
    print("=" * 50)
    
    while True:
        try:
            command = input("\n> ").strip().split()
            
            if not command:
                continue
                
            cmd = command[0].lower()
            
            if cmd == 'quit' or cmd == 'exit':
                print("done!")
                break
                
            elif cmd == 'help':
                print("Available commands:")
                print("predict <head_id> <relation_id> <top_k>")
                print("score <head_id> <relation_id> <tail_id>")
                print("similar <entity_id> <top_k>")
                print("info")
                print("quit")
                
            elif cmd == 'predict':
                if len(command) != 4:
                    print("Usage: predict <head_id> <relation_id> <top_k>")
                    continue
                    
                head_id, rel_id, top_k = int(command[1]), int(command[2]), int(command[3])
                results = inference_system.predict_tails([(head_id, rel_id)], top_k, return_scores=True)
                
                print(f"Top {top_k} predictions for ({head_id}, {rel_id}):")
                predictions = results['predictions'][0]
                scores = results['scores'][0]
                
                for i, (pred_id, score) in enumerate(zip(predictions, scores)):
                    print(f"  {i+1:2d}. Entity {pred_id:6d} (score: {score:.4f})")
                    
            elif cmd == 'score':
                if len(command) != 4:
                    print("Usage: score <head_id> <relation_id> <tail_id>")
                    continue
                    
                head_id, rel_id, tail_id = int(command[1]), int(command[2]), int(command[3])
                results = inference_system.score_triplets([(head_id, rel_id, tail_id)], method='both')
                
                print(f"Scores for triplet ({head_id}, {rel_id}, {tail_id}):")
                print(f"  Generator similarity:     {results['generator_scores'][0]:.4f}")
                print(f"  Discriminator probability: {results['discriminator_probabilities'][0]:.4f}")
                print(f"  Discriminator logit:      {results['discriminator_logits'][0]:.4f}")
                
            elif cmd == 'similar':
                if len(command) != 3:
                    print("Usage: similar <entity_id> <top_k>")
                    continue
                    
                entity_id, top_k = int(command[1]), int(command[2])
                results = inference_system.find_similar_entities([entity_id], top_k)
                
                print(f"Top {top_k} entities similar to {entity_id}:")
                similar_data = results['similar_entities'][0]
                
                for i, (sim_id, score) in enumerate(zip(similar_data['similar_entities'], similar_data['similarity_scores'])):
                    print(f"  {i+1:2d}. Entity {sim_id:6d} (similarity: {score:.4f})")
                    
            elif cmd == 'info':
                info = inference_system.get_model_info()
                print("Model Information:")
                print(f"  Entities: {info['model_architecture']['num_entities']:,}")
                print(f"  Relations: {info['model_architecture']['num_relations']:,}")
                print(f"  Embedding dim: {info['model_architecture']['embedding_dim']}")
                print(f"  Best Hit@10: {info['training_performance']['best_validation_hit10']:.4f}")
                print(f"  Device: {info['device']}")
                
            else:
                print(f"Unknown command: {cmd}. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\ndone! ")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Prot-B-GAN Inference System')
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    
    parser.add_argument('--task', type=str, default='interactive',
                       choices=['predict_tails', 'score_triplets', 'similar_entities', 'analyze_relations', 'interactive', 'model_info'],
                       help='Inference task to perform')
    
    parser.add_argument('--input_triplets', type=str, default='',
                       help='Input triplets as JSON string (e.g., "[[0,1,2],[3,4,5]]")')
    
    parser.add_argument('--input_pairs', type=str, default='',
                       help='Input head-relation pairs as JSON string (e.g., "[[0,1],[2,3]]")')
    
    parser.add_argument('--input_entities', type=str, default='',
                       help='Input entity IDs as JSON string (e.g., "[0,1,2,3]")')
    
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top results to return')
    
    parser.add_argument('--output_file', type=str, default='',
                       help='Output file to save results (JSON format)')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Initialize inference system
    inference_system = ProtBGANInference(args.checkpoint_path, args.device)
    
    if args.task == 'interactive':
        interactive_mode(inference_system)
        return
    
    results = None
    
    if args.task == 'model_info':
        results = inference_system.get_model_info()
        
    elif args.task == 'predict_tails':
        if not args.input_pairs:
            print("Error: --input_pairs required for predict_tails task")
            return
        
        pairs = json.loads(args.input_pairs)
        results = inference_system.predict_tails(pairs, args.top_k, return_scores=True)
        
    elif args.task == 'score_triplets':
        if not args.input_triplets:
            print("Error: --input_triplets required for score_triplets task")
            return
        
        triplets = json.loads(args.input_triplets)
        results = inference_system.score_triplets(triplets, method='both')
        
    elif args.task == 'similar_entities':
        if not args.input_entities:
            print("Error: --input_entities required for similar_entities task")
            return
        
        entities = json.loads(args.input_entities)
        results = inference_system.find_similar_entities(entities, args.top_k)
    
    # Output results
    if results:
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output_file}")
        else:
            print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
