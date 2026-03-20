#!/usr/bin/env python3
"""
DEFUSE Training Script
"""

import os
import json
import argparse
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from models import DEFUSEModel
from loss import pretrain_loss, defuse_loss
from data import load_parquet_data, create_dataloader, batch_to_dict, split_features_labels
from metrics import compute_metrics


def train_epoch_pretrain(model, loader, keys, optimizer, device):
    """Stage 1: Pretrain tn/dp classifiers"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    pbar = tqdm(loader, desc="Pretrain")
    for batch in pbar:
        batch_dict = batch_to_dict(batch, keys, device)
        x_dict, targets = split_features_labels(batch_dict)
        
        optimizer.zero_grad()
        outputs = model(x_dict)
        loss = pretrain_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / n_batches


def train_epoch_defuse(model, loader, keys, optimizer, device):
    """Stage 2: Train CVR with DEFUSE loss"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    pbar = tqdm(loader, desc="DEFUSE")
    for batch in pbar:
        batch_dict = batch_to_dict(batch, keys, device)
        x_dict, targets = split_features_labels(batch_dict)
        
        optimizer.zero_grad()
        outputs = model(x_dict)
        loss = defuse_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / n_batches


def evaluate(model, loader, keys, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            batch_dict = batch_to_dict(batch, keys, device)
            x_dict, _ = split_features_labels(batch_dict)
            
            outputs = model(x_dict)
            preds = torch.sigmoid(outputs['cv_logits']).squeeze(-1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_dict['label_oracle'].cpu().numpy())
    
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    
    return compute_metrics(labels, preds)


def main():
    parser = argparse.ArgumentParser(description='DEFUSE Training')
    parser.add_argument('--data_dir', type=str, 
                        default='/mnt/workspace/walter.wan/open_research/defer/data_v4_sample10')
    parser.add_argument('--log_dir', type=str, 
                        default='/mnt/workspace/walter.wan/open_research/DEFUSE/logs_pytorch')
    parser.add_argument('--pretrain_epochs', type=int, default=1)
    parser.add_argument('--finetune_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embed_dim', type=int, default=8)
    parser.add_argument('--hidden_dims', type=str, default='256,256,128')
    args = parser.parse_args()
    
    # Parse hidden dims
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    encoder_path = os.path.join(args.log_dir, 'encoder.json')
    train_x, train_labels, test_x, test_labels, encoder, feature_cols = load_parquet_data(
        args.data_dir, encoder_path
    )
    
    # Create dataloaders
    train_loader, train_keys = create_dataloader(train_x, train_labels, args.batch_size, shuffle=True)
    test_loader, test_keys = create_dataloader(test_x, test_labels, args.batch_size, shuffle=False)
    
    # Model
    model = DEFUSEModel(
        vocab_sizes=encoder.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dims=hidden_dims
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Stage 1: Pretrain
    print("\n" + "="*60)
    print("Stage 1: Pretrain tn/dp classifiers")
    print("="*60)
    
    for epoch in range(args.pretrain_epochs):
        start_time = time.time()
        loss = train_epoch_pretrain(model, train_loader, train_keys, optimizer, device)
        elapsed = time.time() - start_time
        print(f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs}: loss={loss:.4f}, time={elapsed:.1f}s")
    
    # Stage 2: DEFUSE training
    print("\n" + "="*60)
    print("Stage 2: Train CVR with DEFUSE loss")
    print("="*60)
    
    for epoch in range(args.finetune_epochs):
        start_time = time.time()
        loss = train_epoch_defuse(model, train_loader, train_keys, optimizer, device)
        elapsed = time.time() - start_time
        
        # Evaluate
        metrics = evaluate(model, test_loader, test_keys, device)
        print(f"DEFUSE Epoch {epoch+1}/{args.finetune_epochs}: loss={loss:.4f}, "
              f"AUC={metrics['auc']:.4f}, PR-AUC={metrics['prauc']:.4f}, "
              f"LogLoss={metrics['logloss']:.4f}, time={elapsed:.1f}s")
    
    # Final results
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    print(f"Method: DEFUSE")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"PR-AUC: {metrics['prauc']:.4f}")
    print(f"LogLoss: {metrics['logloss']:.4f}")
    
    # Save results
    results = {
        'method': 'defuse',
        'auc': metrics['auc'],
        'prauc': metrics['prauc'],
        'logloss': metrics['logloss'],
        'args': vars(args),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(args.log_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Save model
    model_path = os.path.join(args.log_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()
