#!/usr/bin/env python3
"""Run Oracle baseline - uses true conversion labels."""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import Config
from data import load_criteo_data, DataDF
from models import get_model
from loss import get_loss_fn
from metrics import evaluate
from trainer import CriteoDataset, collate_fn


def run_oracle(sample_ratio: float = 1.0):
    print("Starting Oracle (true labels baseline)...")
    start_time = time.time()
    
    config = Config()
    device = 'cpu'
    
    print("Loading data...")
    features, click_ts, pay_ts = load_criteo_data(config.data.data_path)
    
    if sample_ratio < 1.0:
        n_samples = int(len(features) * sample_ratio)
        print(f"Using {n_samples} samples ({sample_ratio*100:.0f}% of {len(features)})")
        np.random.seed(42)
        idx = np.random.choice(len(features), n_samples, replace=False)
        idx.sort()
        features = features.iloc[idx].reset_index(drop=True)
        click_ts = click_ts[idx]
        pay_ts = pay_ts[idx]
    else:
        print(f"Using full data: {len(features)} samples")
    
    # Oracle uses true labels
    data = DataDF.from_raw(features, click_ts, pay_ts, config.data.attribution_window)
    print(f"Data: {len(data)} samples, pos_rate={data.labels.mean()*100:.2f}%")
    
    # Create model
    model = get_model(
        "MLP_SIG",
        hidden_dims=config.model.hidden_dims,
        embed_dim=config.data.embed_dim,
        use_bn=config.model.use_batch_norm,
    ).to(device)
    
    # Create dataloader
    dataset = CriteoDataset(data, config.data.cat_bin_sizes)
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    loss_fn = get_loss_fn("vanilla")
    
    print("\nTraining Oracle...")
    model.train()
    total_loss = 0
    n_batches = len(loader)
    
    for i, batch in enumerate(tqdm(loader, desc="Training")):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(batch["num_features"], batch["cat_features"])
        loss = loss_fn(outputs, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if (i + 1) % 1000 == 0:
            print(f"  Batch {i+1}/{n_batches}, loss={total_loss/(i+1):.4f}")
    
    print(f"Training loss: {total_loss/len(loader):.4f}")
    
    # Evaluate
    print("\nEvaluating...")
    model.eval()
    all_probs = []
    all_labels = []
    
    test_loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["num_features"], batch["cat_features"])
            probs = torch.sigmoid(outputs["logits"])
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
    
    all_probs = np.concatenate(all_probs).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    
    metrics = evaluate(all_labels, all_probs, "Oracle")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Oracle Results:")
    print(f"  AUC={metrics['auc']:.4f}")
    print(f"  PR-AUC={metrics['pr_auc']:.4f}")
    print(f"  LogLoss={metrics['logloss']:.4f}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"{'='*50}")
    
    output = {
        "method": "Oracle",
        "auc": float(metrics['auc']),
        "pr_auc": float(metrics['pr_auc']),
        "logloss": float(metrics['logloss']),
        "time_minutes": elapsed/60,
        "n_samples": len(data),
        "sample_ratio": sample_ratio
    }
    
    os.makedirs("results_full", exist_ok=True)
    with open("results_full/Oracle.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("Results saved to results_full/Oracle.json")
    return metrics


if __name__ == "__main__":
    sample_ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    run_oracle(sample_ratio)
