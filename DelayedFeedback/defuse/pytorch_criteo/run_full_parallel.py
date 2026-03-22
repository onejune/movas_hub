#!/usr/bin/env python3
"""
Run full data experiments in parallel (one method per call)
Usage: python run_full_parallel.py <method>
"""
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

from config import Config, SECONDS_AN_HOUR
from data import load_criteo_data, DataDF
from models import get_model
from loss import get_loss_fn
from metrics import evaluate
from trainer import CriteoDataset, collate_fn


def run_method(method: str, sample_ratio: float = 1.0):
    """Run a single method on full or sampled data."""
    print(f"Starting {method}...")
    start_time = time.time()
    
    config = Config()
    device = 'cpu'
    
    # Load data
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
    
    data = DataDF.from_raw(features, click_ts, pay_ts, config.data.attribution_window)
    print(f"Data: {len(data)} samples, pos_rate={data.labels.mean()*100:.2f}%")
    
    ob_win = config.data.observation_window
    attr_win = config.data.attribution_window
    
    # Prepare train data
    if method == "Vanilla":
        train_data = data.form_vanilla(ob_win)
    elif method in ["FNW", "FNC"]:
        train_data = data.form_vanilla(ob_win)
    elif method == "DFM":
        train_data = data.add_fake_neg(attr_win)
    elif method == "DEFER":
        train_data = data.add_defer_duplicate_samples(ob_win, attr_win)
    elif method == "ES-DFM":
        train_data = data.add_esdfm_cut_fake_neg(ob_win)
    elif method in ["DEFUSE", "Bi-DEFUSE"]:
        train_data = data.add_inw_outw_delay_positive(ob_win)
    elif method == "Oracle":
        train_data = data.form_oracle()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Train data: {len(train_data)} samples, pos_rate={train_data.labels.mean()*100:.2f}%")
    
    # Pretrain auxiliary models if needed
    aux_models = {}
    if method in ["ES-DFM", "DEFUSE", "DEFER"]:
        print("\nPretraining tn/dp model...")
        pretrain_data = data.construct_tn_dp_data(ob_win, attr_win)
        pretrain_dataset = CriteoDataset(pretrain_data, config.data.cat_bin_sizes)
        pretrain_loader = DataLoader(
            pretrain_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        esdfm_model = get_model(
            "MLP_tn_dp",
            hidden_dims=config.model.hidden_dims,
            embed_dim=config.data.embed_dim,
            use_bn=config.model.use_batch_norm,
        ).to(device)
        
        optimizer = torch.optim.Adam(esdfm_model.parameters(), lr=config.train.learning_rate)
        loss_fn = get_loss_fn("pretrain_tn_dp")
        
        esdfm_model.train()
        for batch in tqdm(pretrain_loader, desc="Pretrain"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = esdfm_model(batch["num_features"], batch["cat_features"])
            loss = loss_fn(outputs, batch)
            loss.backward()
            optimizer.step()
        
        aux_models["esdfm"] = esdfm_model
        esdfm_model.eval()
        print("Pretrain done")
    
    # Create model
    if method == "Bi-DEFUSE":
        model_type = "Bi-DEFUSE"
    elif method == "DFM":
        model_type = "MLP_EXP_DELAY"
    else:
        model_type = "MLP_SIG"
    
    model = get_model(
        model_type,
        hidden_dims=config.model.hidden_dims,
        embed_dim=config.data.embed_dim,
        use_bn=config.model.use_batch_norm,
    ).to(device)
    
    # Create dataloader
    dataset = CriteoDataset(train_data, config.data.cat_bin_sizes)
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    loss_map = {
        "Vanilla": "vanilla",
        "FNW": "fnw",
        "FNC": "fnc",
        "DFM": "dfm",
        "DEFER": "defer",
        "ES-DFM": "esdfm",
        "DEFUSE": "defuse",
        "Bi-DEFUSE": "bidefuse",
        "Oracle": "vanilla",  # Oracle uses same loss as Vanilla
    }
    loss_fn = get_loss_fn(loss_map[method])
    
    # Train
    print(f"\nTraining {method}...")
    model.train()
    total_loss = 0
    n_batches = len(loader)
    
    for i, batch in enumerate(tqdm(loader, desc="Training")):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(batch["num_features"], batch["cat_features"])
        
        # Add auxiliary outputs
        if aux_models:
            with torch.no_grad():
                if "esdfm" in aux_models:
                    aux_out = aux_models["esdfm"](batch["num_features"], batch["cat_features"])
                    outputs["tn_logits"] = aux_out["tn_logits"]
                    outputs["dp_logits"] = aux_out["dp_logits"]
        
        loss = loss_fn(outputs, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if (i + 1) % 1000 == 0:
            print(f"  Batch {i+1}/{n_batches}, loss={total_loss/(i+1):.4f}")
    
    print(f"Training loss: {total_loss/len(loader):.4f}")
    
    # Evaluate on original data (oracle labels)
    print("\nEvaluating...")
    test_dataset = CriteoDataset(data, config.data.cat_bin_sizes)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if hasattr(model, 'predict'):
                outputs = model.predict(batch["num_features"], batch["cat_features"])
                probs = outputs.get("prob", torch.sigmoid(outputs.get("logits", outputs["logits_inw"])))
            else:
                outputs = model(batch["num_features"], batch["cat_features"])
                probs = torch.sigmoid(outputs["logits"])
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
    
    all_probs = np.concatenate(all_probs).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    
    metrics = evaluate(all_labels, all_probs, method)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"{method} Results:")
    print(f"  AUC={metrics['auc']:.4f}")
    print(f"  PR-AUC={metrics['pr_auc']:.4f}")
    print(f"  LogLoss={metrics['logloss']:.4f}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"{'='*50}")
    
    # Save results
    output = {
        "method": method,
        "auc": float(metrics['auc']),
        "pr_auc": float(metrics['pr_auc']),
        "logloss": float(metrics['logloss']),
        "time_minutes": elapsed/60,
        "n_samples": len(data),
        "sample_ratio": sample_ratio
    }
    
    os.makedirs("results_full", exist_ok=True)
    with open(f"results_full/{method}.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to results_full/{method}.json")
    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_full_parallel.py <method> [sample_ratio]")
        print("Methods: Vanilla, FNW, FNC, DFM, DEFER, ES-DFM, DEFUSE, Bi-DEFUSE")
        sys.exit(1)
    
    method = sys.argv[1]
    sample_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    run_method(method, sample_ratio)
