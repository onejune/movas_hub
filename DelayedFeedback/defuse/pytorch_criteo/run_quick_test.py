#!/usr/bin/env python3
"""
Quick test with 1% data (~160K samples)
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
from metrics import evaluate, ScalarMovingAverage
from trainer import CriteoDataset, collate_fn


def run_method(data, method, config, aux_models=None, device='cpu'):
    """Run a single method."""
    print(f"\n{'='*40}")
    print(f"Running {method}")
    print(f"{'='*40}")
    
    # Prepare data
    ob_win = config.data.observation_window
    attr_win = config.data.attribution_window
    
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
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Train data: {len(train_data)} samples, pos_rate={train_data.labels.mean()*100:.2f}%")
    
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
    }
    loss_fn = get_loss_fn(loss_map[method])
    
    # Train
    model.train()
    if aux_models:
        for m in aux_models.values():
            m.eval()
    
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
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
                if "defer" in aux_models:
                    aux_out = aux_models["defer"](batch["num_features"], batch["cat_features"])
                    outputs["dp_logits"] = aux_out["logits"]
        
        loss = loss_fn(outputs, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Training loss: {total_loss/len(loader):.4f}")
    
    # Evaluate on original data (oracle labels)
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
        for batch in test_loader:
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
    print(f"Results: AUC={metrics['auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, LogLoss={metrics['logloss']:.4f}")
    
    return metrics


def main():
    print("="*60)
    print("DEFUSE Quick Test (1% data)")
    print("="*60)
    
    config = Config()
    device = 'cpu'
    
    # Load 1% data
    print("\nLoading data...")
    start = time.time()
    features, click_ts, pay_ts = load_criteo_data(config.data.data_path)
    
    # Sample 1%
    n_samples = len(features) // 100
    print(f"Using {n_samples} samples (1% of {len(features)})")
    
    np.random.seed(42)
    idx = np.random.choice(len(features), n_samples, replace=False)
    idx.sort()
    
    features = features.iloc[idx].reset_index(drop=True)
    click_ts = click_ts[idx]
    pay_ts = pay_ts[idx]
    
    data = DataDF.from_raw(features, click_ts, pay_ts, config.data.attribution_window)
    print(f"Data: {len(data)} samples, pos_rate={data.labels.mean()*100:.2f}%")
    print(f"Loaded in {time.time()-start:.1f}s")
    
    # Pretrain auxiliary models
    print("\n" + "="*60)
    print("Phase 1: Pretraining Auxiliary Models")
    print("="*60)
    
    pretrain_data = data.construct_tn_dp_data(
        config.data.observation_window,
        config.data.attribution_window
    )
    
    # ES-DFM pretrain model
    print("\nPretraining tn/dp model...")
    esdfm_model = get_model('MLP_tn_dp', hidden_dims=config.model.hidden_dims)
    dataset = CriteoDataset(pretrain_data, config.data.cat_bin_sizes)
    loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(esdfm_model.parameters(), lr=config.train.learning_rate)
    loss_fn = get_loss_fn('pretrain_tn_dp')
    
    esdfm_model.train()
    for batch in tqdm(loader, desc="Pretrain tn/dp"):
        optimizer.zero_grad()
        outputs = esdfm_model(batch["num_features"], batch["cat_features"])
        loss = loss_fn(outputs, batch)
        loss.backward()
        optimizer.step()
    
    # DEFER pretrain model
    print("\nPretraining dp model...")
    defer_model = get_model('MLP_dp', hidden_dims=config.model.hidden_dims)
    optimizer = torch.optim.Adam(defer_model.parameters(), lr=config.train.learning_rate)
    loss_fn = get_loss_fn('pretrain_dp')
    
    defer_model.train()
    for batch in tqdm(loader, desc="Pretrain dp"):
        optimizer.zero_grad()
        outputs = defer_model(batch["num_features"], batch["cat_features"])
        loss = loss_fn(outputs, batch)
        loss.backward()
        optimizer.step()
    
    # Run all methods
    print("\n" + "="*60)
    print("Phase 2: Training and Evaluation")
    print("="*60)
    
    results = {}
    methods = ["Vanilla", "FNW", "FNC", "DEFER", "ES-DFM", "DEFUSE", "Bi-DEFUSE"]
    # Skip DFM for now (needs delay/elapsed time handling)
    
    for method in methods:
        aux_models = None
        if method in ["ES-DFM", "DEFUSE", "Bi-DEFUSE"]:
            aux_models = {"esdfm": esdfm_model}
        elif method == "DEFER":
            aux_models = {"defer": defer_model}
        
        try:
            results[method] = run_method(data, method, config, aux_models, device)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results[method] = {"auc": 0, "pr_auc": 0, "logloss": 0}
    
    # Summary
    print("\n" + "="*60)
    print("Final Results Summary")
    print("="*60)
    print(f"{'Method':<12} | {'AUC':>8} | {'PR-AUC':>8} | {'LogLoss':>8}")
    print("-" * 45)
    for method, metrics in sorted(results.items(), key=lambda x: -x[1]['auc']):
        print(f"{method:<12} | {metrics['auc']:>8.4f} | {metrics['pr_auc']:>8.4f} | {metrics['logloss']:>8.4f}")
    
    # Save results
    os.makedirs(config.data.output_dir, exist_ok=True)
    with open(os.path.join(config.data.output_dir, "quick_test_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
