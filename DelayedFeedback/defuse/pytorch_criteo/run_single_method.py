"""Run a single method for full data experiment."""
import sys
import warnings
warnings.filterwarnings('ignore')

import time
import json
import numpy as np
import torch

from config import Config
from data import load_criteo_data, DataDF
from trainer import CriteoDataset, Trainer
from models import get_model as create_model
from metrics import evaluate

def run_method(method: str):
    print(f"Starting {method} on full data...")
    start_time = time.time()
    
    config = Config()
    config.training.method = method
    
    # Load full data
    features, click_ts, pay_ts = load_criteo_data(config.data.data_path)
    print(f"Loaded {len(features)} samples")
    
    data = DataDF.from_raw(features, click_ts, pay_ts, config.data.attribution_window)
    
    # Prepare data based on method
    if method in ["Vanilla", "FNW", "FNC"]:
        train_data = data.form_vanilla(config.data.observation_window)
    elif method in ["DEFER", "ES-DFM", "DFM"]:
        train_data = data.add_esdfm_delay_positive(config.data.observation_window)
    elif method in ["DEFUSE", "Bi-DEFUSE"]:
        train_data = data.add_inw_outw_delay_positive(config.data.observation_window)
    
    print(f"Train data: {len(train_data)} samples, pos_rate={train_data.labels.mean()*100:.2f}%")
    
    # Create dataset and trainer
    dataset = CriteoDataset(train_data, config)
    model = create_model(method, config)
    trainer = Trainer(model, config)
    
    # Train
    trainer.train(dataset, method=method)
    
    # Evaluate on test data (last 5 days)
    test_data = data.get_test_data(config.data.test_days)
    test_dataset = CriteoDataset(test_data, config)
    
    results = evaluate(model, test_dataset, config)
    
    elapsed = time.time() - start_time
    print(f"\n{method} Results: AUC={results['auc']:.4f}, PR-AUC={results['pr_auc']:.4f}, LogLoss={results['logloss']:.4f}")
    print(f"Time: {elapsed/60:.1f} minutes")
    
    # Save results
    output = {
        "method": method,
        "auc": results['auc'],
        "pr_auc": results['pr_auc'],
        "logloss": results['logloss'],
        "time_minutes": elapsed/60
    }
    with open(f"results_{method}.json", "w") as f:
        json.dump(output, f, indent=2)
    
    return results

if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "Vanilla"
    run_method(method)
