"""
MLGB Models Benchmark Example
=============================

Benchmark 33+ CTR ranking models from MLGB library.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctr_auto_hyperopt.data.preprocessing import DataProcessor
from ctr_auto_hyperopt.search.mlgb_search import MLGBSearch, quick_benchmark


def load_data(sample_frac=0.1):
    """Load CTR sample data."""
    DATA_PATH = '/mnt/data/oss_wanjun/pai_work/defer_sample_parquet/'
    
    date_dirs = sorted([d for d in os.listdir(DATA_PATH) if d.startswith('sample_date=')])
    
    # 2 days train, 1 day test
    train_df = pd.concat([
        pd.read_parquet(os.path.join(DATA_PATH, d))
        for d in date_dirs[:2]
    ], ignore_index=True)
    test_df = pd.read_parquet(os.path.join(DATA_PATH, date_dirs[2]))
    
    # Sample
    if sample_frac < 1.0:
        train_df = train_df.sample(frac=sample_frac, random_state=42)
        test_df = test_df.sample(frac=sample_frac, random_state=42)
    
    feature_cols = [
        'business_type', 'offerid', 'country', 'bundle',
        'adx', 'make', 'model', 'demand_pkgname', 'campaignid'
    ]
    
    return (
        train_df[feature_cols], train_df['label'].values,
        test_df[feature_cols], test_df['label'].values
    )


def main():
    print("="*60)
    print("MLGB Models Benchmark")
    print("="*60)
    print(f"Start: {datetime.now()}")
    
    # Load data
    print("\n[1] Loading data...")
    X_train, y_train, X_test, y_test = load_data(sample_frac=0.1)
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Preprocess
    print("\n[2] Preprocessing...")
    processor = DataProcessor()
    data = processor.fit_transform(
        X_train, y_train, X_test, y_test,
        sparse_features=list(X_train.columns),
        cross_features=[('country', 'adx'), ('make', 'model')],
        high_cardinality_threshold=50,
    )
    
    # Benchmark top models
    print("\n[3] Running benchmark...")
    top_models = [
        'MLP', 'DeepFM', 'DCN', 'XDeepFM', 'AutoInt',
        'FiBiNet', 'AFN', 'NFM', 'WDL', 'EDCN'
    ]
    
    searcher = MLGBSearch(
        models=top_models,
        time_budget=600,  # 10 min
        n_trials=50,
        n_epochs=3,
        verbose=1,
    )
    
    results, study = searcher.search(data)
    
    # Results
    print("\n" + "="*60)
    print("Final Leaderboard")
    print("="*60)
    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r['model_name']:15s} AUC: {r['score']:.6f}")
    
    print(f"\nBest: {results[0]['model_name']} (AUC={results[0]['score']:.6f})")
    print(f"End: {datetime.now()}")


if __name__ == '__main__':
    main()
