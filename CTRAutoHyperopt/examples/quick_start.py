"""
Quick Start Example for CTR Auto HyperOpt
==========================================

This example shows how to use AutoCTR for automatic model selection
and hyperparameter optimization on CTR prediction tasks.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctr_auto_hyperopt import AutoCTR


def load_sample_data():
    """Load sample CTR data."""
    DATA_PATH = '/mnt/data/oss_wanjun/pai_work/defer_sample_parquet/'
    
    date_dirs = sorted([d for d in os.listdir(DATA_PATH) if d.startswith('sample_date=')])
    
    # Load 2 days for training, 1 day for testing
    train_df = pd.concat([
        pd.read_parquet(os.path.join(DATA_PATH, d))
        for d in date_dirs[:2]
    ], ignore_index=True)
    
    test_df = pd.read_parquet(os.path.join(DATA_PATH, date_dirs[2]))
    
    # Sample for quick demo
    train_df = train_df.sample(frac=0.1, random_state=42)
    test_df = test_df.sample(frac=0.1, random_state=42)
    
    # Features (exclude time-related columns)
    feature_cols = [
        'business_type', 'offerid', 'country', 'bundle',
        'adx', 'make', 'model', 'demand_pkgname', 'campaignid'
    ]
    
    X_train = train_df[feature_cols]
    y_train = train_df['label'].values
    X_test = test_df[feature_cols]
    y_test = test_df['label'].values
    
    return X_train, y_train, X_test, y_test


def main():
    print("="*60)
    print("CTR Auto HyperOpt - Quick Start Example")
    print("="*60)
    
    # Load data
    print("\n[1] Loading data...")
    X_train, y_train, X_test, y_test = load_sample_data()
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Initialize AutoCTR
    print("\n[2] Initializing AutoCTR...")
    auto_ctr = AutoCTR(
        task='binary',
        metric='auc',
        time_budget=300,  # 5 minutes
        n_trials=20,
        include_models=['lgbm', 'xgboost', 'DeepFM', 'xDeepFM', 'DCN'],
        device='cpu',
        verbose=1,
    )
    
    # Fit
    print("\n[3] Searching for best model...")
    auto_ctr.fit(
        X_train, y_train,
        cross_features=[
            ('country', 'adx'),
            ('make', 'model'),
        ],
        high_cardinality_threshold=50,
    )
    
    # Results
    print("\n[4] Results:")
    print(f"Best Model: {auto_ctr.best_model_name}")
    print(f"Best AUC: {auto_ctr.best_score:.6f}")
    print(f"Best Params: {auto_ctr.best_params}")
    
    # Leaderboard
    print("\n[5] Leaderboard:")
    print(auto_ctr.get_leaderboard())
    
    # Predict
    print("\n[6] Predicting on test set...")
    y_pred = auto_ctr.predict_proba(X_test)
    
    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(y_test, y_pred)
    print(f"Test AUC: {test_auc:.6f}")


if __name__ == '__main__':
    main()
