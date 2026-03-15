"""
DeepCTR Models Benchmark
========================
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

import torch
import optuna
from optuna.trial import Trial

# DeepCTR
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM, DCN, DCNMix, xDeepFM, AutoInt, NFM, AFM, WDL, FiBiNET

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data(sample_frac=0.1):
    """Load CTR data."""
    DATA_PATH = '/mnt/data/oss_wanjun/pai_work/defer_sample_parquet/'
    date_dirs = sorted([d for d in os.listdir(DATA_PATH) if d.startswith('sample_date=')])
    
    train_df = pd.concat([
        pd.read_parquet(os.path.join(DATA_PATH, d))
        for d in date_dirs[:2]
    ], ignore_index=True)
    test_df = pd.read_parquet(os.path.join(DATA_PATH, date_dirs[2]))
    
    if sample_frac < 1.0:
        train_df = train_df.sample(frac=sample_frac, random_state=42)
        test_df = test_df.sample(frac=sample_frac, random_state=42)
    
    return train_df, test_df


def preprocess(train_df, test_df, sparse_features, cross_features=None, threshold=50):
    """Preprocess data for DeepCTR."""
    # Cross features
    if cross_features:
        for col1, col2 in cross_features:
            cross_name = f'{col1}_x_{col2}'
            train_df[cross_name] = train_df[col1].astype(str) + '_' + train_df[col2].astype(str)
            test_df[cross_name] = test_df[col1].astype(str) + '_' + test_df[col2].astype(str)
            sparse_features.append(cross_name)
    
    # High cardinality handling
    for col in sparse_features:
        vc = train_df[col].value_counts()
        valid = set(vc[vc >= threshold].index)
        train_df[col] = train_df[col].apply(lambda x: x if x in valid else 'OTHER')
        test_df[col] = test_df[col].apply(lambda x: x if x in valid else 'OTHER')
    
    # Label encode
    encoders = {}
    for col in sparse_features:
        le = LabelEncoder()
        all_vals = pd.concat([train_df[col], test_df[col]]).astype(str).unique()
        le.fit(all_vals)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        encoders[col] = le
    
    # Feature columns
    feature_columns = [
        SparseFeat(col, vocabulary_size=int(train_df[col].max()) + 1, embedding_dim=16)
        for col in sparse_features
    ]
    
    # Inputs
    train_input = {col: train_df[col].values for col in sparse_features}
    test_input = {col: test_df[col].values for col in sparse_features}
    
    return feature_columns, train_input, train_df['label'].values, test_input, test_df['label'].values


def run_benchmark():
    print("="*60)
    print("DeepCTR Models Benchmark")
    print("="*60)
    print(f"Start: {datetime.now()}")
    
    # Load data
    print("\n[1] Loading data...")
    train_df, test_df = load_data(sample_frac=0.1)
    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Preprocess
    print("\n[2] Preprocessing...")
    sparse_features = ['business_type', 'offerid', 'country', 'bundle', 'adx', 'make', 'model', 'demand_pkgname', 'campaignid']
    cross_features = [('country', 'adx'), ('make', 'model')]
    
    feature_columns, train_input, y_train, test_input, y_test = preprocess(
        train_df.copy(), test_df.copy(), sparse_features.copy(), cross_features, threshold=50
    )
    print(f"Features: {len(feature_columns)}")
    
    # Models to benchmark
    MODELS = {
        'DeepFM': DeepFM,
        'DCN': DCN,
        'xDeepFM': xDeepFM,
        'AutoInt': AutoInt,
        'NFM': NFM,
        'WDL': WDL,
        'FiBiNET': FiBiNET,
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Benchmark each model
    print("\n[3] Running benchmark...")
    results = []
    
    for model_name, model_class in MODELS.items():
        print(f"\n  Training {model_name}...")
        
        try:
            # Model-specific params
            kwargs = {
                'linear_feature_columns': feature_columns,
                'dnn_feature_columns': feature_columns,
                'dnn_hidden_units': (256, 128, 64),
                'dnn_dropout': 0.2,
                'device': device,
            }
            
            if model_name == 'xDeepFM':
                kwargs['cin_layer_size'] = (128, 128)
            elif model_name == 'AutoInt':
                kwargs['att_layer_num'] = 3
                kwargs['att_head_num'] = 2
            elif model_name == 'DCN':
                kwargs['cross_num'] = 3
            elif model_name == 'FiBiNET':
                kwargs['bilinear_type'] = 'interaction'
            elif model_name == 'AFM':
                kwargs.pop('dnn_hidden_units')
                kwargs.pop('dnn_dropout')
                kwargs['attention_factor'] = 8
            
            model = model_class(**kwargs)
            model.compile(
                optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
                loss='binary_crossentropy',
                metrics=['binary_crossentropy', 'auc']
            )
            
            model.fit(
                train_input, y_train,
                batch_size=1024,
                epochs=3,
                verbose=0,
                validation_data=(test_input, y_test)
            )
            
            y_pred = model.predict(test_input, batch_size=1024)
            auc = roc_auc_score(y_test, y_pred)
            logloss = log_loss(y_test, y_pred)
            
            results.append({
                'model': model_name,
                'auc': auc,
                'logloss': logloss,
            })
            print(f"    AUC: {auc:.6f}, LogLoss: {logloss:.6f}")
            
        except Exception as e:
            print(f"    Failed: {e}")
            results.append({
                'model': model_name,
                'auc': 0.5,
                'logloss': 1.0,
            })
    
    # Sort by AUC
    results = sorted(results, key=lambda x: -x['auc'])
    
    print("\n" + "="*60)
    print("Final Leaderboard")
    print("="*60)
    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r['model']:15s} AUC: {r['auc']:.6f}  LogLoss: {r['logloss']:.6f}")
    
    print(f"\nBest: {results[0]['model']} (AUC={results[0]['auc']:.6f})")
    print(f"End: {datetime.now()}")
    
    return results


if __name__ == '__main__':
    run_benchmark()
