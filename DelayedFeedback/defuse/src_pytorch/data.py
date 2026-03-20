#!/usr/bin/env python3
"""
DEFUSE Data Loading and Preprocessing
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class CategoryEncoder:
    """Simple category encoder for sparse features"""
    
    def __init__(self):
        self.vocab = {}
        self.vocab_size = {}
        
    def fit(self, df, cols):
        """Fit encoder on dataframe columns"""
        for col in tqdm(cols, desc="Fitting encoder"):
            unique_vals = df[col].unique()
            self.vocab[col] = {v: i+1 for i, v in enumerate(unique_vals)}  # 0 for unknown
            self.vocab_size[col] = len(unique_vals) + 1
        return self
    
    def transform(self, df, cols):
        """Transform dataframe columns to indices"""
        result = {}
        for col in cols:
            if col in self.vocab:
                result[col] = df[col].map(lambda x: self.vocab[col].get(x, 0)).values
            else:
                result[col] = np.zeros(len(df), dtype=np.int64)
        return result
    
    def save(self, path):
        """Save encoder to JSON file"""
        with open(path, 'w') as f:
            json.dump({'vocab': self.vocab, 'vocab_size': self.vocab_size}, f)
    
    def load(self, path):
        """Load encoder from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.vocab_size = data['vocab_size']
        return self


def load_parquet_data(data_dir, encoder_path=None):
    """
    Load and prepare data from parquet files
    
    Args:
        data_dir: Directory containing train.parquet, test.parquet, feature_cols.json
        encoder_path: Optional path to save/load encoder
        
    Returns:
        train_x, train_labels, test_x, test_labels, encoder, feature_cols
    """
    print("Loading data...")
    train_df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))
    
    with open(os.path.join(data_dir, 'feature_cols.json'), 'r') as f:
        feature_cols = json.load(f)
    
    print(f"Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")
    print(f"Features: {len(feature_cols)}")
    
    # Encoder
    encoder = CategoryEncoder()
    if encoder_path and os.path.exists(encoder_path):
        print(f"Loading encoder from {encoder_path}")
        encoder.load(encoder_path)
    else:
        print("Fitting encoder...")
        encoder.fit(train_df, feature_cols)
        if encoder_path:
            encoder.save(encoder_path)
            print(f"Encoder saved to {encoder_path}")
    
    # Transform features
    print("Transforming features...")
    train_x = encoder.transform(train_df, feature_cols)
    test_x = encoder.transform(test_df, feature_cols)
    
    # Labels
    train_labels = {
        'label': train_df['pos_label'].values,  # observed label (window positive)
        'label_oracle': train_df['label_oracle'].values,
        'tn_label': train_df['tn_label'].values,
        'dp_label': train_df['dp_label'].values,
        'pos_label': train_df['pos_label'].values,
    }
    
    test_labels = {
        'label': test_df['label_oracle'].values,  # use oracle for evaluation
        'label_oracle': test_df['label_oracle'].values,
    }
    
    return train_x, train_labels, test_x, test_labels, encoder, feature_cols


def create_dataloader(x_dict, labels, batch_size, shuffle=True, num_workers=4):
    """
    Create PyTorch DataLoader from feature dict and labels
    
    Args:
        x_dict: Dict of feature name -> numpy array
        labels: Dict of label name -> numpy array
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        
    Returns:
        DataLoader, list of keys (for unpacking batches)
    """
    tensors = {}
    for k, v in x_dict.items():
        tensors[k] = torch.LongTensor(v)
    for k, v in labels.items():
        tensors[k] = torch.FloatTensor(v)
    
    keys = list(tensors.keys())
    dataset = TensorDataset(*[tensors[k] for k in keys])
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=True
    )
    return loader, keys


def batch_to_dict(batch, keys, device):
    """
    Convert batch tuple to dict and move to device
    
    Args:
        batch: Tuple from DataLoader
        keys: List of keys matching batch elements
        device: Target device
        
    Returns:
        Dict of tensors on device
    """
    return {k: v.to(device) for k, v in zip(keys, batch)}


def split_features_labels(batch_dict, label_keys=None):
    """
    Split batch dict into features and labels
    
    Args:
        batch_dict: Dict from batch_to_dict
        label_keys: List of label key names
        
    Returns:
        x_dict (features), targets (labels)
    """
    if label_keys is None:
        label_keys = ['label', 'label_oracle', 'tn_label', 'dp_label', 'pos_label']
    
    x_dict = {k: v for k, v in batch_dict.items() if k not in label_keys}
    targets = {k: v for k, v in batch_dict.items() if k in label_keys}
    
    return x_dict, targets
