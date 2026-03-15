"""
CTR 深度学习模型 - Optuna 超参寻优
==================================

使用 Optuna 对深度学习模型进行超参数自动搜索:
- embed_dim
- hidden_dims (层数和宽度)
- learning_rate
- dropout
- batch_size
- optimizer
"""

import os
import warnings
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

import optuna
from optuna.trial import Trial

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 配置
DATA_PATH = '/mnt/data/oss_wanjun/pai_work/defer_sample_parquet/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEATURE_COLS = [
    'business_type', 'offerid', 'country', 'bundle',
    'adx', 'make', 'model', 'demand_pkgname', 'campaignid'
]

CROSS_FEATURES = [
    ('country', 'adx'),
    ('country', 'business_type'),
    ('make', 'model'),
    ('adx', 'business_type'),
]


# ============================================================================
# 数据加载 (全局缓存)
# ============================================================================

_DATA_CACHE = {}

def load_data(n_train_days=2, n_test_days=1, sample_frac=0.1):
    """加载数据（带缓存）"""
    cache_key = (n_train_days, n_test_days, sample_frac)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]
    
    date_dirs = sorted([d for d in os.listdir(DATA_PATH) if d.startswith('sample_date=')])
    
    train_df = pd.concat([
        pd.read_parquet(os.path.join(DATA_PATH, d)) 
        for d in date_dirs[:n_train_days]
    ], ignore_index=True)
    test_df = pd.read_parquet(os.path.join(DATA_PATH, date_dirs[n_train_days]))
    
    train_df = train_df.sample(frac=sample_frac, random_state=42)
    test_df = test_df.sample(frac=sample_frac, random_state=42)
    
    # 特征工程
    X_train = train_df[FEATURE_COLS].copy()
    X_test = test_df[FEATURE_COLS].copy()
    
    for col1, col2 in CROSS_FEATURES:
        X_train[f'{col1}_x_{col2}'] = X_train[col1].astype(str) + '_' + X_train[col2].astype(str)
        X_test[f'{col1}_x_{col2}'] = X_test[col1].astype(str) + '_' + X_test[col2].astype(str)
    
    # 高基数处理
    for col in X_train.columns:
        vc = X_train[col].value_counts()
        valid = set(vc[vc >= 50].index)
        X_train[col] = X_train[col].apply(lambda x: x if x in valid else 'OTHER')
        X_test[col] = X_test[col].apply(lambda x: x if x in valid else 'OTHER')
    
    # Label 编码
    vocab_sizes = {}
    for col in X_train.columns:
        le = LabelEncoder()
        all_vals = pd.concat([X_train[col], X_test[col]]).astype(str).unique()
        le.fit(all_vals)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        vocab_sizes[col] = len(le.classes_)
    
    y_train = train_df['label'].values.astype(np.float32)
    y_test = test_df['label'].values.astype(np.float32)
    
    result = (X_train, y_train, X_test, y_test, vocab_sizes)
    _DATA_CACHE[cache_key] = result
    return result


def create_dataloader(X, y, batch_size=1024, shuffle=True):
    """创建 DataLoader"""
    X_tensor = torch.LongTensor(X.values)
    y_tensor = torch.FloatTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ============================================================================
# 模型定义
# ============================================================================

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(size + 1, embed_dim, padding_idx=0)
            for col, size in vocab_sizes.items()
        })
        
    def forward(self, x):
        embeds = []
        for i, col in enumerate(self.embeddings.keys()):
            embeds.append(self.embeddings[col](x[:, i]))
        return torch.stack(embeds, dim=1)


class FlexibleMLP(nn.Module):
    """可配置的 MLP"""
    def __init__(self, vocab_sizes, embed_dim, hidden_dims, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_sizes, embed_dim)
        input_dim = len(vocab_sizes) * embed_dim
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        embed = self.embedding(x)
        embed = embed.view(embed.size(0), -1)
        return torch.sigmoid(self.mlp(embed)).squeeze()


class FlexibleWideDeep(nn.Module):
    """可配置的 Wide & Deep"""
    def __init__(self, vocab_sizes, embed_dim, hidden_dims, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_sizes, embed_dim)
        input_dim = len(vocab_sizes) * embed_dim
        
        self.wide = nn.Linear(input_dim, 1)
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*layers)
        
    def forward(self, x):
        embed = self.embedding(x)
        embed_flat = embed.view(embed.size(0), -1)
        return torch.sigmoid(self.wide(embed_flat) + self.deep(embed_flat)).squeeze()


# ============================================================================
# Optuna 目标函数
# ============================================================================

def create_objective(model_type: str, X_train, y_train, X_test, y_test, vocab_sizes, n_epochs=5):
    """创建 Optuna 目标函数"""
    
    def objective(trial: Trial) -> float:
        # 超参数搜索空间
        embed_dim = trial.suggest_categorical('embed_dim', [8, 16, 32])
        n_layers = trial.suggest_int('n_layers', 1, 4)
        hidden_dims = [
            trial.suggest_categorical(f'hidden_dim_{i}', [64, 128, 256, 512])
            for i in range(n_layers)
        ]
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # 创建模型
        if model_type == 'MLP':
            model = FlexibleMLP(vocab_sizes, embed_dim, hidden_dims, dropout)
        elif model_type == 'WideDeep':
            model = FlexibleWideDeep(vocab_sizes, embed_dim, hidden_dims, dropout)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(DEVICE)
        
        # 创建 DataLoader
        train_loader = create_dataloader(X_train, y_train, batch_size, shuffle=True)
        test_loader = create_dataloader(X_test, y_test, batch_size, shuffle=False)
        
        # 优化器
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        criterion = nn.BCELoss()
        
        # 训练
        best_auc = 0
        for epoch in range(n_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
            
            # 验证
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(DEVICE)
                    pred = model(X_batch).cpu().numpy()
                    preds.extend(pred)
                    labels.extend(y_batch.numpy())
            
            auc = roc_auc_score(labels, preds)
            best_auc = max(best_auc, auc)
            
            # Optuna 剪枝
            trial.report(auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_auc
    
    return objective


# ============================================================================
# 主函数
# ============================================================================

def run_optuna_search(
    model_type: str,
    n_trials: int = 30,
    n_epochs: int = 5,
    timeout: int = 300
):
    """运行 Optuna 超参搜索"""
    print(f"\n{'='*60}")
    print(f"Optuna 超参搜索: {model_type}")
    print(f"{'='*60}")
    print(f"Trials: {n_trials}, Epochs: {n_epochs}, Timeout: {timeout}s")
    
    # 加载数据
    X_train, y_train, X_test, y_test, vocab_sizes = load_data()
    print(f"训练集: {len(X_train):,}, 测试集: {len(X_test):,}")
    
    # 创建 study
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
    )
    
    # 目标函数
    objective = create_objective(
        model_type, X_train, y_train, X_test, y_test, vocab_sizes, n_epochs
    )
    
    # 运行搜索
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    # 结果
    print(f"\n{'='*40}")
    print(f"搜索完成!")
    print(f"{'='*40}")
    print(f"最优 AUC: {study.best_value:.6f}")
    print(f"最优参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study


def main():
    print("="*70)
    print("CTR 深度学习 - Optuna 超参寻优")
    print("="*70)
    print(f"开始时间: {datetime.now()}")
    print(f"设备: {DEVICE}")
    
    # 加载数据（预热缓存）
    print("\n加载数据...")
    X_train, y_train, X_test, y_test, vocab_sizes = load_data()
    print(f"训练集: {len(X_train):,}, 测试集: {len(X_test):,}")
    print(f"特征数: {len(X_train.columns)}")
    
    results = {}
    
    # 搜索 MLP
    study_mlp = run_optuna_search(
        model_type='MLP',
        n_trials=20,
        n_epochs=5,
        timeout=300  # 5分钟
    )
    results['MLP'] = {
        'best_auc': study_mlp.best_value,
        'best_params': study_mlp.best_params
    }
    
    # 搜索 Wide & Deep
    study_wd = run_optuna_search(
        model_type='WideDeep',
        n_trials=20,
        n_epochs=5,
        timeout=300  # 5分钟
    )
    results['WideDeep'] = {
        'best_auc': study_wd.best_value,
        'best_params': study_wd.best_params
    }
    
    # 汇总
    print("\n" + "="*70)
    print("最终结果汇总")
    print("="*70)
    for model_name, res in results.items():
        print(f"\n{model_name}:")
        print(f"  最优 AUC: {res['best_auc']:.6f}")
        print(f"  最优参数: {res['best_params']}")
    
    print(f"\n完成时间: {datetime.now()}")
    
    return results


if __name__ == '__main__':
    results = main()
