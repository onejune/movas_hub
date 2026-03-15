"""
CTR 预估建模 - 深度学习模型对比
================================

对比多种深度学习模型:
1. MLP (多层感知机)
2. Wide & Deep
3. DeepFM
4. DCN (Deep & Cross Network)
"""

import os
import sys
import warnings
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# 配置
DATA_PATH = '/mnt/data/oss_wanjun/pai_work/defer_sample_parquet/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

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
# 数据加载
# ============================================================================

def load_and_preprocess(n_train_days=2, n_test_days=1, sample_frac=0.1):
    """加载并预处理数据"""
    date_dirs = sorted([d for d in os.listdir(DATA_PATH) if d.startswith('sample_date=')])
    
    train_df = pd.concat([
        pd.read_parquet(os.path.join(DATA_PATH, d)) 
        for d in date_dirs[:n_train_days]
    ], ignore_index=True)
    test_df = pd.read_parquet(os.path.join(DATA_PATH, date_dirs[n_train_days]))
    
    # 采样
    train_df = train_df.sample(frac=sample_frac, random_state=42)
    test_df = test_df.sample(frac=sample_frac, random_state=42)
    
    print(f"训练集: {len(train_df):,}, 测试集: {len(test_df):,}")
    
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
    
    # Label 编码 + 统计每列的类别数
    encoders = {}
    vocab_sizes = {}
    for col in X_train.columns:
        le = LabelEncoder()
        all_vals = pd.concat([X_train[col], X_test[col]]).astype(str).unique()
        le.fit(all_vals)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le
        vocab_sizes[col] = len(le.classes_)
    
    y_train = train_df['label'].values.astype(np.float32)
    y_test = test_df['label'].values.astype(np.float32)
    
    print(f"特征数: {len(X_train.columns)}")
    print(f"各特征类别数: {vocab_sizes}")
    
    return X_train, y_train, X_test, y_test, vocab_sizes


def create_dataloader(X, y, batch_size=1024, shuffle=True):
    """创建 DataLoader"""
    X_tensor = torch.LongTensor(X.values)
    y_tensor = torch.FloatTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================================
# 深度学习模型定义
# ============================================================================

class EmbeddingLayer(nn.Module):
    """通用 Embedding 层"""
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int = 16):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(size + 1, embed_dim, padding_idx=0)
            for col, size in vocab_sizes.items()
        })
        self.num_fields = len(vocab_sizes)
        self.embed_dim = embed_dim
        
    def forward(self, x):
        # x: (batch, num_fields)
        embeds = []
        for i, col in enumerate(self.embeddings.keys()):
            embeds.append(self.embeddings[col](x[:, i]))
        return torch.stack(embeds, dim=1)  # (batch, num_fields, embed_dim)


class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, vocab_sizes, embed_dim=16, hidden_dims=[256, 128, 64]):
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
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        embed = self.embedding(x)  # (batch, num_fields, embed_dim)
        embed = embed.view(embed.size(0), -1)  # (batch, num_fields * embed_dim)
        return torch.sigmoid(self.mlp(embed)).squeeze()


class WideAndDeep(nn.Module):
    """Wide & Deep 模型"""
    def __init__(self, vocab_sizes, embed_dim=16, hidden_dims=[256, 128, 64]):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_sizes, embed_dim)
        self.num_fields = len(vocab_sizes)
        
        # Wide 部分 - 线性模型
        self.wide = nn.Linear(self.num_fields * embed_dim, 1)
        
        # Deep 部分
        input_dim = self.num_fields * embed_dim
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*layers)
        
    def forward(self, x):
        embed = self.embedding(x)
        embed_flat = embed.view(embed.size(0), -1)
        
        wide_out = self.wide(embed_flat)
        deep_out = self.deep(embed_flat)
        
        return torch.sigmoid(wide_out + deep_out).squeeze()


class DeepFM(nn.Module):
    """DeepFM 模型"""
    def __init__(self, vocab_sizes, embed_dim=16, hidden_dims=[256, 128, 64]):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_sizes, embed_dim)
        self.num_fields = len(vocab_sizes)
        self.embed_dim = embed_dim
        
        # FM 一阶部分
        self.fm_first_order = nn.ModuleDict({
            col: nn.Embedding(size + 1, 1, padding_idx=0)
            for col, size in vocab_sizes.items()
        })
        
        # Deep 部分
        input_dim = self.num_fields * embed_dim
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*layers)
        
    def forward(self, x):
        # FM 一阶
        first_order = []
        for i, col in enumerate(self.fm_first_order.keys()):
            first_order.append(self.fm_first_order[col](x[:, i]))
        first_order = torch.cat(first_order, dim=1).sum(dim=1, keepdim=True)
        
        # FM 二阶 (通过 embedding)
        embed = self.embedding(x)  # (batch, num_fields, embed_dim)
        sum_square = torch.sum(embed, dim=1) ** 2  # (batch, embed_dim)
        square_sum = torch.sum(embed ** 2, dim=1)  # (batch, embed_dim)
        second_order = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)
        
        # Deep
        embed_flat = embed.view(embed.size(0), -1)
        deep_out = self.deep(embed_flat)
        
        return torch.sigmoid(first_order + second_order + deep_out).squeeze()


class DCN(nn.Module):
    """Deep & Cross Network"""
    def __init__(self, vocab_sizes, embed_dim=16, hidden_dims=[256, 128], num_cross_layers=3):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_sizes, embed_dim)
        self.num_fields = len(vocab_sizes)
        input_dim = self.num_fields * embed_dim
        
        # Cross Network
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_cross_layers)
        ])
        
        # Deep Network
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        self.deep = nn.Sequential(*layers)
        
        # 输出层
        self.output = nn.Linear(input_dim + hidden_dims[-1], 1)
        
    def forward(self, x):
        embed = self.embedding(x)
        x0 = embed.view(embed.size(0), -1)  # (batch, input_dim)
        
        # Cross Network
        xl = x0
        for layer in self.cross_layers:
            xl = x0 * layer(xl) + xl
        
        # Deep Network
        deep_out = self.deep(x0)
        
        # Concatenate and output
        concat = torch.cat([xl, deep_out], dim=1)
        return torch.sigmoid(self.output(concat)).squeeze()


# ============================================================================
# 训练与评估
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """训练模型"""
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    best_auc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                pred = model(X_batch).cpu().numpy()
                preds.extend(pred)
                labels.extend(y_batch.numpy())
        
        auc = roc_auc_score(labels, preds)
        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict().copy()
        
        print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, AUC={auc:.4f}")
    
    model.load_state_dict(best_state)
    return model, best_auc


def evaluate_model(model, test_loader):
    """评估模型"""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch).cpu().numpy()
            preds.extend(pred)
            labels.extend(y_batch.numpy())
    
    preds = np.array(preds)
    labels = np.array(labels)
    
    auc = roc_auc_score(labels, preds)
    ll = log_loss(labels, preds)
    pcoc = preds.mean() / labels.mean()
    
    return {'AUC': auc, 'LogLoss': ll, 'PCOC': pcoc}


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("="*70)
    print("CTR 深度学习模型对比")
    print("="*70)
    print(f"开始时间: {datetime.now()}")
    print(f"设备: {DEVICE}")
    print()
    
    # 加载数据
    X_train, y_train, X_test, y_test, vocab_sizes = load_and_preprocess(
        n_train_days=2, n_test_days=1, sample_frac=0.1
    )
    
    # 创建 DataLoader
    train_loader = create_dataloader(X_train, y_train, batch_size=1024, shuffle=True)
    test_loader = create_dataloader(X_test, y_test, batch_size=1024, shuffle=False)
    
    # 模型列表
    models_config = {
        'MLP': lambda: MLP(vocab_sizes, embed_dim=16, hidden_dims=[256, 128, 64]),
        'Wide&Deep': lambda: WideAndDeep(vocab_sizes, embed_dim=16, hidden_dims=[256, 128, 64]),
        'DeepFM': lambda: DeepFM(vocab_sizes, embed_dim=16, hidden_dims=[256, 128, 64]),
        'DCN': lambda: DCN(vocab_sizes, embed_dim=16, hidden_dims=[256, 128], num_cross_layers=3),
    }
    
    results = []
    
    for name, model_fn in models_config.items():
        print(f"\n{'='*50}")
        print(f"训练 {name}...")
        print(f"{'='*50}")
        
        model = model_fn()
        model, best_auc = train_model(model, train_loader, test_loader, epochs=10, lr=0.001)
        
        metrics = evaluate_model(model, test_loader)
        metrics['Model'] = name
        results.append(metrics)
        
        print(f"最终: AUC={metrics['AUC']:.4f}, LogLoss={metrics['LogLoss']:.4f}, PCOC={metrics['PCOC']:.4f}")
    
    # 汇总结果
    print("\n" + "="*70)
    print("深度学习模型对比结果")
    print("="*70)
    df = pd.DataFrame(results)[['Model', 'AUC', 'LogLoss', 'PCOC']].sort_values('AUC', ascending=False)
    print(df.to_string(index=False))
    
    print(f"\n完成时间: {datetime.now()}")
    
    return results


if __name__ == '__main__':
    results = main()
