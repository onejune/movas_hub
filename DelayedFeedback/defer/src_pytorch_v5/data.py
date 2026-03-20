"""
Defer 数据加载 - PyTorch v2 版本

数据格式: parquet (对齐 TF v2)
- 249 个类别特征
- 多窗口标签 (24h, 48h, 72h, 168h)
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

# 时间常量
SECONDS_A_DAY = 60 * 60 * 24
SECONDS_AN_HOUR = 60 * 60

# 时间窗口（小时）
TIME_WINDOWS = [24, 48, 72, 168]


class FeatureEncoder:
    """类别特征编码器 - 与 TF v2 兼容"""
    def __init__(self):
        self.encoders = {}
        self.vocab_sizes = {}
    
    def fit(self, df, feature_cols):
        """拟合编码器"""
        for col in tqdm(feature_cols, desc="拟合编码器"):
            unique_vals = df[col].unique()
            self.encoders[col] = {v: i+1 for i, v in enumerate(unique_vals)}  # 0 留给 OOV
            self.vocab_sizes[col] = len(unique_vals) + 1
        return self
    
    def transform(self, df, feature_cols):
        """转换特征为 numpy 数组 [N, num_features]"""
        result = np.zeros((len(df), len(feature_cols)), dtype=np.int64)
        for i, col in enumerate(feature_cols):
            encoder = self.encoders.get(col, {})
            result[:, i] = df[col].map(lambda x: encoder.get(x, 0)).values
        return result
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'encoders': self.encoders, 'vocab_sizes': self.vocab_sizes}, f)
    
    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            # 转换 key 为原始类型
            self.encoders = {}
            for col, enc in data['encoders'].items():
                self.encoders[col] = {k: v for k, v in enc.items()}
            self.vocab_sizes = data['vocab_sizes']
        return self
    
    def get_vocab_sizes_list(self, feature_cols):
        """返回 vocab_sizes 列表，顺序与 feature_cols 一致"""
        return [self.vocab_sizes[col] for col in feature_cols]


class DeferDataset(Dataset):
    """
    Defer 数据集 - 适配 parquet 格式
    
    Args:
        features: [N, 249] 类别特征 (已编码)
        labels: dict or array, 标签数据
        method: 训练方法 (vanilla, oracle, winadapt)
    """
    def __init__(self, features, labels, method='oracle'):
        self.features = features
        self.labels = labels
        self.method = method
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        cate_feats = torch.tensor(self.features[idx], dtype=torch.long)
        
        if isinstance(self.labels, dict):
            # WinAdapt: 多窗口标签
            label = {k: torch.tensor(v[idx], dtype=torch.float32) for k, v in self.labels.items()}
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return {
            "cate_feats": cate_feats,
            "label": label,
        }


def load_data(data_dir, encoder_path=None):
    """
    加载 parquet 数据
    
    Args:
        data_dir: 数据目录
        encoder_path: 编码器路径 (可选，用于加载已有编码器)
    
    Returns:
        train_df, test_df, feature_cols, label_cols, encoder, meta
    """
    print("="*60)
    print("加载数据...")
    print("="*60)
    
    # 读取数据
    train_df = pq.read_table(os.path.join(data_dir, 'train.parquet')).to_pandas()
    test_df = pq.read_table(os.path.join(data_dir, 'test.parquet')).to_pandas()
    
    # 读取配置
    with open(os.path.join(data_dir, 'feature_cols.json')) as f:
        feature_cols = json.load(f)
    with open(os.path.join(data_dir, 'label_cols.json')) as f:
        label_cols = json.load(f)
    with open(os.path.join(data_dir, 'meta.json')) as f:
        meta = json.load(f)
    
    # 编码特征
    encoder = FeatureEncoder()
    if encoder_path and os.path.exists(encoder_path):
        print(f"加载已有编码器: {encoder_path}")
        encoder.load(encoder_path)
    else:
        print("拟合编码器...")
        encoder.fit(train_df, feature_cols)
        if encoder_path:
            os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
            encoder.save(encoder_path)
            print(f"编码器已保存: {encoder_path}")
    
    # 打印数据统计
    print(f"\n数据统计:")
    print(f"  训练集: {len(train_df):,} 行")
    print(f"  测试集: {len(test_df):,} 行")
    print(f"  特征数: {len(feature_cols)}")
    print(f"  时间窗口: {TIME_WINDOWS}")
    print(f"  训练集转化率: {meta['train_cvr']:.4f}")
    print(f"  测试集转化率: {meta['test_cvr']:.4f}")
    
    return train_df, test_df, feature_cols, label_cols, encoder, meta


def prepare_data(train_df, test_df, feature_cols, encoder, method='oracle', window=24):
    """
    准备训练和测试数据
    
    Args:
        train_df, test_df: DataFrame
        feature_cols: 特征列名列表
        encoder: FeatureEncoder
        method: 'vanilla', 'oracle', 'fnw', 'fnc', 'dfm', 'esdfm', 'winadapt'
        window: vanilla/fnw/fnc 方法使用的窗口 (小时)
    
    Returns:
        train_features, train_labels, test_features, test_labels
    """
    method = method.lower()
    print(f"\n准备数据 (method={method})...")
    
    # 转换特征
    train_features = encoder.transform(train_df, feature_cols)
    test_features = encoder.transform(test_df, feature_cols)
    
    # 测试标签始终使用 oracle
    test_labels = test_df['label_oracle'].values.astype(np.float32)
    
    if method == 'vanilla':
        # 使用指定窗口的标签
        label_col = f'label_{window}h'
        train_labels = train_df[label_col].values.astype(np.float32)
        print(f"  使用标签: {label_col}")
        
    elif method == 'oracle':
        # 使用最终真实标签
        train_labels = train_df['label_oracle'].values.astype(np.float32)
        print(f"  使用标签: label_oracle")
    
    elif method in ['fnw', 'fnc']:
        # FNW/FNC: 需要标签 + elapsed_time
        label_col = f'label_{window}h'
        train_labels = {
            'label': train_df[label_col].values.astype(np.float32),
            'elapsed_time': np.full(len(train_df), window, dtype=np.float32),  # 窗口时间
        }
        print(f"  使用标签: {label_col} + elapsed_time={window}h")
    
    elif method == 'dfm':
        # DFM: 需要标签 + delay_time
        # delay_time: 从点击到转化的时间（小时），负样本为 0
        if 'delay_time' in train_df.columns:
            delay_time = train_df['delay_time'].values.astype(np.float32)
        else:
            # 兼容旧数据：使用 diff_hours 或默认 0
            delay_time = np.zeros(len(train_df), dtype=np.float32)
            print("  警告: 数据中缺少 delay_time 列，使用默认值 0")
        
        train_labels = {
            'label': train_df['label_oracle'].values.astype(np.float32),
            'delay_time': delay_time,
        }
        print(f"  使用标签: label_oracle + delay_time (mean={delay_time[train_df['label_oracle']==1].mean():.1f}h)")
    
    elif method == 'esdfm':
        # ES-DFM: 需要 tn/dp/pos 标签
        # tn_label: 真负样本 (观察窗口结束后仍未转化)
        # dp_label: 延迟正样本 (窗口外转化)
        # pos_label: 窗口内正样本
        if 'dp_label' in train_df.columns:
            dp_label = train_df['dp_label'].values.astype(np.float32)
            tn_label = train_df['tn_label'].values.astype(np.float32)
            pos_label = train_df['pos_label'].values.astype(np.float32)
        else:
            # 兼容旧数据：从现有标签推导
            tn_label = (train_df['label_oracle'] == 0).values.astype(np.float32)
            dp_label = np.zeros(len(train_df), dtype=np.float32)  # 无法准确计算
            pos_label = train_df['label_24h'].values.astype(np.float32)
            print("  警告: 数据中缺少 dp_label 列，使用默认值 0")
        
        train_labels = {
            'label': train_df['label_oracle'].values.astype(np.float32),
            'tn_label': tn_label,
            'dp_label': dp_label,
            'pos_label': pos_label,
        }
        print(f"  使用标签: label_oracle + tn/dp/pos labels (dp_rate={dp_label.mean()*100:.2f}%)")
        
    elif method == 'winadapt':
        # 多窗口标签 + 可观察标记
        train_labels = {}
        for w in TIME_WINDOWS:
            train_labels[f'label_{w}h'] = train_df[f'label_{w}h'].values.astype(np.float32)
            train_labels[f'observable_{w}h'] = train_df[f'observable_{w}h'].values.astype(np.float32)
        print(f"  使用多窗口标签: {TIME_WINDOWS}")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return train_features, train_labels, test_features, test_labels


def create_dataloader(features, labels, batch_size, shuffle=True, num_workers=4, method='oracle'):
    """创建 DataLoader"""
    dataset = DeferDataset(features, labels, method=method)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    DATA_DIR = "/mnt/workspace/walter.wan/open_research/defer/data_v2"
    
    # 测试数据加载
    train_df, test_df, feature_cols, label_cols, encoder, meta = load_data(DATA_DIR)
    
    # 测试数据准备
    train_features, train_labels, test_features, test_labels = prepare_data(
        train_df, test_df, feature_cols, encoder, method='oracle'
    )
    
    print(f"\n特征形状: {train_features.shape}")
    print(f"标签形状: {train_labels.shape}")
    
    # 测试 DataLoader
    loader = create_dataloader(train_features, train_labels, batch_size=4096)
    batch = next(iter(loader))
    print(f"\nBatch cate_feats shape: {batch['cate_feats'].shape}")
    print(f"Batch label shape: {batch['label'].shape}")
