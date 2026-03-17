# -*- coding: utf-8 -*-
"""
CTR 模型对比实验
================
- 使用最近 30 天数据
- 每个模型只跑 1 epoch
- 支持特征配置扩展
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DeepCTR
THIRD_PARTY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'third_party')
sys.path.insert(0, os.path.join(THIRD_PARTY, 'DeepCTR-Torch'))

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM, DCN, xDeepFM, AutoInt, NFM, WDL, FiBiNET

from ctr_auto_hyperopt.config import FeatureConfig, DEFAULT_CONFIG


# ============================================================
# 数据加载
# ============================================================

def load_data(data_path: str, n_days: int = 30, sample_frac: float = 1.0):
    """加载最近 N 天数据
    
    Args:
        data_path: 数据目录
        n_days: 使用最近多少天
        sample_frac: 采样比例
    """
    date_dirs = sorted([d for d in os.listdir(data_path) if d.startswith('sample_date=')])
    
    # 取最近 n_days 天
    recent_dirs = date_dirs[-n_days:] if len(date_dirs) >= n_days else date_dirs
    
    # 按时间切分：前 n-1 天训练，最后 1 天测试
    train_dirs = recent_dirs[:-1]
    test_dirs = recent_dirs[-1:]
    
    print(f"训练集: {len(train_dirs)} 天 ({train_dirs[0]} ~ {train_dirs[-1]})")
    print(f"测试集: {len(test_dirs)} 天 ({test_dirs[0]})")
    
    train_df = pd.concat([
        pd.read_parquet(os.path.join(data_path, d))
        for d in train_dirs
    ], ignore_index=True)
    
    test_df = pd.concat([
        pd.read_parquet(os.path.join(data_path, d))
        for d in test_dirs
    ], ignore_index=True)
    
    if sample_frac < 1.0:
        train_df = train_df.sample(frac=sample_frac, random_state=42)
        test_df = test_df.sample(frac=sample_frac, random_state=42)
    
    return train_df, test_df


# ============================================================
# 特征处理
# ============================================================

def preprocess_features(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                        config: FeatureConfig):
    """根据配置预处理特征
    
    Args:
        train_df: 训练数据
        test_df: 测试数据
        config: 特征配置
    
    Returns:
        feature_columns, train_input, y_train, test_input, y_test
    """
    sparse_features = config.sparse_features.copy()
    
    # 创建交叉特征
    for col1, col2 in config.cross_features:
        cross_name = f'{col1}_x_{col2}'
        if col1 in train_df.columns and col2 in train_df.columns:
            train_df[cross_name] = train_df[col1].astype(str) + '_' + train_df[col2].astype(str)
            test_df[cross_name] = test_df[col1].astype(str) + '_' + test_df[col2].astype(str)
            sparse_features.append(cross_name)
    
    # 高基数特征处理
    threshold = config.high_cardinality_threshold
    for col in sparse_features:
        if col not in train_df.columns:
            continue
        vc = train_df[col].value_counts()
        valid = set(vc[vc >= threshold].index)
        train_df[col] = train_df[col].apply(lambda x: x if x in valid else 'OTHER')
        test_df[col] = test_df[col].apply(lambda x: x if x in valid else 'OTHER')
    
    # Label Encoding
    encoders = {}
    for col in sparse_features:
        if col not in train_df.columns:
            continue
        le = LabelEncoder()
        all_vals = pd.concat([train_df[col], test_df[col]]).astype(str).unique()
        le.fit(all_vals)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        encoders[col] = le
    
    # 过滤有效特征
    valid_sparse = [c for c in sparse_features if c in train_df.columns]
    valid_dense = [c for c in config.dense_features if c in train_df.columns]
    
    # 构建特征列
    feature_columns = []
    for col in valid_sparse:
        vocab_size = int(train_df[col].max()) + 1
        feature_columns.append(SparseFeat(col, vocabulary_size=vocab_size, embedding_dim=config.embed_dim))
    
    for col in valid_dense:
        feature_columns.append(DenseFeat(col, 1))
    
    # 构建输入
    all_cols = valid_sparse + valid_dense
    train_input = {col: train_df[col].values for col in all_cols}
    test_input = {col: test_df[col].values for col in all_cols}
    
    y_train = train_df[config.label_col].values
    y_test = test_df[config.label_col].values
    
    print(f"特征数: {len(feature_columns)} (sparse: {len(valid_sparse)}, dense: {len(valid_dense)})")
    
    return feature_columns, train_input, y_train, test_input, y_test


# ============================================================
# 模型训练
# ============================================================

MODELS = {
    'DeepFM': DeepFM,
    'DCN': DCN,
    'xDeepFM': xDeepFM,
    'AutoInt': AutoInt,
    'NFM': NFM,
    'WDL': WDL,
    'FiBiNET': FiBiNET,
}


def train_and_evaluate(model_name: str, model_class, feature_columns,
                       train_input, y_train, test_input, y_test,
                       epochs: int = 1, batch_size: int = 1024,
                       device: str = 'cpu'):
    """训练并评估单个模型"""
    
    # 模型特定参数
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
    
    model = model_class(**kwargs)
    model.compile(
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        loss='binary_crossentropy',
        metrics=['binary_crossentropy', 'auc']
    )
    
    model.fit(
        train_input, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(test_input, y_test)
    )
    
    y_pred = model.predict(test_input, batch_size=batch_size)
    auc = roc_auc_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred)
    
    return auc, logloss


# ============================================================
# 主函数
# ============================================================

def run_comparison(data_path: str = '/mnt/data/oss_wanjun/pai_work/defer_sample_parquet/',
                   n_days: int = 30,
                   sample_frac: float = 1.0,
                   epochs: int = 1,
                   config: FeatureConfig = None):
    """运行模型对比实验"""
    
    if config is None:
        config = DEFAULT_CONFIG
    
    print("=" * 60)
    print("CTR 模型对比实验")
    print("=" * 60)
    print(f"开始时间: {datetime.now()}")
    print(f"数据天数: {n_days}")
    print(f"采样比例: {sample_frac}")
    print(f"训练轮数: {epochs}")
    print()
    
    # 加载数据
    print("[1] 加载数据...")
    train_df, test_df = load_data(data_path, n_days, sample_frac)
    print(f"训练集: {len(train_df):,}, 测试集: {len(test_df):,}")
    print()
    
    # 特征处理
    print("[2] 特征处理...")
    print(f"稀疏特征: {config.sparse_features}")
    print(f"交叉特征: {config.cross_features}")
    feature_columns, train_input, y_train, test_input, y_test = preprocess_features(
        train_df.copy(), test_df.copy(), config
    )
    print()
    
    # 训练模型
    print("[3] 训练模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    print()
    
    results = []
    for model_name, model_class in MODELS.items():
        print(f"--- {model_name} ---")
        try:
            auc, logloss = train_and_evaluate(
                model_name, model_class, feature_columns,
                train_input, y_train, test_input, y_test,
                epochs=epochs, device=device
            )
            results.append({
                'model': model_name,
                'auc': auc,
                'logloss': logloss,
            })
            print(f"AUC: {auc:.6f}, LogLoss: {logloss:.6f}")
        except Exception as e:
            print(f"Failed: {e}")
            results.append({
                'model': model_name,
                'auc': 0.5,
                'logloss': 1.0,
            })
        print()
    
    # 排序输出
    results = sorted(results, key=lambda x: -x['auc'])
    
    print("=" * 60)
    print("最终排名")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r['model']:15s} AUC: {r['auc']:.6f}  LogLoss: {r['logloss']:.6f}")
    
    print()
    print(f"最优模型: {results[0]['model']} (AUC={results[0]['auc']:.6f})")
    print(f"结束时间: {datetime.now()}")
    
    return results


if __name__ == '__main__':
    # 使用默认配置
    # run_comparison(n_days=30, epochs=1)
    
    # 或者自定义配置
    config = FeatureConfig(
        sparse_features=['business_type', 'offerid', 'country', 'bundle', 'adx', 'make', 'model', 'demand_pkgname', 'campaignid'],
        cross_features=[('country', 'adx'), ('make', 'model')],
        high_cardinality_threshold=50,
        embed_dim=16,
    )
    
    run_comparison(
        n_days=30,
        sample_frac=1.0,  # 全量数据
        epochs=1,         # 只跑 1 epoch
        config=config,
    )
