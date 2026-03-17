# -*- coding: utf-8 -*-
"""
MLGB 模型 Benchmark
===================
支持: DeepFM, DCN, EDCN, xDeepFM, AutoInt, NFM, WDL, FiBiNet, PNN, MaskNet, AFN, DLRM
"""

import os
import sys
import gc
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

# MLGB
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY = os.path.join(PROJECT_ROOT, 'third_party')
sys.path.insert(0, os.path.join(THIRD_PARTY, 'mlgb'))

from mlgb import get_model
from mlgb.utils import get_sparse_feature_name_dict

# 飞书通知
sys.path.append('/mnt/workspace/walter.wan/utils')
from feishu_notifier import FeishuNotifier


# ============================================================
# 配置
# ============================================================

DATA_PATH = '/mnt/data/oss_wanjun/pai_work/defer_sample_parquet/'
RESULT_DIR = '/tmp/ctr_benchmark'
os.makedirs(RESULT_DIR, exist_ok=True)

DEFAULT_CONFIG = {
    'train_days': 7,
    'test_days': 1,
    'sample_rate': 0.3,
    'epochs': 3,
    'batch_size': 2048,
    'embed_dim': 8,
    'learning_rate': 0.001,
    'high_cardinality_threshold': 50,
}

# 默认跑的模型
DEFAULT_MODELS = ['DeepFM', 'DCN', 'EDCN', 'AutoInt', 'FiBiNet', 'MaskNet', 'AFN', 'PNN', 'NFM', 'WDL']

# 核心特征
CORE_FEATURES = [
    'country', 'adx', 'bundle', 'make', 'model', 'os', 'osv',
    'business_type', 'campaignid', 'offerid', 'demand_pkgname',
    'carrier', 'city', 'connectiontype', 'devicetype',
    'duf_inner_dev_pkg_clk_7d', 'duf_inner_dev_pkg_imp_7d',
    'duf_inner_dev_adx_clk_7d', 'duf_inner_dev_adx_imp_7d',
    'duf_outer_dev_pkg_clk_7d', 'duf_outer_dev_pkg_imp_7d',
    'duf_outer_dev_adx_clk_7d', 'duf_outer_dev_adx_imp_7d',
    'duf_inner_dev_pkg_gmv_7d', 'duf_outer_dev_pkg_gmv_7d',
    'duf_inner_dev_pkg_clk_30d', 'duf_inner_dev_pkg_imp_30d',
    'duf_outer_dev_pkg_clk_30d', 'duf_outer_dev_pkg_imp_30d',
    'ruf2_dev_pkg_imp_1h', 'ruf2_dev_pkg_clk_1h',
    'huf_deviceid_demand_pkgname_imp_24h', 'huf_deviceid_demand_pkgname_clk_24h',
]


# ============================================================
# 工具函数
# ============================================================

def calc_pcoc(y_true, y_pred):
    pred_mean = np.mean(y_pred)
    true_mean = np.mean(y_true)
    return pred_mean / true_mean if true_mean > 0 else float('inf')


def evaluate(y_true, y_pred):
    return {
        'auc': roc_auc_score(y_true, y_pred),
        'logloss': log_loss(y_true, y_pred),
        'pcoc': calc_pcoc(y_true, y_pred),
    }


def load_data(train_days, test_days, sample_rate):
    date_dirs = sorted([d for d in os.listdir(DATA_PATH) if d.startswith('sample_date=')])
    
    total = train_days + test_days
    recent = date_dirs[-total:]
    train_dirs, test_dirs = recent[:train_days], recent[train_days:]
    
    print(f"训练: {train_dirs[0]} ~ {train_dirs[-1]} ({len(train_dirs)}天)")
    print(f"测试: {test_dirs[0]} ~ {test_dirs[-1]} ({len(test_dirs)}天)")
    
    train_df = pd.concat([pd.read_parquet(os.path.join(DATA_PATH, d)) for d in train_dirs], ignore_index=True)
    test_df = pd.concat([pd.read_parquet(os.path.join(DATA_PATH, d)) for d in test_dirs], ignore_index=True)
    
    if sample_rate < 1.0:
        train_df = train_df.sample(frac=sample_rate, random_state=42).reset_index(drop=True)
        test_df = test_df.sample(frac=sample_rate, random_state=42).reset_index(drop=True)
    
    print(f"样本: 训练 {len(train_df):,}, 测试 {len(test_df):,}")
    return train_df, test_df


def preprocess(train_df, test_df, features, threshold, embed_dim):
    """预处理并构建 mlgb feature_names 格式"""
    feats = [f for f in features if f in train_df.columns]
    print(f"有效特征: {len(feats)}")
    
    sparse_feature_dicts = []
    
    for col in feats:
        vc = train_df[col].value_counts()
        valid = set(vc[vc >= threshold].index)
        train_df[col] = train_df[col].where(train_df[col].isin(valid), 'OTHER')
        test_df[col] = test_df[col].where(test_df[col].isin(valid), 'OTHER')
    
    for col in feats:
        le = LabelEncoder()
        all_vals = pd.concat([train_df[col].astype(str), test_df[col].astype(str)]).unique()
        le.fit(all_vals)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        
        vocab_size = len(le.classes_) + 1
        feat_dict = get_sparse_feature_name_dict(
            feature_name=col,
            feature_nunique=vocab_size,
            input_length=1,
        )
        # 覆盖 embed_dim
        feat_dict['embed_dim'] = embed_dim
        sparse_feature_dicts.append(feat_dict)
    
    # mlgb feature_names 格式: (dense_tuple, sparse_tuple, seq_tuple)
    feature_names = (
        (),  # dense features (empty)
        tuple(sparse_feature_dicts),  # sparse features
        (),  # sequence features (empty)
    )
    
    return train_df, test_df, feats, feature_names


# ============================================================
# MLGB 训练
# ============================================================

def train_one_model(model_name, feature_names, x_train, y_train, x_test, y_test, config, device):
    print(f"\n--- {model_name} ---")
    start = datetime.now()
    
    try:
        # 使用 mlgb 的 get_model 接口
        model = get_model(
            feature_names=feature_names,
            model_name=model_name,
            task='binary',
            aim='ranking',
            lang='torch',
            device=device,
            seed=42,
            model_l1=0.0,
            model_l2=1e-6,
        )
        
        # mlgb 需要先 forward 一次初始化参数
        # 输入格式: (dense_array, sparse_array, seq_array)
        init_x = (
            np.array([]).reshape(32, 0).astype(np.float32),  # dense
            x_train[1][:32],  # sparse
            np.array([]).reshape(32, 0, 0).astype(np.int32),  # seq
        )
        with torch.no_grad():
            _ = model(init_x)
        
        # 打印参数数量
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  参数量: {n_params:,}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        loss_fn = nn.BCELoss()
        
        batch_size = config['batch_size']
        n_samples = len(y_train)
        n_batches = n_samples // batch_size
        
        for epoch in range(config['epochs']):
            model.train()
            total_loss = 0
            
            indices = np.random.permutation(n_samples)
            for i in range(n_batches):
                batch_idx = indices[i*batch_size:(i+1)*batch_size]
                # mlgb 输入: (dense, sparse, seq)
                batch_x = (
                    np.array([]).reshape(len(batch_idx), 0).astype(np.float32),  # dense
                    x_train[1][batch_idx],  # sparse
                    np.array([]).reshape(len(batch_idx), 0, 0).astype(np.int32),  # seq
                )
                batch_y = torch.tensor(y_train[batch_idx], dtype=torch.float32, device=device)
                
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = loss_fn(outputs, batch_y) + model.l1l2_loss()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}")
        
        # 预测
        model.eval()
        y_pred = []
        n_test = len(y_test)
        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                end_idx = min(i + batch_size, n_test)
                batch_x = (
                    np.array([]).reshape(end_idx - i, 0).astype(np.float32),
                    x_test[1][i:end_idx],
                    np.array([]).reshape(end_idx - i, 0, 0).astype(np.int32),
                )
                outputs = model(batch_x).squeeze()
                if outputs.dim() == 0:
                    y_pred.append(outputs.cpu().item())
                else:
                    y_pred.extend(outputs.cpu().numpy())
        y_pred = np.array(y_pred)
        
        metrics = evaluate(y_test, y_pred)
        elapsed = (datetime.now() - start).total_seconds()
        
        result = {'model': f"mlgb-{model_name}", **metrics, 'time_sec': elapsed, 'status': 'success'}
        print(f"  AUC={metrics['auc']:.4f}, PCOC={metrics['pcoc']:.3f}, {elapsed:.0f}s")
        
        del model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return {'model': f"mlgb-{model_name}", 'status': 'failed', 'error': str(e)}


def train_mlgb(train_df, test_df, feats, feature_names, config, models=None):
    models = models or DEFAULT_MODELS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    print(f"模型: {models}")
    
    # 构建输入 (mlgb 格式: tuple of arrays)
    # x = (dense_array, sparse_array, seq_array)
    sparse_train = train_df[feats].values.astype(np.int32)
    sparse_test = test_df[feats].values.astype(np.int32)
    
    x_train = (
        np.array([]).reshape(len(train_df), 0).astype(np.float32),  # dense
        sparse_train,  # sparse
        np.array([]).reshape(len(train_df), 0, 0).astype(np.int32),  # seq
    )
    x_test = (
        np.array([]).reshape(len(test_df), 0).astype(np.float32),
        sparse_test,
        np.array([]).reshape(len(test_df), 0, 0).astype(np.int32),
    )
    y_train = train_df['label'].values.astype(np.int32)
    y_test = test_df['label'].values.astype(np.int32)
    
    results = []
    for model_name in models:
        result = train_one_model(model_name, feature_names, x_train, y_train, x_test, y_test, config, device)
        results.append(result)
    
    return results


# ============================================================
# 主函数
# ============================================================

def run(config=None, models=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    start = datetime.now()
    
    print("=" * 50)
    print("MLGB Benchmark")
    print("=" * 50)
    
    # 加载数据
    train_df, test_df = load_data(config['train_days'], config['test_days'], config['sample_rate'])
    
    # 预处理
    train_df, test_df, feats, feature_names = preprocess(
        train_df.copy(), test_df.copy(), CORE_FEATURES, 
        config['high_cardinality_threshold'], config['embed_dim']
    )
    
    # 训练
    results = train_mlgb(train_df, test_df, feats, feature_names, config, models)
    
    # 排序
    success_results = sorted([r for r in results if r['status'] == 'success'], key=lambda x: -x['auc'])
    failed_results = [r for r in results if r['status'] == 'failed']
    
    elapsed = (datetime.now() - start).total_seconds()
    
    # 输出
    print("\n" + "=" * 50)
    print("结果 (按 AUC 排序)")
    print("=" * 50)
    for r in success_results:
        print(f"{r['model']:<20} AUC={r['auc']:.4f}, PCOC={r['pcoc']:.3f}, {r['time_sec']:.0f}s")
    if failed_results:
        print(f"\n失败: {[r['model'] for r in failed_results]}")
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")
    
    # 保存
    result_file = os.path.join(RESULT_DIR, f"mlgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w') as f:
        json.dump({'results': results, 'config': config, 'time_sec': elapsed}, f, indent=2)
    print(f"结果: {result_file}")
    
    return success_results


if __name__ == '__main__':
    results = run()
    
    # 飞书通知
    try:
        msg = "⚔️ MLGB Benchmark 完成\n\n"
        for r in results:
            msg += f"{r['model']}: AUC={r['auc']:.4f}, PCOC={r['pcoc']:.3f}\n"
        FeishuNotifier.notify(msg)
    except Exception as e:
        print(f"飞书通知失败: {e}")
