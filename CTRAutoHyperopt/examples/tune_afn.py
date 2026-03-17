# -*- coding: utf-8 -*-
"""
AFN 调参优化 (Optuna)
=====================
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
import optuna
from optuna.samplers import TPESampler

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
# 数据加载
# ============================================================

def calc_pcoc(y_true, y_pred):
    pred_mean = np.mean(y_pred)
    true_mean = np.mean(y_true)
    return pred_mean / true_mean if true_mean > 0 else float('inf')


def load_data(train_days=7, test_days=1, sample_rate=0.3):
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


def preprocess(train_df, test_df, features, threshold=50, embed_dim=8):
    feats = [f for f in features if f in train_df.columns]
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
        feat_dict['embed_dim'] = embed_dim
        sparse_feature_dicts.append(feat_dict)
    
    feature_names = ((), tuple(sparse_feature_dicts), ())
    return train_df, test_df, feats, feature_names


# ============================================================
# 训练函数
# ============================================================

def train_model(model_name, feature_names, x_train, y_train, x_test, y_test, 
                epochs=3, batch_size=2048, lr=0.001, l2=1e-6, device='cpu'):
    
    model = get_model(
        feature_names=feature_names,
        model_name=model_name,
        task='binary',
        aim='ranking',
        lang='torch',
        device=device,
        seed=42,
        model_l1=0.0,
        model_l2=l2,
    )
    
    init_x = (
        np.array([]).reshape(32, 0).astype(np.float32),
        x_train[1][:32],
        np.array([]).reshape(32, 0, 0).astype(np.int32),
    )
    with torch.no_grad():
        _ = model(init_x)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    
    n_samples = len(y_train)
    n_batches = n_samples // batch_size
    
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        for i in range(n_batches):
            batch_idx = indices[i*batch_size:(i+1)*batch_size]
            batch_x = (
                np.array([]).reshape(len(batch_idx), 0).astype(np.float32),
                x_train[1][batch_idx],
                np.array([]).reshape(len(batch_idx), 0, 0).astype(np.int32),
            )
            batch_y = torch.tensor(y_train[batch_idx], dtype=torch.float32, device=device)
            
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = loss_fn(outputs, batch_y) + model.l1l2_loss()
            loss.backward()
            optimizer.step()
    
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
    auc = roc_auc_score(y_test, y_pred)
    pcoc = calc_pcoc(y_test, y_pred)
    
    del model
    gc.collect()
    
    return auc, pcoc


# ============================================================
# Optuna 调参
# ============================================================

def create_objective(model_name, feature_names, x_train, y_train, x_test, y_test, device):
    
    def objective(trial):
        embed_dim = trial.suggest_categorical('embed_dim', [4, 8, 16, 32])
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        l2 = trial.suggest_float('l2', 1e-7, 1e-4, log=True)
        batch_size = trial.suggest_categorical('batch_size', [1024, 2048, 4096])
        epochs = trial.suggest_int('epochs', 3, 10)
        
        new_sparse_features = []
        for feat_dict in feature_names[1]:
            new_dict = feat_dict.copy()
            new_dict['embed_dim'] = embed_dim
            new_sparse_features.append(new_dict)
        new_feature_names = ((), tuple(new_sparse_features), ())
        
        try:
            auc, pcoc = train_model(
                model_name=model_name,
                feature_names=new_feature_names,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                l2=l2,
                device=device,
            )
            
            trial.set_user_attr('pcoc', pcoc)
            
            print(f"  Trial {trial.number}: AUC={auc:.4f}, PCOC={pcoc:.3f}, "
                  f"embed={embed_dim}, lr={lr:.1e}, l2={l2:.1e}, bs={batch_size}, ep={epochs}")
            
            return auc
            
        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            return 0.5
    
    return objective


def main():
    start = datetime.now()
    
    print("=" * 50)
    print("AFN 调参优化 (Optuna) - 50 trials")
    print("=" * 50)
    
    # 加载数据
    train_df, test_df = load_data(train_days=7, test_days=1, sample_rate=0.3)
    train_df, test_df, feats, feature_names = preprocess(
        train_df.copy(), test_df.copy(), CORE_FEATURES, threshold=50, embed_dim=8
    )
    
    sparse_train = train_df[feats].values.astype(np.int32)
    sparse_test = test_df[feats].values.astype(np.int32)
    
    x_train = (
        np.array([]).reshape(len(train_df), 0).astype(np.float32),
        sparse_train,
        np.array([]).reshape(len(train_df), 0, 0).astype(np.int32),
    )
    x_test = (
        np.array([]).reshape(len(test_df), 0).astype(np.float32),
        sparse_test,
        np.array([]).reshape(len(test_df), 0, 0).astype(np.int32),
    )
    y_train = train_df['label'].values.astype(np.int32)
    y_test = test_df['label'].values.astype(np.int32)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
    )
    
    objective = create_objective('AFN', feature_names, x_train, y_train, x_test, y_test, device)
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    
    best = study.best_trial
    elapsed = (datetime.now() - start).total_seconds()
    
    print(f"\n{'='*50}")
    print(f"AFN 最佳结果:")
    print(f"{'='*50}")
    print(f"  AUC: {best.value:.4f}")
    print(f"  PCOC: {best.user_attrs.get('pcoc', 'N/A')}")
    print(f"  参数: {best.params}")
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")
    
    result = {
        'model': 'AFN',
        'best_auc': best.value,
        'best_pcoc': best.user_attrs.get('pcoc'),
        'best_params': best.params,
        'n_trials': 50,
    }
    
    result_file = os.path.join(RESULT_DIR, f"tune_afn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"结果: {result_file}")
    
    # 飞书通知
    try:
        msg = f"⚔️ AFN 调参完成\n\n"
        msg += f"最佳 AUC: {best.value:.4f}\n"
        msg += f"最佳 PCOC: {best.user_attrs.get('pcoc'):.3f}\n"
        msg += f"参数: {best.params}\n"
        msg += f"耗时: {elapsed/60:.1f} 分钟"
        FeishuNotifier.notify(msg)
    except Exception as e:
        print(f"飞书通知失败: {e}")
    
    return result


if __name__ == '__main__':
    main()
