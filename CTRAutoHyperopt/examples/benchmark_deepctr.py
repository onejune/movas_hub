# -*- coding: utf-8 -*-
"""
DeepCTR 模型 Benchmark
======================
支持: DeepFM, DCN, xDeepFM, AutoInt, NFM, WDL, FiBiNET, PNN
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

# DeepCTR
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY = os.path.join(PROJECT_ROOT, 'third_party')
sys.path.insert(0, os.path.join(THIRD_PARTY, 'DeepCTR-Torch'))

from deepctr_torch.inputs import SparseFeat
from deepctr_torch.models import DeepFM, DCN, xDeepFM, AutoInt, NFM, WDL, FiBiNET, PNN

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
    'dnn_hidden_units': (256, 128),
    'high_cardinality_threshold': 50,
}

# 模型列表
DEEPCTR_MODELS = {
    'DeepFM': DeepFM,
    'DCN': DCN,
    'xDeepFM': xDeepFM,
    'AutoInt': AutoInt,
    'NFM': NFM,
    'WDL': WDL,
    'FiBiNET': FiBiNET,
    'PNN': PNN,
}

# 默认跑的模型（xDeepFM 太慢，默认跳过）
DEFAULT_MODELS = ['DeepFM', 'DCN', 'AutoInt', 'WDL', 'FiBiNET']

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


def preprocess(train_df, test_df, features, threshold):
    feats = [f for f in features if f in train_df.columns]
    print(f"有效特征: {len(feats)}")
    
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
    
    return train_df, test_df, feats


# ============================================================
# DeepCTR 训练
# ============================================================

def create_model(model_name, feature_columns, config, device):
    model_class = DEEPCTR_MODELS[model_name]
    dnn = config['dnn_hidden_units']
    
    if model_name == 'xDeepFM':
        return model_class(feature_columns, feature_columns, cin_layer_size=(64, 64), dnn_hidden_units=dnn, device=device)
    elif model_name == 'AutoInt':
        return model_class(feature_columns, feature_columns, att_layer_num=2, att_head_num=2, dnn_hidden_units=dnn, device=device)
    elif model_name == 'FiBiNET':
        return model_class(feature_columns, feature_columns, bilinear_type='interaction', dnn_hidden_units=dnn, device=device)
    elif model_name == 'PNN':
        return model_class(feature_columns, feature_columns, dnn_hidden_units=dnn, use_inner=True, use_outter=False, device=device)
    else:
        return model_class(feature_columns, feature_columns, dnn_hidden_units=dnn, device=device)


def train_one_model(model_name, feature_columns, train_input, y_train, test_input, y_test, config, device):
    print(f"\n--- {model_name} ---")
    start = datetime.now()
    
    try:
        model = create_model(model_name, feature_columns, config, device)
        model.compile('adam', 'binary_crossentropy', metrics=['auc'])
        model.fit(train_input, y_train, batch_size=config['batch_size'], epochs=config['epochs'], verbose=1)
        
        y_pred = model.predict(test_input, batch_size=config['batch_size']).flatten()
        metrics = evaluate(y_test, y_pred)
        elapsed = (datetime.now() - start).total_seconds()
        
        result = {'model': model_name, **metrics, 'time_sec': elapsed, 'status': 'success'}
        print(f"AUC={metrics['auc']:.4f}, PCOC={metrics['pcoc']:.3f}, {elapsed:.0f}s")
        
        del model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return {'model': model_name, 'status': 'failed', 'error': str(e)}


def train_deepctr(train_df, test_df, feats, config, models=None):
    models = models or DEFAULT_MODELS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    print(f"模型: {models}")
    
    # 构建特征列
    feature_columns = []
    for col in feats:
        vocab_size = int(max(train_df[col].max(), test_df[col].max())) + 2
        feature_columns.append(SparseFeat(col, vocabulary_size=vocab_size, embedding_dim=config['embed_dim']))
    
    train_input = {col: train_df[col].values for col in feats}
    test_input = {col: test_df[col].values for col in feats}
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    results = []
    for model_name in models:
        if model_name not in DEEPCTR_MODELS:
            print(f"跳过未知模型: {model_name}")
            continue
        result = train_one_model(model_name, feature_columns, train_input, y_train, test_input, y_test, config, device)
        results.append(result)
    
    return results


# ============================================================
# 主函数
# ============================================================

def run(config=None, models=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    start = datetime.now()
    
    print("=" * 50)
    print("DeepCTR Benchmark")
    print("=" * 50)
    
    # 加载数据
    train_df, test_df = load_data(config['train_days'], config['test_days'], config['sample_rate'])
    
    # 预处理
    train_df, test_df, feats = preprocess(
        train_df.copy(), test_df.copy(), CORE_FEATURES, config['high_cardinality_threshold']
    )
    
    # 训练
    results = train_deepctr(train_df, test_df, feats, config, models)
    
    # 排序
    results = sorted([r for r in results if r['status'] == 'success'], key=lambda x: -x['auc'])
    
    elapsed = (datetime.now() - start).total_seconds()
    
    # 输出
    print("\n" + "=" * 50)
    print("结果 (按 AUC 排序)")
    print("=" * 50)
    for r in results:
        print(f"{r['model']:<15} AUC={r['auc']:.4f}, PCOC={r['pcoc']:.3f}, {r['time_sec']:.0f}s")
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")
    
    # 保存
    result_file = os.path.join(RESULT_DIR, f"deepctr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w') as f:
        json.dump({'results': results, 'config': config, 'time_sec': elapsed}, f, indent=2)
    print(f"结果: {result_file}")
    
    return results


if __name__ == '__main__':
    results = run()
    
    # 飞书通知
    try:
        msg = "⚔️ DeepCTR Benchmark 完成\n\n"
        for r in results:
            msg += f"{r['model']}: AUC={r['auc']:.4f}, PCOC={r['pcoc']:.3f}\n"
        FeishuNotifier.notify(msg)
    except Exception as e:
        print(f"飞书通知失败: {e}")
