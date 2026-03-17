# -*- coding: utf-8 -*-
"""
ML 模型 Benchmark (FLAML)
========================
支持: LightGBM, XGBoost, CatBoost
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

# 飞书通知
sys.path.append('/mnt/workspace/walter.wan/utils')
from feishu_notifier import FeishuNotifier


# ============================================================
# 配置
# ============================================================

DATA_PATH = '/mnt/data/oss_wanjun/pai_work/defer_sample_parquet/'
RESULT_DIR = '/tmp/ctr_benchmark'
os.makedirs(RESULT_DIR, exist_ok=True)

# 默认配置
DEFAULT_CONFIG = {
    'train_days': 7,
    'test_days': 1,
    'sample_rate': 0.3,
    'time_budget': 900,  # 15 分钟
    'models': ['lgbm', 'xgboost', 'catboost'],
    'high_cardinality_threshold': 50,
}

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
# ML 训练
# ============================================================

def train_flaml(X_train, y_train, X_test, y_test, time_budget, models):
    from flaml import AutoML
    
    print(f"FLAML 搜索: time_budget={time_budget}s, models={models}")
    
    automl = AutoML()
    automl.fit(
        X_train, y_train,
        task='classification',
        metric='roc_auc',
        time_budget=time_budget,
        estimator_list=models,
        verbose=1,
    )
    
    y_pred = automl.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_pred)
    
    result = {
        'model': f"FLAML-{automl.best_estimator}",
        'best_estimator': automl.best_estimator,
        'best_config': automl.best_config,
        **metrics,
    }
    
    print(f"最佳: {automl.best_estimator}, AUC={metrics['auc']:.4f}, PCOC={metrics['pcoc']:.3f}")
    
    del automl
    gc.collect()
    
    return result


# ============================================================
# 主函数
# ============================================================

def run(config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    start = datetime.now()
    
    print("=" * 50)
    print("ML Benchmark (FLAML)")
    print("=" * 50)
    print(f"配置: {json.dumps(config, indent=2)}")
    print()
    
    # 加载数据
    train_df, test_df = load_data(config['train_days'], config['test_days'], config['sample_rate'])
    
    # 预处理
    train_df, test_df, feats = preprocess(
        train_df.copy(), test_df.copy(), CORE_FEATURES, config['high_cardinality_threshold']
    )
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    X_train = train_df[feats]
    X_test = test_df[feats]
    
    # 训练
    result = train_flaml(X_train, y_train, X_test, y_test, config['time_budget'], config['models'])
    
    elapsed = (datetime.now() - start).total_seconds()
    result['time_sec'] = elapsed
    result['config'] = config
    
    # 保存结果
    result_file = os.path.join(RESULT_DIR, f"ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n结果: {result_file}")
    print(f"耗时: {elapsed/60:.1f} 分钟")
    
    return result


if __name__ == '__main__':
    result = run()
    
    # 飞书通知
    try:
        msg = f"""⚔️ ML Benchmark 完成

最佳模型: {result['model']}
AUC: {result['auc']:.4f}
LogLoss: {result['logloss']:.4f}
PCOC: {result['pcoc']:.3f}

耗时: {result['time_sec']/60:.1f} 分钟"""
        FeishuNotifier.notify(msg)
    except Exception as e:
        print(f"飞书通知失败: {e}")
