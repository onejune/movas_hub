# -*- coding: utf-8 -*-
"""
CTR Benchmark V3
================
全量数据, 7+1天, FLAML(ML) + DeepCTR(DL)
按 business_type 分组评估

变更 vs V2:
- 全量数据 (sample_rate=1.0)
- 训练 7 天 + 测试 1 天
- 新增 DeepCTR 模型: DeepFM, DCN, xDeepFM, AutoInt, WDL, PNN(bug已修复)
- 新增 business_type 分组评估
- 完成后发飞书通知
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

# 飞书通知
sys.path.append('/mnt/workspace/walter.wan/utils')
from feishu_notifier import FeishuNotifier

# DeepCTR
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY = os.path.join(PROJECT_ROOT, 'third_party')
sys.path.insert(0, os.path.join(THIRD_PARTY, 'DeepCTR-Torch'))
from deepctr_torch.inputs import SparseFeat
from deepctr_torch.models import DeepFM, DCN, xDeepFM, AutoInt, WDL, PNN


# ============================================================
# 配置
# ============================================================

DATA_PATH = '/mnt/data/oss_wanjun/pai_work/defer_sample_parquet/'
RESULT_DIR = '/tmp/ctr_benchmark_v3'
os.makedirs(RESULT_DIR, exist_ok=True)

CONFIG = {
    'train_days': 7,
    'test_days': 1,
    'sample_rate': 1.0,          # 全量
    'flaml_time_budget': 1800,   # 30 分钟
    'dl_epochs': 3,
    'dl_batch_size': 4096,
    'embed_dim': 16,
    'dnn_hidden_units': (512, 256, 128),
    'high_cardinality_threshold': 50,
}

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

DL_MODELS = {
    'DeepFM': DeepFM,
    'DCN': DCN,
    'xDeepFM': xDeepFM,
    'AutoInt': AutoInt,
    'WDL': WDL,
    'PNN': PNN,
}


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
        'n': len(y_true),
    }


def evaluate_by_business_type(test_df, y_pred, label_col='label', bt_col='business_type'):
    """按 business_type 分组评估"""
    results = {}
    for bt, idx in test_df.groupby(bt_col).groups.items():
        yt = test_df.loc[idx, label_col].values
        yp = y_pred[idx]
        if len(yt) < 100 or yt.sum() < 5:
            continue
        try:
            results[bt] = evaluate(yt, yp)
        except Exception:
            pass
    return results


def load_data():
    date_dirs = sorted([d for d in os.listdir(DATA_PATH) if d.startswith('sample_date=')])
    total = CONFIG['train_days'] + CONFIG['test_days']
    recent = date_dirs[-total:]
    train_dirs = recent[:CONFIG['train_days']]
    test_dirs = recent[CONFIG['train_days']:]

    print(f"训练: {train_dirs[0]} ~ {train_dirs[-1]} ({len(train_dirs)}天)")
    print(f"测试: {test_dirs[0]} ~ {test_dirs[-1]} ({len(test_dirs)}天)")

    train_df = pd.concat(
        [pd.read_parquet(os.path.join(DATA_PATH, d)) for d in train_dirs],
        ignore_index=True
    )
    test_df = pd.concat(
        [pd.read_parquet(os.path.join(DATA_PATH, d)) for d in test_dirs],
        ignore_index=True
    )

    sr = CONFIG['sample_rate']
    if sr < 1.0:
        train_df = train_df.sample(frac=sr, random_state=42).reset_index(drop=True)
        test_df = test_df.sample(frac=sr, random_state=42).reset_index(drop=True)

    print(f"样本: 训练 {len(train_df):,}, 测试 {len(test_df):,}")
    return train_df, test_df


def preprocess(train_df, test_df):
    feats = [f for f in CORE_FEATURES if f in train_df.columns]
    print(f"有效特征: {len(feats)}")
    threshold = CONFIG['high_cardinality_threshold']

    for col in feats:
        vc = train_df[col].value_counts()
        valid = set(vc[vc >= threshold].index)
        train_df[col] = train_df[col].where(train_df[col].isin(valid), 'OTHER')
        test_df[col] = test_df[col].where(test_df[col].isin(valid), 'OTHER')

    encoders = {}
    for col in feats:
        le = LabelEncoder()
        all_vals = pd.concat([train_df[col].astype(str), test_df[col].astype(str)]).unique()
        le.fit(all_vals)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        encoders[col] = le

    return train_df, test_df, feats, encoders


# ============================================================
# Phase 1: FLAML ML
# ============================================================

def run_flaml(X_train, y_train, X_test, y_test, test_df_raw):
    from flaml import AutoML
    print(f"\nFLAML 搜索: time_budget={CONFIG['flaml_time_budget']}s")

    automl = AutoML()
    automl.fit(
        X_train, y_train,
        task='classification',
        metric='roc_auc',
        time_budget=CONFIG['flaml_time_budget'],
        estimator_list=['lgbm', 'xgboost', 'catboost'],
        n_jobs=1,   # 禁用多进程，避免子进程被 kill 导致父进程静默退出
        verbose=2,
    )

    y_pred = automl.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_pred)
    bt_metrics = evaluate_by_business_type(test_df_raw.reset_index(drop=True), y_pred)

    result = {
        'model': f"FLAML-{automl.best_estimator}",
        'best_estimator': automl.best_estimator,
        'best_config': automl.best_config,
        **metrics,
        'by_business_type': bt_metrics,
    }

    print(f"最佳: {automl.best_estimator}")
    print(f"整体: AUC={metrics['auc']:.4f}, PCOC={metrics['pcoc']:.3f}, LogLoss={metrics['logloss']:.4f}")
    print("分组:")
    for bt, m in sorted(bt_metrics.items()):
        print(f"  {bt}: AUC={m['auc']:.4f}, PCOC={m['pcoc']:.3f}, n={m['n']:,}")

    del automl
    gc.collect()
    return result


# ============================================================
# Phase 2: DeepCTR
# ============================================================

def build_feature_columns(train_df, test_df, feats):
    cols = []
    for col in feats:
        vocab_size = int(max(train_df[col].max(), test_df[col].max())) + 2
        cols.append(SparseFeat(col, vocabulary_size=vocab_size, embedding_dim=CONFIG['embed_dim']))
    return cols


def create_dl_model(model_name, feature_columns, device):
    dnn = CONFIG['dnn_hidden_units']
    if model_name == 'xDeepFM':
        return DL_MODELS[model_name](feature_columns, feature_columns,
                                     cin_layer_size=(128, 128), dnn_hidden_units=dnn, device=device)
    elif model_name == 'AutoInt':
        return DL_MODELS[model_name](feature_columns, feature_columns,
                                     att_layer_num=2, att_head_num=2, dnn_hidden_units=dnn, device=device)
    elif model_name == 'PNN':
        # PNN 只接受 dnn_feature_columns，不接受 linear_feature_columns
        return DL_MODELS[model_name](feature_columns, dnn_hidden_units=dnn,
                                     use_inner=True, use_outter=False, device=device)
    else:
        return DL_MODELS[model_name](feature_columns, feature_columns,
                                     dnn_hidden_units=dnn, device=device)


def run_deepctr(train_df, test_df, feats, test_df_raw):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDeepCTR 设备: {device}")

    feature_columns = build_feature_columns(train_df, test_df, feats)
    train_input = {col: train_df[col].values for col in feats}
    test_input = {col: test_df[col].values for col in feats}
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    results = []
    for model_name in DL_MODELS:
        print(f"\n--- {model_name} ---")
        t0 = datetime.now()
        try:
            model = create_dl_model(model_name, feature_columns, device)
            model.compile('adam', 'binary_crossentropy', metrics=['auc'])
            model.fit(train_input, y_train,
                      batch_size=CONFIG['dl_batch_size'],
                      epochs=CONFIG['dl_epochs'], verbose=1)

            y_pred = model.predict(test_input, batch_size=CONFIG['dl_batch_size']).flatten()
            metrics = evaluate(y_test, y_pred)
            bt_metrics = evaluate_by_business_type(test_df_raw.reset_index(drop=True), y_pred)
            elapsed = (datetime.now() - t0).total_seconds()

            result = {
                'model': model_name,
                **metrics,
                'time_sec': elapsed,
                'status': 'success',
                'by_business_type': bt_metrics,
            }
            print(f"AUC={metrics['auc']:.4f}, PCOC={metrics['pcoc']:.3f}, {elapsed:.0f}s")
            for bt, m in sorted(bt_metrics.items()):
                print(f"  {bt}: AUC={m['auc']:.4f}, PCOC={m['pcoc']:.3f}, n={m['n']:,}")

            del model
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"Failed: {e}")
            traceback.print_exc()
            result = {'model': model_name, 'status': 'failed', 'error': str(e), 'auc': 0}

        results.append(result)

    return results


# ============================================================
# 主函数
# ============================================================

def main():
    start = datetime.now()
    print("=" * 60)
    print("CTR Benchmark V3 — 全量 7+1天 ML+DL")
    print("=" * 60)
    print(f"开始: {start}")
    print(f"配置: {json.dumps({k: v for k, v in CONFIG.items() if k != 'dnn_hidden_units'}, indent=2)}")

    # 加载数据
    train_df, test_df = load_data()
    test_df_raw = test_df.copy()  # 保留原始列（含 business_type）

    # 预处理
    train_df, test_df, feats, encoders = preprocess(train_df.copy(), test_df.copy())

    y_train = train_df['label'].values
    y_test = test_df['label'].values
    X_train = train_df[feats]
    X_test = test_df[feats]

    all_results = []

    # Phase 1: FLAML
    print("\n" + "=" * 60)
    print("Phase 1: FLAML (ML)")
    print("=" * 60)
    try:
        ml_result = run_flaml(X_train, y_train, X_test, y_test, test_df_raw)
        ml_result['status'] = 'success'
        all_results.append(ml_result)
    except Exception as e:
        print(f"FLAML 失败: {e}")

    # Phase 2: DeepCTR
    print("\n" + "=" * 60)
    print("Phase 2: DeepCTR (DL)")
    print("=" * 60)
    try:
        dl_results = run_deepctr(train_df, test_df, feats, test_df_raw)
        all_results.extend(dl_results)
    except Exception as e:
        print(f"DeepCTR 失败: {e}")

    # 汇总
    elapsed = (datetime.now() - start).total_seconds()
    success = [r for r in all_results if r.get('status') == 'success']
    success = sorted(success, key=lambda x: -x.get('auc', 0))

    print("\n" + "=" * 60)
    print("最终结果 (按 AUC 排序)")
    print("=" * 60)
    print(f"{'模型':<20} {'AUC':>8} {'LogLoss':>10} {'PCOC':>8} {'样本数':>10}")
    print("-" * 60)
    for r in success:
        print(f"{r['model']:<20} {r.get('auc',0):>8.4f} {r.get('logloss',0):>10.4f} {r.get('pcoc',0):>8.3f} {r.get('n',0):>10,}")
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")

    # 保存
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(RESULT_DIR, f"benchmark_v3_{ts}.json")
    with open(result_file, 'w') as f:
        json.dump({'results': success, 'config': CONFIG, 'time_sec': elapsed, 'timestamp': ts},
                  f, indent=2, default=str)
    print(f"结果: {result_file}")

    # 飞书通知
    try:
        top5 = '\n'.join([
            f"  {r['model']}: AUC={r.get('auc',0):.4f}, PCOC={r.get('pcoc',0):.3f}"
            for r in success[:5]
        ])
        msg = f"""⚔️ CTR Benchmark V3 完成

配置: 全量数据, 7+1天, ML+DL
模型数: {len(success)} 个成功
总耗时: {elapsed/60:.1f} 分钟

Top 5:
{top5}
"""
        FeishuNotifier.notify(msg)
        print("✅ 飞书通知已发送")
    except Exception as e:
        print(f"飞书通知失败: {e}")

    return success


if __name__ == '__main__':
    main()
