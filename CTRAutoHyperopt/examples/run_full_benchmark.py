# -*- coding: utf-8 -*-
"""
CTR 模型全量对比实验
====================
- 30 天训练，3 天测试
- 每个模型只跑 1 epoch
- 评估指标：AUC + PCOC
- 完成后发飞书通知
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

# 飞书通知
sys.path.append('/mnt/workspace/walter.wan/utils')
from feishu_notifier import FeishuNotifier


# ============================================================
# 评估指标
# ============================================================

def calc_pcoc(y_true, y_pred):
    """计算 PCOC (Predicted / Observed CTR)
    
    PCOC = mean(y_pred) / mean(y_true)
    理想值为 1.0，表示预估值和真实值的均值一致
    """
    pred_mean = np.mean(y_pred)
    true_mean = np.mean(y_true)
    if true_mean == 0:
        return float('inf')
    return pred_mean / true_mean


# ============================================================
# 数据加载
# ============================================================

def load_data(data_path: str, train_days: int = 30, test_days: int = 3):
    """加载数据：N 天训练，M 天测试
    
    Args:
        data_path: 数据目录
        train_days: 训练天数
        test_days: 测试天数
    """
    date_dirs = sorted([d for d in os.listdir(data_path) if d.startswith('sample_date=')])
    
    total_days = train_days + test_days
    if len(date_dirs) < total_days:
        print(f"Warning: Only {len(date_dirs)} days available, using all")
        total_days = len(date_dirs)
        test_days = min(test_days, total_days // 4)
        train_days = total_days - test_days
    
    # 取最近的数据
    recent_dirs = date_dirs[-total_days:]
    train_dirs = recent_dirs[:train_days]
    test_dirs = recent_dirs[train_days:]
    
    print(f"训练集: {len(train_dirs)} 天 ({train_dirs[0]} ~ {train_dirs[-1]})")
    print(f"测试集: {len(test_dirs)} 天 ({test_dirs[0]} ~ {test_dirs[-1]})")
    
    train_dfs = []
    for d in train_dirs:
        df = pd.read_parquet(os.path.join(data_path, d))
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    test_dfs = []
    for d in test_dirs:
        df = pd.read_parquet(os.path.join(data_path, d))
        test_dfs.append(df)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    return train_df, test_df


# ============================================================
# 特征处理
# ============================================================

def preprocess_features(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                        config: FeatureConfig):
    """根据配置预处理特征"""
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
        vocab_size = int(max(train_df[col].max(), test_df[col].max())) + 2  # +2 for safety
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
    'WDL': WDL,
    'FiBiNET': FiBiNET,
    # 'NFM': NFM,  # 容易报错，暂时跳过
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
    pcoc = calc_pcoc(y_test, y_pred)
    
    return auc, logloss, pcoc


# ============================================================
# 主函数
# ============================================================

def run_benchmark(data_path: str = '/mnt/data/oss_wanjun/pai_work/defer_sample_parquet/',
                  train_days: int = 30,
                  test_days: int = 3,
                  epochs: int = 1,
                  config: FeatureConfig = None,
                  send_feishu: bool = True):
    """运行模型对比实验"""
    
    if config is None:
        config = DEFAULT_CONFIG
    
    start_time = datetime.now()
    
    print("=" * 60)
    print("CTR 模型全量对比实验")
    print("=" * 60)
    print(f"开始时间: {start_time}")
    print(f"训练天数: {train_days}")
    print(f"测试天数: {test_days}")
    print(f"训练轮数: {epochs}")
    print()
    
    # 加载数据
    print("[1] 加载数据...")
    train_df, test_df = load_data(data_path, train_days, test_days)
    print(f"训练集: {len(train_df):,}, 测试集: {len(test_df):,}")
    print(f"正样本率: 训练 {train_df['label'].mean():.4f}, 测试 {test_df['label'].mean():.4f}")
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
        print(f"{'='*20} {model_name} {'='*20}")
        try:
            model_start = datetime.now()
            auc, logloss, pcoc = train_and_evaluate(
                model_name, model_class, feature_columns,
                train_input, y_train, test_input, y_test,
                epochs=epochs, device=device
            )
            model_time = (datetime.now() - model_start).total_seconds()
            
            results.append({
                'model': model_name,
                'auc': auc,
                'logloss': logloss,
                'pcoc': pcoc,
                'time_sec': model_time,
            })
            print(f"AUC: {auc:.6f}, LogLoss: {logloss:.6f}, PCOC: {pcoc:.4f}, Time: {model_time:.1f}s")
        except Exception as e:
            print(f"Failed: {e}")
            results.append({
                'model': model_name,
                'auc': 0.5,
                'logloss': 1.0,
                'pcoc': 0.0,
                'time_sec': 0,
            })
        print()
    
    # 排序输出
    results = sorted(results, key=lambda x: -x['auc'])
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print("=" * 60)
    print("最终排名")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r['model']:15s} AUC: {r['auc']:.6f}  PCOC: {r['pcoc']:.4f}  LogLoss: {r['logloss']:.6f}")
    
    print()
    print(f"最优模型: {results[0]['model']} (AUC={results[0]['auc']:.6f})")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"结束时间: {end_time}")
    
    # 发送飞书通知
    if send_feishu:
        send_feishu_report(results, train_days, test_days, len(train_df), len(test_df), total_time)
    
    return results


def send_feishu_report(results, train_days, test_days, train_size, test_size, total_time):
    """发送飞书通知"""
    
    # 构建表格
    lines = [
        "=" * 40,
        "⚔️  萧十一郎 工作汇报",
        "=" * 40,
        "",
        "📊 CTR 模型对比实验完成",
        "",
        f"数据配置:",
        f"  - 训练: {train_days} 天, {train_size:,} 样本",
        f"  - 测试: {test_days} 天, {test_size:,} 样本",
        f"  - 总耗时: {total_time/60:.1f} 分钟",
        "",
        "模型排名 (按 AUC):",
        "-" * 40,
    ]
    
    for i, r in enumerate(results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        lines.append(f"{medal} {r['model']:12s} AUC={r['auc']:.4f} PCOC={r['pcoc']:.4f}")
    
    lines.extend([
        "-" * 40,
        "",
        f"🏆 最优模型: {results[0]['model']}",
        f"   AUC: {results[0]['auc']:.6f}",
        f"   PCOC: {results[0]['pcoc']:.4f}",
        "",
        "以上，老板。晚安！",
    ])
    
    message = "\n".join(lines)
    
    try:
        FeishuNotifier.send_text(message)
        print("\n✅ 飞书通知已发送")
    except Exception as e:
        print(f"\n❌ 飞书通知发送失败: {e}")


if __name__ == '__main__':
    config = FeatureConfig(
        sparse_features=['business_type', 'offerid', 'country', 'bundle', 'adx', 'make', 'model', 'demand_pkgname', 'campaignid'],
        cross_features=[('country', 'adx'), ('make', 'model')],
        high_cardinality_threshold=50,
        embed_dim=16,
    )
    
    run_benchmark(
        train_days=30,
        test_days=3,
        epochs=1,
        config=config,
        send_feishu=True,
    )
