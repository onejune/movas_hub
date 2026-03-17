# -*- coding: utf-8 -*-
"""
CTR 模型全量对比实验 V2
=======================
- 自动发现所有特征（252个）
- 30 天训练，3 天测试
- 每个模型只跑 1 epoch
- 评估指标：AUC + LogLoss + PCOC
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

# 飞书通知
sys.path.append('/mnt/workspace/walter.wan/utils')
from feishu_notifier import FeishuNotifier


# ============================================================
# 配置
# ============================================================

DATA_PATH = '/mnt/data/oss_wanjun/pai_work/defer_sample_parquet/'
TRAIN_DAYS = 30
TEST_DAYS = 3
EPOCHS = 1

# 排除的列（非特征）
EXCLUDE_COLS = ['req_hour', 'sample_bid_date', 'sample_date', 'label', 'diff_hours']

# 高基数阈值
HIGH_CARDINALITY_THRESHOLD = 50

# Embedding 维度
EMBED_DIM = 8  # 特征多，用小一点的 embedding


# ============================================================
# 评估指标
# ============================================================

def calc_pcoc(y_true, y_pred):
    """PCOC = mean(y_pred) / mean(y_true)"""
    pred_mean = np.mean(y_pred)
    true_mean = np.mean(y_true)
    if true_mean == 0:
        return float('inf')
    return pred_mean / true_mean


# ============================================================
# 数据加载
# ============================================================

def load_data():
    """加载数据"""
    date_dirs = sorted([d for d in os.listdir(DATA_PATH) if d.startswith('sample_date=')])
    
    total_days = TRAIN_DAYS + TEST_DAYS
    recent_dirs = date_dirs[-total_days:]
    train_dirs = recent_dirs[:TRAIN_DAYS]
    test_dirs = recent_dirs[TRAIN_DAYS:]
    
    print(f"训练集: {len(train_dirs)} 天 ({train_dirs[0]} ~ {train_dirs[-1]})")
    print(f"测试集: {len(test_dirs)} 天 ({test_dirs[0]} ~ {test_dirs[-1]})")
    
    train_df = pd.concat([pd.read_parquet(os.path.join(DATA_PATH, d)) for d in train_dirs], ignore_index=True)
    test_df = pd.concat([pd.read_parquet(os.path.join(DATA_PATH, d)) for d in test_dirs], ignore_index=True)
    
    return train_df, test_df


# ============================================================
# 特征自动发现
# ============================================================

def auto_discover_features(df):
    """自动发现特征 - 所有特征都是类别特征"""
    sparse_features = []
    
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        # 所有特征都当作稀疏（类别）特征
        sparse_features.append(col)
    
    return sparse_features, []  # 没有稠密特征


# ============================================================
# 特征处理
# ============================================================

def preprocess_features(train_df, test_df, sparse_features, dense_features):
    """预处理特征"""
    
    # 高基数特征处理
    for col in sparse_features:
        vc = train_df[col].value_counts()
        valid = set(vc[vc >= HIGH_CARDINALITY_THRESHOLD].index)
        train_df[col] = train_df[col].apply(lambda x: x if x in valid else 'OTHER')
        test_df[col] = test_df[col].apply(lambda x: x if x in valid else 'OTHER')
    
    # Label Encoding
    for col in sparse_features:
        le = LabelEncoder()
        all_vals = pd.concat([train_df[col].astype(str), test_df[col].astype(str)]).unique()
        le.fit(all_vals)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
    
    # 稠密特征填充
    for col in dense_features:
        train_df[col] = train_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)
    
    # 构建特征列
    feature_columns = []
    for col in sparse_features:
        vocab_size = int(max(train_df[col].max(), test_df[col].max())) + 2
        feature_columns.append(SparseFeat(col, vocabulary_size=vocab_size, embedding_dim=EMBED_DIM))
    
    for col in dense_features:
        feature_columns.append(DenseFeat(col, 1))
    
    # 构建输入
    all_cols = sparse_features + dense_features
    train_input = {col: train_df[col].values for col in all_cols}
    test_input = {col: test_df[col].values for col in all_cols}
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    return feature_columns, train_input, y_train, test_input, y_test


# ============================================================
# 模型训练
# ============================================================

MODELS = {
    'DeepFM': DeepFM,
    'DCN': DCN,
    # 'xDeepFM': xDeepFM,  # 跳过，太慢
    'AutoInt': AutoInt,
    'WDL': WDL,
    'FiBiNET': FiBiNET,
}


def train_and_evaluate(model_name, model_class, feature_columns,
                       train_input, y_train, test_input, y_test,
                       epochs=1, device='cpu'):
    """训练并评估模型"""
    
    # 创建模型
    if model_name == 'xDeepFM':
        model = model_class(
            feature_columns, feature_columns,
            cin_layer_size=(64, 64),
            dnn_hidden_units=(128, 64),
            device=device
        )
    elif model_name == 'AutoInt':
        model = model_class(
            feature_columns, feature_columns,
            att_layer_num=2,
            att_head_num=2,
            dnn_hidden_units=(128, 64),
            device=device
        )
    elif model_name == 'FiBiNET':
        model = model_class(
            feature_columns, feature_columns,
            bilinear_type='interaction',
            dnn_hidden_units=(128, 64),
            device=device
        )
    else:
        model = model_class(
            feature_columns, feature_columns,
            dnn_hidden_units=(128, 64),
            device=device
        )
    
    model.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy', 'auc'])
    
    # 训练
    model.fit(train_input, y_train, batch_size=2048, epochs=epochs, verbose=1)
    
    # 预测
    y_pred = model.predict(test_input, batch_size=2048)
    y_pred = y_pred.flatten()
    
    # 评估
    auc = roc_auc_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred)
    pcoc = calc_pcoc(y_test, y_pred)
    
    return auc, logloss, pcoc


# ============================================================
# 主函数
# ============================================================

def main():
    start_time = datetime.now()
    
    print("=" * 60)
    print("CTR 模型全量对比实验 V2 (252 特征)")
    print("=" * 60)
    print(f"开始时间: {start_time}")
    print()
    
    # 加载数据
    print("[1] 加载数据...")
    train_df, test_df = load_data()
    print(f"训练集: {len(train_df):,}, 测试集: {len(test_df):,}")
    print(f"正样本率: 训练 {train_df['label'].mean():.4f}, 测试 {test_df['label'].mean():.4f}")
    print()
    
    # 自动发现特征
    print("[2] 特征发现...")
    sparse_features, dense_features = auto_discover_features(train_df)
    print(f"稀疏特征: {len(sparse_features)}")
    print(f"稠密特征: {len(dense_features)}")
    print()
    
    # 特征处理
    print("[3] 特征处理...")
    feature_columns, train_input, y_train, test_input, y_test = preprocess_features(
        train_df.copy(), test_df.copy(), sparse_features, dense_features
    )
    print(f"总特征数: {len(feature_columns)}")
    print()
    
    # 训练模型
    print("[4] 训练模型...")
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
                epochs=EPOCHS, device=device
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
            import traceback
            traceback.print_exc()
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
    
    # 输出结果
    print("=" * 60)
    print("实验结果（按 AUC 排序）")
    print("=" * 60)
    print(f"{'模型':<15} {'AUC':>10} {'LogLoss':>10} {'PCOC':>8} {'耗时(s)':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<15} {r['auc']:>10.6f} {r['logloss']:>10.6f} {r['pcoc']:>8.4f} {r['time_sec']:>10.1f}")
    print("-" * 60)
    print(f"总耗时: {total_time:.1f}s")
    
    # 发飞书通知
    try:
        msg = f"""CTR 模型 Benchmark V2 完成 (252特征)

训练集: {len(train_df):,} 样本 ({TRAIN_DAYS}天)
测试集: {len(test_df):,} 样本 ({TEST_DAYS}天)
特征数: {len(feature_columns)} (稀疏{len(sparse_features)} + 稠密{len(dense_features)})

结果 (按AUC排序):
"""
        for r in results:
            msg += f"  {r['model']}: AUC={r['auc']:.4f}, LogLoss={r['logloss']:.4f}, PCOC={r['pcoc']:.3f}, {r['time_sec']:.0f}s\n"
        msg += f"\n总耗时: {total_time:.1f}s"
        
        FeishuNotifier.notify(msg)
        print("\n✅ 飞书通知已发送")
    except Exception as e:
        print(f"\n⚠️ 飞书通知失败: {e}")
    
    # 保存结果
    result_file = f"/tmp/benchmark_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(result_file, 'w') as f:
        f.write(f"CTR Benchmark V2 Results\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"Features: {len(feature_columns)} (sparse: {len(sparse_features)}, dense: {len(dense_features)})\n")
        f.write(f"Train: {len(train_df):,}, Test: {len(test_df):,}\n\n")
        for r in results:
            f.write(f"{r['model']}: AUC={r['auc']:.6f}, LogLoss={r['logloss']:.6f}, PCOC={r['pcoc']:.4f}, Time={r['time_sec']:.1f}s\n")
    print(f"结果已保存: {result_file}")


if __name__ == '__main__':
    main()
