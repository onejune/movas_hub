#!/usr/bin/env python3
"""
Permutation Importance 分析
方法: 打乱某个特征的值，看 AUC 下降多少

优点:
- 不需要理解模型内部结构
- 直接衡量特征对预测的贡献
- 可以发现非线性关系

步骤:
1. 计算基准 AUC
2. 对每个特征: 打乱该特征值，计算新 AUC
3. 重要性 = 基准 AUC - 打乱后 AUC
"""

import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
import glob
from collections import defaultdict

# 配置
DATA_PATH = "/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet"
SCHEMA_PATH = "./conf/combine_schema"
OUTPUT_DIR = "./output/permutation_importance"
PRED_PATH = "/mnt/workspace/walter.wan/model_experiment/dnn/dnn_ivr16_v1/pred_results"  # 预测结果路径

SAMPLE_DATE = "2026-03-03"  # 验证集日期
NUM_FILES = 5  # 读取的文件数
N_REPEATS = 3  # 每个特征重复打乱次数


def load_features(schema_path):
    """加载特征列表"""
    features = []
    with open(schema_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                features.append(line)
    return features


def load_predictions(pred_path, date):
    """加载模型预测结果"""
    pred_dir = os.path.join(pred_path, date)
    if not os.path.exists(pred_dir):
        print(f"预测结果目录不存在: {pred_dir}")
        return None
    
    files = glob.glob(f"{pred_dir}/part-*.parquet")
    if not files:
        files = glob.glob(f"{pred_dir}/*.csv")
    
    print(f"找到 {len(files)} 个预测文件")
    
    dfs = []
    for f in files[:NUM_FILES]:
        if f.endswith('.parquet'):
            df = pq.read_table(f).to_pandas()
        else:
            df = pd.read_csv(f)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"预测数据: {len(df)} 行, 列: {df.columns.tolist()}")
    return df


def load_data_with_predictions(data_path, pred_path, date, features, num_files=5):
    """加载原始数据和预测结果"""
    print(f"\n=== 加载数据 ===")
    
    # 加载预测结果
    pred_df = load_predictions(pred_path, date)
    
    # 如果没有预测结果，需要用模型重新预测
    if pred_df is None:
        print("没有找到预测结果，需要用模型重新推理")
        print("这需要启动 Spark 环境，暂不支持")
        return None, None
    
    # 加载原始特征数据
    print(f"\n加载原始特征: {data_path}/{date}")
    base_features = list(set([f.split('#')[0] for f in features]))
    base_features.append('label')
    
    files = sorted(glob.glob(f"{data_path}/{date}/part-*.parquet"))[:num_files]
    
    dfs = []
    for f in files:
        pf = pq.ParquetFile(f)
        available = set(pf.schema.names)
        cols = [c for c in base_features if c in available]
        df = pq.read_table(f, columns=cols).to_pandas()
        dfs.append(df)
    
    feature_df = pd.concat(dfs, ignore_index=True)
    print(f"特征数据: {len(feature_df)} 行")
    
    # 对齐数据
    min_len = min(len(feature_df), len(pred_df))
    feature_df = feature_df.iloc[:min_len].reset_index(drop=True)
    pred_df = pred_df.iloc[:min_len].reset_index(drop=True)
    
    return feature_df, pred_df


def calculate_auc(y_true, y_pred):
    """计算 AUC"""
    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return 0.5


def permutation_importance_simple(feature_df, pred_df, features, n_repeats=3):
    """
    简化版 Permutation Importance
    由于我们没法重新推理，用一个近似方法:
    分析特征值分布与预测值的相关性
    """
    print("\n=== 特征-预测相关性分析 ===")
    
    # 获取预测列名
    pred_col = None
    for col in ['prediction', 'pred', 'score', 'probability']:
        if col in pred_df.columns:
            pred_col = col
            break
    
    if pred_col is None:
        print(f"找不到预测列，可用列: {pred_df.columns.tolist()}")
        return pd.DataFrame()
    
    y_pred = pred_df[pred_col].values
    y_true = feature_df['label'].values if 'label' in feature_df.columns else pred_df.get('label', pred_df.get('y_true'))
    
    if y_true is None:
        print("找不到 label 列")
        return pd.DataFrame()
    
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    
    base_auc = calculate_auc(y_true, y_pred)
    print(f"基准 AUC: {base_auc:.4f}")
    
    results = []
    
    for i, feat in enumerate(features):
        base_feat = feat.split('#')[0]
        
        if base_feat not in feature_df.columns:
            results.append({
                'feature': feat,
                'importance': 0,
                'importance_std': 0,
                'status': 'NOT_IN_DATA'
            })
            continue
        
        col = feature_df[base_feat]
        
        # 多次打乱计算
        auc_drops = []
        for _ in range(n_repeats):
            # 打乱特征值
            shuffled = col.sample(frac=1, random_state=np.random.randint(10000)).values
            
            # 由于我们没法用打乱后的特征重新推理
            # 这里用一个近似: 计算打乱前后特征与预测值的相关性变化
            # 这不是真正的 permutation importance，但可以给出参考
            
            # 计算原始相关性
            if col.dtype in ['object', 'category']:
                # 分类特征: 用 groupby 的预测均值方差
                try:
                    grp = pd.DataFrame({'feat': col, 'pred': y_pred}).groupby('feat')['pred'].agg(['mean', 'std', 'count'])
                    grp = grp[grp['count'] >= 20]
                    orig_var = grp['mean'].var() if len(grp) > 1 else 0
                    
                    grp_shuffled = pd.DataFrame({'feat': shuffled, 'pred': y_pred}).groupby('feat')['pred'].agg(['mean', 'std', 'count'])
                    grp_shuffled = grp_shuffled[grp_shuffled['count'] >= 20]
                    shuffled_var = grp_shuffled['mean'].var() if len(grp_shuffled) > 1 else 0
                    
                    # 方差下降 = 特征重要
                    auc_drop = orig_var - shuffled_var
                except:
                    auc_drop = 0
            else:
                # 数值特征: 用相关系数
                try:
                    orig_corr = np.abs(np.corrcoef(col.fillna(0), y_pred)[0, 1])
                    shuffled_corr = np.abs(np.corrcoef(shuffled, y_pred)[0, 1])
                    auc_drop = orig_corr - shuffled_corr
                except:
                    auc_drop = 0
            
            auc_drops.append(auc_drop)
        
        importance = np.mean(auc_drops)
        importance_std = np.std(auc_drops)
        
        results.append({
            'feature': feat,
            'importance': round(float(importance), 6),
            'importance_std': round(float(importance_std), 6),
            'status': 'OK'
        })
        
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(features)}")
    
    df = pd.DataFrame(results)
    df = df.sort_values('importance', ascending=False)
    
    # 分类
    p75 = df[df['status'] == 'OK']['importance'].quantile(0.75)
    p50 = df[df['status'] == 'OK']['importance'].quantile(0.50)
    p25 = df[df['status'] == 'OK']['importance'].quantile(0.25)
    
    def classify(row):
        if row['status'] != 'OK':
            return row['status']
        imp = row['importance']
        if imp >= p75:
            return 'HIGH'
        elif imp >= p50:
            return 'MEDIUM_HIGH'
        elif imp >= p25:
            return 'MEDIUM_LOW'
        else:
            return 'LOW'
    
    df['importance_level'] = df.apply(classify, axis=1)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Permutation Importance 分析 (近似版)")
    print("=" * 60)
    
    # 加载特征
    features = load_features(SCHEMA_PATH)
    print(f"特征数: {len(features)}")
    
    # 加载数据
    feature_df, pred_df = load_data_with_predictions(
        DATA_PATH, PRED_PATH, SAMPLE_DATE, features, NUM_FILES
    )
    
    if feature_df is None:
        print("\n无法加载数据，退出")
        return
    
    # 分析
    result = permutation_importance_simple(feature_df, pred_df, features, N_REPEATS)
    
    if len(result) > 0:
        result.to_csv(os.path.join(OUTPUT_DIR, "permutation_importance.csv"), index=False)
        
        print("\n" + "=" * 60)
        print("Top 30 重要特征")
        print("=" * 60)
        print(result.head(30).to_string(index=False))
        
        print("\n" + "=" * 60)
        print("Bottom 20 特征")
        print("=" * 60)
        print(result.tail(20).to_string(index=False))
        
        print("\n--- 重要性分布 ---")
        print(result['importance_level'].value_counts())
        
        # 与 lift 分析对比
        lift_path = "./output/feature_importance/importance.csv"
        if os.path.exists(lift_path):
            lift_df = pd.read_csv(lift_path)
            merged = result.merge(lift_df[['feature', 'action', 'score']], on='feature', how='left', suffixes=('', '_lift'))
            
            # 找出两种方法结论不一致的特征
            merged['perm_action'] = merged['importance_level'].apply(
                lambda x: 'KEEP' if x in ['HIGH', 'MEDIUM_HIGH'] else ('REVIEW' if x == 'MEDIUM_LOW' else 'REMOVE')
            )
            
            disagree = merged[(merged['action'] != merged['perm_action']) & (merged['action'].notna())]
            if len(disagree) > 0:
                print("\n--- 两种方法结论不一致的特征 ---")
                print(disagree[['feature', 'importance', 'importance_level', 'perm_action', 'action', 'score']].head(20).to_string(index=False))
    
    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
