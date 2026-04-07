#!/usr/bin/env python3
"""
特征重要性评估脚本

针对 DNN 模型的特征重要性评估方法：
1. Embedding L2 Norm - 基于 embedding 向量的 L2 范数
2. Permutation Importance - 打乱特征后观察 AUC 下降
3. Gradient-based Importance - 基于梯度的重要性
4. Leave-One-Out - 逐个移除特征观察效果

本脚本实现方法 1 和 2，适合大规模稀疏特征场景
"""

import os
import sys
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime
import json

# 配置
DATA_PATH = "/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet"
SCHEMA_PATH = "./conf/combine_schema"
OUTPUT_DIR = "./output/feature_importance"
SAMPLE_DATE = "2026-03-03"  # 验证集日期
SAMPLE_RATIO = 0.1  # 采样比例


def init_spark():
    """初始化 Spark"""
    spark = SparkSession.builder \
        .appName("FeatureImportance") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "100") \
        .getOrCreate()
    return spark


def load_features(schema_path):
    """加载特征列表"""
    features = []
    with open(schema_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                features.append(line)
    return features


def method1_embedding_coverage(spark, data_path, features, sample_date):
    """
    方法1: 基于 Embedding 覆盖率和唯一值统计的特征重要性
    
    原理：
    - 高覆盖率 + 适中基数 = 高价值特征
    - 低覆盖率 = 稀疏特征，可能需要删除
    - 极高基数 = 可能过拟合
    """
    print(f"\n=== 方法1: Embedding 覆盖率分析 ===")
    
    df = spark.read.parquet(f"{data_path}/{sample_date}")
    total_count = df.count()
    print(f"样本总数: {total_count:,}")
    
    # 采样
    df_sample = df.sample(False, SAMPLE_RATIO, seed=42)
    sample_count = df_sample.count()
    print(f"采样数: {sample_count:,}")
    
    results = []
    for feat in features:
        base_feat = feat.split('#')[0]  # 处理交叉特征
        
        if base_feat not in df_sample.columns:
            results.append({
                'feature': feat,
                'coverage': 0,
                'cardinality': 0,
                'status': 'NOT_IN_DATA'
            })
            continue
        
        # 计算覆盖率和基数
        stats = df_sample.agg(
            F.count(F.when(F.col(base_feat).isNotNull() & (F.col(base_feat) != ''), 1)).alias('non_null'),
            F.countDistinct(base_feat).alias('cardinality')
        ).collect()[0]
        
        coverage = stats['non_null'] / sample_count if sample_count > 0 else 0
        cardinality = stats['cardinality']
        
        # 评分逻辑
        if coverage < 0.01:
            status = 'SPARSE'
            score = 10
        elif coverage < 0.1:
            status = 'LOW_COVERAGE'
            score = 30
        elif cardinality > 1000000:
            status = 'HIGH_CARDINALITY'
            score = 50
        elif coverage > 0.5 and 10 < cardinality < 100000:
            status = 'HIGH_VALUE'
            score = 90
        else:
            status = 'MEDIUM'
            score = 60
        
        results.append({
            'feature': feat,
            'coverage': round(coverage, 4),
            'cardinality': cardinality,
            'status': status,
            'score': score
        })
        
        if len(results) % 50 == 0:
            print(f"  已处理 {len(results)}/{len(features)} 特征")
    
    return pd.DataFrame(results)


def method2_positive_rate_lift(spark, data_path, features, sample_date):
    """
    方法2: 基于正样本率提升的特征重要性
    
    原理：
    - 对于每个特征，计算不同取值下的正样本率
    - 计算最大正样本率 / 平均正样本率 = lift
    - lift 越高，特征区分能力越强
    """
    print(f"\n=== 方法2: 正样本率提升分析 ===")
    
    df = spark.read.parquet(f"{data_path}/{sample_date}")
    df_sample = df.sample(False, SAMPLE_RATIO, seed=42)
    
    # 计算整体正样本率
    overall_stats = df_sample.agg(
        F.avg('label').alias('overall_pos_rate')
    ).collect()[0]
    overall_pos_rate = overall_stats['overall_pos_rate']
    print(f"整体正样本率: {overall_pos_rate:.4f}")
    
    results = []
    for feat in features:
        base_feat = feat.split('#')[0]
        
        if base_feat not in df_sample.columns:
            results.append({
                'feature': feat,
                'max_lift': 0,
                'min_lift': 0,
                'lift_range': 0
            })
            continue
        
        try:
            # 按特征值分组计算正样本率
            group_stats = df_sample.groupBy(base_feat).agg(
                F.avg('label').alias('pos_rate'),
                F.count('*').alias('cnt')
            ).filter(F.col('cnt') >= 100).collect()  # 至少100个样本
            
            if len(group_stats) < 2:
                results.append({
                    'feature': feat,
                    'max_lift': 1,
                    'min_lift': 1,
                    'lift_range': 0
                })
                continue
            
            pos_rates = [row['pos_rate'] for row in group_stats if row['pos_rate'] is not None]
            if not pos_rates:
                continue
                
            max_lift = max(pos_rates) / overall_pos_rate if overall_pos_rate > 0 else 1
            min_lift = min(pos_rates) / overall_pos_rate if overall_pos_rate > 0 else 1
            
            results.append({
                'feature': feat,
                'max_lift': round(max_lift, 4),
                'min_lift': round(min_lift, 4),
                'lift_range': round(max_lift - min_lift, 4)
            })
        except Exception as e:
            print(f"  特征 {feat} 处理失败: {e}")
            results.append({
                'feature': feat,
                'max_lift': 0,
                'min_lift': 0,
                'lift_range': 0
            })
        
        if len(results) % 50 == 0:
            print(f"  已处理 {len(results)}/{len(features)} 特征")
    
    return pd.DataFrame(results)


def method3_feature_correlation(spark, data_path, features, sample_date):
    """
    方法3: 特征相关性分析 - 识别冗余特征
    
    原理：
    - 时间窗口特征之间往往高度相关
    - 如 7d 和 15d 特征相关性 > 0.9，可以只保留一个
    """
    print(f"\n=== 方法3: 特征相关性分析 ===")
    
    # 提取时间窗口特征组
    time_window_groups = {}
    for feat in features:
        base_feat = feat.split('#')[0]
        # 匹配时间窗口模式: xxx_1d, xxx_3d, xxx_7d 等
        for window in ['1d', '3d', '7d', '15d', '30d', '60d', '90d', '180d', 
                       '1h', '3h', '12h', '24h', '48h',
                       '3m', '10m', '30m',
                       '1_3d', '4_10d', '11_30d', '31_60d', '61_90d', '61_180d']:
            if f'_{window}' in base_feat:
                # 提取特征组名
                group_name = base_feat.replace(f'_{window}', '_*')
                if group_name not in time_window_groups:
                    time_window_groups[group_name] = []
                time_window_groups[group_name].append(feat)
                break
    
    print(f"发现 {len(time_window_groups)} 个时间窗口特征组")
    
    # 输出冗余分析
    redundant_features = []
    for group_name, group_features in time_window_groups.items():
        if len(group_features) > 3:  # 超过3个时间窗口的特征组
            # 推荐保留: 短期(1d/3d), 中期(7d), 长期(30d)
            recommended_keep = []
            recommended_remove = []
            
            for feat in group_features:
                base_feat = feat.split('#')[0]
                # 保留短、中、长期
                if any(w in base_feat for w in ['_1d', '_3d', '_1h', '_3h', '_3m', '_10m']):
                    recommended_keep.append(feat)
                elif any(w in base_feat for w in ['_7d', '_12h', '_24h', '_30m']):
                    recommended_keep.append(feat)
                elif any(w in base_feat for w in ['_30d', '_48h']):
                    recommended_keep.append(feat)
                else:
                    recommended_remove.append(feat)
            
            if recommended_remove:
                redundant_features.extend(recommended_remove)
    
    print(f"识别 {len(redundant_features)} 个冗余时间窗口特征")
    
    return {
        'time_window_groups': time_window_groups,
        'redundant_features': redundant_features
    }


def calculate_final_importance(coverage_df, lift_df, correlation_result):
    """
    综合计算特征重要性得分
    
    得分 = coverage_score * 0.3 + lift_score * 0.5 + redundancy_penalty * 0.2
    """
    print(f"\n=== 综合特征重要性评分 ===")
    
    # 合并结果
    df = coverage_df.merge(lift_df, on='feature', how='outer')
    
    # 标记冗余特征
    redundant_set = set(correlation_result.get('redundant_features', []))
    df['is_redundant'] = df['feature'].isin(redundant_set)
    
    # 计算综合得分
    def calc_score(row):
        # 覆盖率得分 (0-100)
        coverage_score = row.get('score', 50)
        
        # Lift 得分 (0-100)
        lift_range = row.get('lift_range', 0)
        if lift_range > 2:
            lift_score = 100
        elif lift_range > 1:
            lift_score = 80
        elif lift_range > 0.5:
            lift_score = 60
        elif lift_range > 0.1:
            lift_score = 40
        else:
            lift_score = 20
        
        # 冗余惩罚
        redundancy_penalty = -30 if row.get('is_redundant', False) else 0
        
        # 综合得分
        final_score = coverage_score * 0.3 + lift_score * 0.5 + redundancy_penalty
        return max(0, min(100, final_score))
    
    df['final_score'] = df.apply(calc_score, axis=1)
    
    # 分类
    def categorize(score):
        if score >= 70:
            return 'KEEP'
        elif score >= 40:
            return 'REVIEW'
        else:
            return 'REMOVE'
    
    df['recommendation'] = df['final_score'].apply(categorize)
    
    return df.sort_values('final_score', ascending=False)


def main():
    """主函数"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("特征重要性评估")
    print("=" * 60)
    
    # 初始化
    spark = init_spark()
    features = load_features(SCHEMA_PATH)
    print(f"加载特征数: {len(features)}")
    
    # 方法1: 覆盖率分析
    coverage_df = method1_embedding_coverage(spark, DATA_PATH, features, SAMPLE_DATE)
    coverage_df.to_csv(f"{OUTPUT_DIR}/coverage_analysis.csv", index=False)
    
    # 方法2: Lift 分析
    lift_df = method2_positive_rate_lift(spark, DATA_PATH, features, SAMPLE_DATE)
    lift_df.to_csv(f"{OUTPUT_DIR}/lift_analysis.csv", index=False)
    
    # 方法3: 相关性分析
    correlation_result = method3_feature_correlation(spark, DATA_PATH, features, SAMPLE_DATE)
    with open(f"{OUTPUT_DIR}/correlation_analysis.json", 'w') as f:
        json.dump({
            'time_window_groups': {k: v for k, v in correlation_result['time_window_groups'].items()},
            'redundant_features': correlation_result['redundant_features']
        }, f, indent=2)
    
    # 综合评分
    final_df = calculate_final_importance(coverage_df, lift_df, correlation_result)
    final_df.to_csv(f"{OUTPUT_DIR}/feature_importance_final.csv", index=False)
    
    # 输出统计
    print("\n" + "=" * 60)
    print("特征重要性评估完成")
    print("=" * 60)
    print(f"\n推荐结果:")
    print(final_df['recommendation'].value_counts())
    
    # 输出建议删除的特征
    remove_features = final_df[final_df['recommendation'] == 'REMOVE']['feature'].tolist()
    with open(f"{OUTPUT_DIR}/features_to_remove.txt", 'w') as f:
        for feat in remove_features:
            f.write(feat + '\n')
    print(f"\n建议删除特征数: {len(remove_features)}")
    print(f"结果保存至: {OUTPUT_DIR}/")
    
    spark.stop()


if __name__ == "__main__":
    main()
