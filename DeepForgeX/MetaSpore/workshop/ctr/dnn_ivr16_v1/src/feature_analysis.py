#!/usr/bin/env python3
"""
特征重要性分析脚本
分析combine_schema中特征的覆盖率、分布和与label的相关性
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from collections import defaultdict

# 配置
DATA_PATH = "/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet"
SAMPLE_DATE = "2026-03-20"  # 使用一天数据分析
COMBINE_SCHEMA_PATH = "./conf/combine_schema"
OUTPUT_PATH = "./output/feature_analysis"

def init_spark():
    """初始化Spark"""
    spark = SparkSession.builder \
        .appName("FeatureAnalysis") \
        .config("spark.driver.memory", "16g") \
        .config("spark.executor.memory", "16g") \
        .config("spark.sql.shuffle.partitions", "100") \
        .getOrCreate()
    return spark

def load_combine_schema(path):
    """加载combine_schema特征列表"""
    features = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # 处理交叉特征格式 feature#key
                if '#' in line:
                    feat_name = line.split('#')[0]
                else:
                    feat_name = line
                features.append((line, feat_name))
    return features

def analyze_feature_coverage(spark, df, features, label_col='label'):
    """分析特征覆盖率和与label的相关性"""
    results = []
    total_count = df.count()
    pos_count = df.filter(F.col(label_col) == 1).count()
    neg_count = total_count - pos_count
    
    print(f"Total samples: {total_count}, Pos: {pos_count}, Neg: {neg_count}")
    
    # 获取所有列名
    all_columns = set(df.columns)
    
    for schema_name, feat_name in features:
        if feat_name not in all_columns:
            results.append({
                'schema_name': schema_name,
                'feature_name': feat_name,
                'exists': False,
                'coverage': 0,
                'null_ratio': 1.0,
                'zero_ratio': 1.0,
                'distinct_count': 0,
                'pos_coverage': 0,
                'neg_coverage': 0,
                'coverage_lift': 0,
                'mean_pos': 0,
                'mean_neg': 0,
                'mean_lift': 0
            })
            continue
        
        # 计算基础统计
        stats = df.agg(
            F.count(F.when(F.col(feat_name).isNotNull() & (F.col(feat_name) != '') & (F.col(feat_name) != 'null'), 1)).alias('non_null_count'),
            F.count(F.when((F.col(feat_name) == 0) | (F.col(feat_name) == '0'), 1)).alias('zero_count'),
            F.countDistinct(feat_name).alias('distinct_count'),
            # 正样本覆盖率
            F.count(F.when((F.col(label_col) == 1) & F.col(feat_name).isNotNull() & (F.col(feat_name) != '') & (F.col(feat_name) != 'null') & (F.col(feat_name) != 0), 1)).alias('pos_non_null'),
            # 负样本覆盖率  
            F.count(F.when((F.col(label_col) == 0) & F.col(feat_name).isNotNull() & (F.col(feat_name) != '') & (F.col(feat_name) != 'null') & (F.col(feat_name) != 0), 1)).alias('neg_non_null'),
        ).collect()[0]
        
        non_null_count = stats['non_null_count']
        zero_count = stats['zero_count']
        distinct_count = stats['distinct_count']
        pos_non_null = stats['pos_non_null']
        neg_non_null = stats['neg_non_null']
        
        coverage = non_null_count / total_count if total_count > 0 else 0
        null_ratio = 1 - coverage
        zero_ratio = zero_count / total_count if total_count > 0 else 0
        pos_coverage = pos_non_null / pos_count if pos_count > 0 else 0
        neg_coverage = neg_non_null / neg_count if neg_count > 0 else 0
        coverage_lift = pos_coverage / neg_coverage if neg_coverage > 0 else 0
        
        # 尝试计算数值特征的均值差异
        mean_pos = 0
        mean_neg = 0
        mean_lift = 0
        try:
            mean_stats = df.agg(
                F.avg(F.when(F.col(label_col) == 1, F.col(feat_name).cast('double'))).alias('mean_pos'),
                F.avg(F.when(F.col(label_col) == 0, F.col(feat_name).cast('double'))).alias('mean_neg'),
            ).collect()[0]
            mean_pos = mean_stats['mean_pos'] or 0
            mean_neg = mean_stats['mean_neg'] or 0
            mean_lift = mean_pos / mean_neg if mean_neg > 0 else 0
        except:
            pass
        
        results.append({
            'schema_name': schema_name,
            'feature_name': feat_name,
            'exists': True,
            'coverage': coverage,
            'null_ratio': null_ratio,
            'zero_ratio': zero_ratio,
            'distinct_count': distinct_count,
            'pos_coverage': pos_coverage,
            'neg_coverage': neg_coverage,
            'coverage_lift': coverage_lift,
            'mean_pos': mean_pos,
            'mean_neg': mean_neg,
            'mean_lift': mean_lift
        })
        
        print(f"Analyzed: {feat_name}, coverage={coverage:.4f}, lift={coverage_lift:.4f}")
    
    return pd.DataFrame(results)

def categorize_features(df_results):
    """对特征进行分类和评级"""
    df = df_results.copy()
    
    # 评分规则
    def score_feature(row):
        score = 0
        
        # 覆盖率得分 (0-30分)
        if row['coverage'] >= 0.5:
            score += 30
        elif row['coverage'] >= 0.2:
            score += 20
        elif row['coverage'] >= 0.05:
            score += 10
        elif row['coverage'] >= 0.01:
            score += 5
        
        # 区分度得分 (0-40分) - 基于coverage_lift
        lift = row['coverage_lift']
        if lift > 2.0 or lift < 0.5:
            score += 40
        elif lift > 1.5 or lift < 0.67:
            score += 30
        elif lift > 1.2 or lift < 0.83:
            score += 20
        elif lift > 1.1 or lift < 0.91:
            score += 10
        
        # distinct_count得分 (0-20分)
        distinct = row['distinct_count']
        if 10 <= distinct <= 10000:
            score += 20
        elif 3 <= distinct < 10 or 10000 < distinct <= 100000:
            score += 15
        elif distinct > 100000:
            score += 10
        elif distinct == 2:
            score += 5
        
        # 惩罚项
        if row['zero_ratio'] > 0.95:
            score -= 20
        if row['null_ratio'] > 0.99:
            score -= 30
        
        return max(0, score)
    
    df['importance_score'] = df.apply(score_feature, axis=1)
    
    # 分类
    def categorize(row):
        if not row['exists']:
            return 'NOT_IN_DATA'
        if row['coverage'] < 0.001:
            return 'EXTREMELY_SPARSE'
        if row['coverage'] < 0.01:
            return 'VERY_SPARSE'
        if row['coverage'] < 0.05:
            return 'SPARSE'
        if row['importance_score'] >= 60:
            return 'HIGH_VALUE'
        if row['importance_score'] >= 40:
            return 'MEDIUM_VALUE'
        if row['importance_score'] >= 20:
            return 'LOW_VALUE'
        return 'CANDIDATE_REMOVE'
    
    df['category'] = df.apply(categorize, axis=1)
    
    return df.sort_values('importance_score', ascending=False)

def generate_recommendations(df_results):
    """生成特征裁剪建议"""
    recommendations = {
        'keep_high_value': [],
        'keep_medium_value': [],
        'consider_remove': [],
        'must_remove': [],
        'add_cross_features': []
    }
    
    for _, row in df_results.iterrows():
        feat = row['schema_name']
        cat = row['category']
        score = row['importance_score']
        
        if cat == 'NOT_IN_DATA':
            recommendations['must_remove'].append((feat, 'Feature not in data'))
        elif cat in ['EXTREMELY_SPARSE', 'VERY_SPARSE']:
            recommendations['must_remove'].append((feat, f'Coverage too low: {row["coverage"]:.4f}'))
        elif cat == 'CANDIDATE_REMOVE':
            recommendations['consider_remove'].append((feat, f'Low score: {score}'))
        elif cat == 'HIGH_VALUE':
            recommendations['keep_high_value'].append((feat, f'Score: {score}, Lift: {row["coverage_lift"]:.2f}'))
        elif cat == 'MEDIUM_VALUE':
            recommendations['keep_medium_value'].append((feat, f'Score: {score}'))
        elif cat == 'LOW_VALUE':
            recommendations['consider_remove'].append((feat, f'Low value, Score: {score}'))
    
    # 建议的交叉特征
    recommendations['add_cross_features'] = [
        ('country', 'demand_pkgname', '地区×应用偏好'),
        ('business_type', 'devicetype', '业务线×设备类型'),
        ('adx', 'bundle', '流量源×媒体'),
        ('country', 'business_type', '地区×业务线'),
        ('os', 'demand_pkgname', '系统×应用'),
    ]
    
    return recommendations

def main():
    print("=" * 60)
    print("Feature Analysis for dnn_ivr16_v1")
    print("=" * 60)
    
    # 初始化
    spark = init_spark()
    
    # 加载数据
    data_path = f"{DATA_PATH}/{SAMPLE_DATE}"
    print(f"\nLoading data from: {data_path}")
    df = spark.read.parquet(data_path)
    
    # 加载特征列表
    features = load_combine_schema(COMBINE_SCHEMA_PATH)
    print(f"\nLoaded {len(features)} features from combine_schema")
    
    # 分析特征
    print("\n" + "=" * 60)
    print("Analyzing feature coverage and importance...")
    print("=" * 60)
    df_results = analyze_feature_coverage(spark, df, features)
    
    # 分类和评分
    df_results = categorize_features(df_results)
    
    # 保存结果
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = f"{OUTPUT_PATH}/feature_analysis_result.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # 生成建议
    recommendations = generate_recommendations(df_results)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Category Distribution:")
    print(df_results['category'].value_counts().to_string())
    
    print(f"\n✅ HIGH VALUE Features ({len(recommendations['keep_high_value'])}):")
    for feat, reason in recommendations['keep_high_value'][:20]:
        print(f"  - {feat}: {reason}")
    
    print(f"\n⚠️ MEDIUM VALUE Features ({len(recommendations['keep_medium_value'])}):")
    for feat, reason in recommendations['keep_medium_value'][:10]:
        print(f"  - {feat}: {reason}")
    
    print(f"\n🔻 CONSIDER REMOVE ({len(recommendations['consider_remove'])}):")
    for feat, reason in recommendations['consider_remove'][:20]:
        print(f"  - {feat}: {reason}")
    
    print(f"\n❌ MUST REMOVE ({len(recommendations['must_remove'])}):")
    for feat, reason in recommendations['must_remove']:
        print(f"  - {feat}: {reason}")
    
    print(f"\n➕ SUGGESTED CROSS FEATURES:")
    for f1, f2, desc in recommendations['add_cross_features']:
        print(f"  - {f1} × {f2}: {desc}")
    
    # 保存建议
    with open(f"{OUTPUT_PATH}/recommendations.txt", 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("FEATURE RECOMMENDATIONS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"HIGH VALUE ({len(recommendations['keep_high_value'])}):\n")
        for feat, reason in recommendations['keep_high_value']:
            f.write(f"  {feat}: {reason}\n")
        
        f.write(f"\nMEDIUM VALUE ({len(recommendations['keep_medium_value'])}):\n")
        for feat, reason in recommendations['keep_medium_value']:
            f.write(f"  {feat}: {reason}\n")
        
        f.write(f"\nCONSIDER REMOVE ({len(recommendations['consider_remove'])}):\n")
        for feat, reason in recommendations['consider_remove']:
            f.write(f"  {feat}: {reason}\n")
        
        f.write(f"\nMUST REMOVE ({len(recommendations['must_remove'])}):\n")
        for feat, reason in recommendations['must_remove']:
            f.write(f"  {feat}: {reason}\n")
        
        f.write(f"\nSUGGESTED CROSS FEATURES:\n")
        for f1, f2, desc in recommendations['add_cross_features']:
            f.write(f"  {f1} × {f2}: {desc}\n")
    
    print(f"\nRecommendations saved to: {OUTPUT_PATH}/recommendations.txt")
    
    spark.stop()
    return df_results, recommendations

if __name__ == "__main__":
    main()
