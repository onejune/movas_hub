import pandas as pd
import os, sys
import itertools
import numpy as np
from datetime import datetime
import json

def parse_feature_config(config_file):
    """
    解析特征配置文件，返回一个列表，每个元素是一个特征组合规则。
    每个特征组合规则是一个元组 (combined_feature_name, feature_list)，
    其中 combined_feature_name 是组合后的特征名称，feature_list 是参与组合的单特征列表。
    """
    feature_combinations = []
    with open(config_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            # 去除行末的换行符，并按 '#' 分割
            features = line.strip().split('#')
            
            # 如果该行为空或无效，跳过
            if not features:
                continue
            
            combined_feature_name = f"combined_{'#'.join(features)}"
            feature_combinations.append((combined_feature_name, features))
    print(f'get {len(feature_combinations)} feature_combinations.')
    return feature_combinations


def apply_feature_combinations(df, feature_combinations):
    """
    根据特征组合规则，应用组合特征到 DataFrame 中。
    返回一个新的 DataFrame，包含原始特征和组合特征。
    每个样本只保留一行，组合特征拼接成一个字符串。
    """
    print(f'processing {len(df)} samples with feature combinations...')
    combined_rows = []
    for idx, row in df.iterrows():
        # 对每个样本进行处理
        feature_values = {}
        combined_features_dict = {}
        
        n = 0
        for combined_feature_name, features in feature_combinations:
            # 检查所有参与组合的特征是否存在于 DataFrame 中
            missing_features = [feat for feat in features if feat not in df.columns]
            if missing_features:
                print(f"Warning: The following features are missing in the Parquet file: {missing_features}")
                continue
            
            # 将每个特征的值拆分成列表
            for feat in features:
                if feat not in feature_values:
                    # 如果特征值包含多个部分，拆分成列表
                    feature_value = str(row[feat])
                    feature_values[feat] = feature_value.split('\001') if '\001' in feature_value else [feature_value]
            
            # 对所有特征值进行笛卡尔积
            combinations = list(itertools.product(*[feature_values[feat] for feat in features]))
            
            # 将每个组合拼接成 name1=value1\001name2=value2\001... 格式的字符串
            combined_features = [
                '\001'.join([f"{feat}={val}" for feat, val in zip(features, combo)]) 
                for combo in combinations
            ]
            
            # 将所有组合特征拼接成一个字符串，并存储在字典中
            combined_features_str = '\003'.join(combined_features)  # 使用 \003 作为分隔符
            combined_features_dict[combined_feature_name] = combined_features_str
            n += 1
        
        # 创建新行，包含原始特征和组合特征
        new_row = row.to_dict()  # 将 row 转换为字典，确保是独立副本
        new_row.update(combined_features_dict)  # 更新字典，添加组合特征
        combined_rows.append(new_row)  # 将新行添加到列表中
    
    # 将所有组合后的行转换为 DataFrame
    df_combined = pd.DataFrame(combined_rows)
    print(f'feature combination processing completed')
    return df_combined


def parse_model_file(model_file):
    """
    解析模型文件，返回一个字典，键为特征名称（格式为 name1=value1\001name2=value2\001...），值为对应的权重。
    """
    print(f'model: {model_file}')
    feature_weights = {}
    
    with open(model_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 去除行末的换行符，并按 '\002' 分割
            parts = line.strip().split('\x02')
            
            if len(parts) != 2:
                continue
            
            feature_name, weight_str = parts
            try:
                weight = float(weight_str)
                feature_weights[feature_name] = weight
            except ValueError:
                pass
    
    print(f'loaded {len(feature_weights)} feature weights')
    return feature_weights

def figure_pivr(sample_df, model):
    score = 0
    for feature in sample_df.columns:
        if 'combined_' not in feature:
            continue
        feature_value_list = sample_df[feature].iloc[0].split('\003')
        for feature_value in feature_value_list:
            weight = model.get(feature_value, 0.0)
            score += weight
    return 1 / (1 + np.exp(-score))

def extract_price_from_json(extra_info_str):
    """
    从extraInfo JSON字符串中提取iDspPayout字段
    """
    try:
        extra_info = json.loads(extra_info_str)
        f = extra_info.get('iDspPayout', 0.0)
        if f == 0.0:
            f = extra_info.get('statPayout', 0.0)
        return f
    except (json.JSONDecodeError, TypeError):
        return 0.0
def calculate_pkgname_stats(df_combined, pivr_col, pkg_col='demand_pkgname', price_col='price'):
    """
    计算每个demand_pkgname维度下的pivr和pcpm的平均值和方差
    """
    df_combined['pcpm'] = df_combined[pivr_col] * df_combined[price_col] * 1000
    
    stats = df_combined.groupby(pkg_col).agg({
        pivr_col: ['mean', 'var', 'count'],
        'pcpm': ['mean', 'var']
    }).round(6)
    
    # 重命名列
    stats.columns = [
        f'{pivr_col}_mean', f'{pivr_col}_var', f'{pivr_col}_count',
        'pcpm_mean', 'pcpm_var'
    ]
    stats = stats.reset_index()
    
    return stats

def compare_model_scores(df_combined1, df_combined2, pkg_col='demand_pkgname'):
    """
    对比两个模型在demand_pkgname维度上的pivr和pcpm统计信息
    """
    print('calculating model comparison statistics...')
    
    # 计算模型1的统计信息
    stats1 = calculate_pkgname_stats(df_combined1, 'pivr_1', pkg_col)
    stats1 = stats1.rename(columns={
        'pivr_1_mean': 'model1_pivr_mean', 
        'pivr_1_var': 'model1_pivr_var', 
        'pivr_1_count': 'model1_count',
        'pcpm_mean': 'model1_pcpm_mean',
        'pcpm_var': 'model1_pcpm_var'
    })
    
    # 计算模型2的统计信息
    stats2 = calculate_pkgname_stats(df_combined2, 'pivr_2', pkg_col)
    stats2 = stats2.rename(columns={
        'pivr_2_mean': 'model2_pivr_mean', 
        'pivr_2_var': 'model2_pivr_var', 
        'pivr_2_count': 'model2_count',
        'pcpm_mean': 'model2_pcpm_mean',
        'pcpm_var': 'model2_pcpm_var'
    })
    
    # 合并统计结果
    comparison = pd.merge(stats1, stats2, on=pkg_col, how='outer')
    
    # 计算差异和变化率
    comparison['pivr_mean_diff'] = comparison['model1_pivr_mean'] - comparison['model2_pivr_mean']
    comparison['pivr_mean_change_rate'] = (
        (comparison['model1_pivr_mean'] - comparison['model2_pivr_mean']) / 
        comparison['model2_pivr_mean'] * 100
    ).round(2)
    
    comparison['pcpm_mean_diff'] = comparison['model1_pcpm_mean'] - comparison['model2_pcpm_mean']
    comparison['pcpm_mean_change_rate'] = (
        (comparison['model1_pcpm_mean'] - comparison['model2_pcpm_mean']) / 
        comparison['model2_pcpm_mean'] * 100
    ).round(2)
    
    # 计算总样本数
    comparison['total_samples'] = comparison[['model1_count', 'model2_count']].max(axis=1)
    
    # 按样本数量降序排序
    comparison = comparison.sort_values(by='total_samples', ascending=False)
    
    return comparison

def output_comparison(comparison):
    output_file = f'model_comparison_by_pkgname_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    comparison.to_csv(output_file, index=False, sep='\t')
    print(f'Model comparison result saved to {output_file}')
    
    print("\n" + "="*120)
    print("Model Comparison Analysis Report")
    print("="*120)
    
    # 概览统计
    total_apps = len(comparison)
    apps_with_pivr_changes = len(comparison[comparison['pivr_mean_change_rate'] != 0])
    apps_with_pcpm_changes = len(comparison[comparison['pcpm_mean_change_rate'] != 0])
    avg_pivr_change_rate = comparison['pivr_mean_change_rate'].mean()
    avg_pcpm_change_rate = comparison['pcpm_mean_change_rate'].mean()
    
    print(f"Total apps: {total_apps}")
    print(f"Apps with pivr changes: {apps_with_pivr_changes}")
    print(f"Apps with pcpm changes: {apps_with_pcpm_changes}")
    print(f"Avg pivr change rate: {avg_pivr_change_rate:.2f}%")
    print(f"Avg pcpm change rate: {avg_pcpm_change_rate:.2f}%")
    
    print("\nPivR Comparison by App (sorted by sample count):")
    print("-"*120)
    print(f"{'App Package Name':<30} {'V1 PivR Mean':<15} {'V2 PivR Mean':<15} {'Diff':<12} {'Change Rate(%)':<15} {'Samples':<10}")
    print("-"*120)
    
    for idx, row in comparison.head(20).iterrows():
        app_name = str(row['demand_pkgname'])[:28]  # 限制显示长度
        v1_pivr_mean = f"{row['model1_pivr_mean']:.6f}"
        v2_pivr_mean = f"{row['model2_pivr_mean']:.6f}"
        diff = f"{row['pivr_mean_diff']:.6f}"
        change_rate = f"{row['pivr_mean_change_rate']:.2f}"
        count = f"{int(row['total_samples']):,}"
        
        print(f"{app_name:<30} {v1_pivr_mean:<15} {v2_pivr_mean:<15} {diff:<12} {change_rate:<15} {count:<10}")
    
    if len(comparison) > 20:
        print(f"\n... {len(comparison) - 20} more apps not shown")
    
    print("\nPCPM Comparison by App (sorted by sample count):")
    print("-"*120)
    print(f"{'App Package Name':<30} {'V1 PCPM Mean':<15} {'V2 PCPM Mean':<15} {'Diff':<12} {'Change Rate(%)':<15} {'Samples':<10}")
    print("-"*120)
    
    for idx, row in comparison.head(20).iterrows():
        app_name = str(row['demand_pkgname'])[:28]  # 限制显示长度
        v1_pcpm_mean = f"{row['model1_pcpm_mean']:.6f}"
        v2_pcpm_mean = f"{row['model2_pcpm_mean']:.6f}"
        diff = f"{row['pcpm_mean_diff']:.6f}"
        change_rate = f"{row['pcpm_mean_change_rate']:.2f}"
        count = f"{int(row['total_samples']):,}"
        
        print(f"{app_name:<30} {v1_pcpm_mean:<15} {v2_pcpm_mean:<15} {diff:<12} {change_rate:<15} {count:<10}")
    
    if len(comparison) > 20:
        print(f"\n... {len(comparison) - 20} more apps not shown")
    
    # 显示样本数量统计
    print("\nSample Count Statistics:")
    print("-"*60)
    print(f"{'App Package Name':<30} {'Total Samples':<15}")
    print("-"*60)
    
    for idx, row in comparison.head(20).iterrows():
        app_name = str(row['demand_pkgname'])[:28]  # 限制显示长度
        count = f"{int(row['total_samples']):,}"
        print(f"{app_name:<30} {count:<15}")
    
    if len(comparison) > 20:
        print(f"\n... {len(comparison) - 20} more apps not shown")
    
    # 找出样本数量最多的应用
    if len(comparison) > 0:
        max_samples_app = comparison.iloc[0]
        print(f"\nLargest sample count app: {max_samples_app['demand_pkgname']}")
        print(f"   Sample count: {int(max_samples_app['total_samples']):,}")
        print(f"   V1 pivr mean: {max_samples_app['model1_pivr_mean']:.6f}, V2 pivr mean: {max_samples_app['model2_pivr_mean']:.6f}")
        print(f"   pivr change rate: {max_samples_app['pivr_mean_change_rate']:.2f}%")
        print(f"   V1 pcpm mean: {max_samples_app['model1_pcpm_mean']:.6f}, V2 pcpm mean: {max_samples_app['model2_pcpm_mean']:.6f}")
        print(f"   pcpm change rate: {max_samples_app['pcpm_mean_change_rate']:.2f}%")
    
    print("="*120)

def sample_process(df):
    print(f'sample count before: {df.shape[0]:,}')
    df['online_pivr'] = df['predictScore']
    # 添加price列
    df['price'] = df['extraInfo'].apply(extract_price_from_json)
    #df = df[(df['ts'] >= '2025-06-12 05:10:00') & (df['ts'] <= '2025-06-12 05:48:00')]
    #df = df[df['abtestkey'] == 'yeahdsp_union_ftrl_purchase_ruf_v7_shein_stat']
    df = df[df['business_type'] == 'aecps']
    #df = df[df['ymdh'] == '2025010406']
    df = df.sample(frac=0.01, random_state=42)
    print(f'sample count after: {df.shape[0]:,} (sampling rate: 1%)')
    return df

def main():
    print("IVR Model Comparison Analysis Tool")
    print("="*50)
    
    # 读取 Parquet 样本文件
    print("Reading data...")
    parquet_file = '/mnt/data/oss_dsp_algo/feature_flowback/part=2025-10-22/hour=06/'
    df = pd.read_parquet(parquet_file)
    df = sample_process(df)
    
    # 加载模型配置
    print("\nLoading model configuration...")
    model_path1 = '/mnt/data/oss_dsp_algo/ivr/model/ivr3_tsbf_v6'
    feature_combinations = parse_feature_config(model_path1 + '/conf/combine_schema')
    df_combined_1 = apply_feature_combinations(df, feature_combinations)
    
    #df_combined_2 = df_combined_1
    model_path2 = '/mnt/data/oss_dsp_algo/ivr/model/ivr3_tsbf_v6'
    feature_combinations = parse_feature_config(model_path2 + '/conf/combine_schema')
    df_combined_2 = apply_feature_combinations(df, feature_combinations)

    print("\nLoading model files...")
    model_file1 = model_path1 + '/online_model_2025-10-13'
    model_1 = parse_model_file(model_file1)
    
    model_file2 = model_path2 + '/online_model_2025-10-14'
    model_2 = parse_model_file(model_file2)
    
    print("\nCalculating prediction scores...")
    # 计算每个样本的 pivr
    df_combined_1['pivr_1'] = df_combined_1.apply(lambda row: figure_pivr(pd.DataFrame([row]), model_1), axis=1)
    df_combined_2['pivr_2'] = df_combined_2.apply(lambda row: figure_pivr(pd.DataFrame([row]), model_2), axis=1)
    
    # 合并两个模型的打分结果
    df_merged = df_combined_1[['requestid', 'demand_pkgname', 'pivr_1']].copy()
    df_merged = df_merged.merge(df_combined_2[['requestid', 'pivr_2']], on='requestid', how='inner')
    
    # 对比两个模型在demand_pkgname维度上的打分统计
    comparison = compare_model_scores(df_combined_1, df_combined_2)
    
    # 输出对比结果
    output_comparison(comparison)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()



