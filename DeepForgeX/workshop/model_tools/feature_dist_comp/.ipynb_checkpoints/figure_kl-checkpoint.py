import os, sys
import pandas as pd
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import chisquare, entropy

file1 = '/home/ec2-user/SageMaker/movas/ruf/parquet/sample_2025-06-02/'
file2 = '/home/ec2-user/SageMaker/movas/ruf/parquet/sample_2025-06-04/'

combine_schema_path = '/home/ec2-user/SageMaker/movas/ruf/ruf_v7/conf/combine_schema'
sparse_features = []

def load_features():
    global sparse_features
    # 所有的单特征
    single_features = set()

    with open(combine_schema_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            # 移除行尾的换行符并按 '#' 分割
            feature_combination = line.strip().split('#')
            # 添加每个单特征到集合中
            single_features.update(feature_combination)
    sparse_features = list(single_features) + ['purchase', 'abtestkey']
    print('sparse_features:%s' % len(sparse_features))

def load_sample_parquet(sample_path):
    parquet_files_dir = sample_path
    #print('parquet_files_dir:', parquet_files_dir)
    all_data = []
    for root, dirs, files in os.walk(parquet_files_dir):
        #print(dirs, files)
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                data = pd.read_parquet(file_path, columns = sparse_features)
                all_data.append(data)
    data = pd.concat(all_data, ignore_index=True)
    data = data.astype(str)
    data = data[data['demand_pkgname'] == 'COM.ZZKKO']
    #data = data[(data['abtestkey'] == yeahdsp_union_ftrl_purchase_clk_v2_shein_stat') | 
    #        (data['abtestkey'] == 'YEAHDSP_UNION_FTRL_PURCHASE_CLK_V2_SHEIN_STAT')]

    #保留正样本
    data = data[data['purchase'] == '1']
    return data

load_features()
# 加载数据
df1 = load_sample_parquet(file1)
df2 = load_sample_parquet(file2)

columns_to_compare = sparse_features
# 确保两份数据有相同的列
if not all(col in df1.columns and col in df2.columns for col in columns_to_compare):
    raise ValueError("存在不在两个文件中的特征")

# 初始化一个字典来存储每个特征的分布信息
distribution_info = defaultdict(dict)
# 初始化一个字典来存储每个特征的分布差异
distribution_diffs = {}

# 遍历每一列（特征）
for column in columns_to_compare:
    # 计算每个特征值的数量和总数量
    count1 = df1[column].value_counts().sort_index()
    total1 = len(df1)
    count2 = df2[column].value_counts().sort_index()
    total2 = len(df2)

    # 如果两个分布有不同的索引（即不同的特征值），我们需要对齐它们
    all_values = sorted(set(count1.index) | set(count2.index))
    count1 = count1.reindex(all_values, fill_value=0)
    count2 = count2.reindex(all_values, fill_value=0)

    # 将计数转换为占比
    ratio1 = count1 * 100 / total1
    ratio2 = count2 * 100 / total2
    ratio2 = ratio2 + 1e-10
    kl_divergence = round(entropy(ratio1, qk=ratio2), 6)
    ratio1 = round(ratio1, 6)
    ratio2 = round(ratio2, 6)

    # 将两个分布合并成一个DataFrame，并按占比降序排序
    combined_dist = pd.DataFrame({
        'Value': all_values,
        'File 1 Ratio': ratio1.values,
        'File 2 Ratio': ratio2.values,
        'File 1 Cnt' : count1.values,
        'File 2 Cnt' : count2.values
    }).sort_values(by=['File 1 Ratio', 'File 2 Ratio'], ascending=False, key=lambda x: x.abs())

    # 打印每个特征的取值数量占比，使用格式化字符串确保左对齐
    print(f"Feature: {column}, File1 total: {total1}, File2 total: {total2}")
    header = f"{'feature':<20} {'File1 freq':<20} {'File2 freq':<20} {'File1 cnt':<20} {'File2 cnt':<20}"
    print(header)
    print('-' * len(header))  # 分割线
    n = 0
    for _, row in combined_dist.iterrows():
        formatted_row = f"{str(row['Value']):<20} {row['File 1 Ratio']:<20} {row['File 2 Ratio']:<20} {row['File 1 Cnt']:<20} {row['File 2 Cnt']:<20}"
        print(formatted_row)
        if n >= 20:
            break
        n += 1
    print()  # 空行分隔不同特征

    # 存储分布信息
    distribution_info[column]['combined'] = combined_dist
    distribution_info[column]['kl'] = kl_divergence

print('-' * 30, 'KL result', '-' * 30)
sorted_dict = sorted(distribution_info.items(), key=lambda item: item[1]['kl'], reverse = True)

for ele in sorted_dict:
    feature = ele[0]
    score = ele[1]['kl']
    print(f"Feature: {feature:<40}", f"  KL: {score:<20}")
    
