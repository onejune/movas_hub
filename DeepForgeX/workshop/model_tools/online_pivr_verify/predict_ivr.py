#从回流日志中提取特征，并调用 online model 预估 ivr，验证线上线下打分一致性
import pandas as pd
import os, sys
import itertools
import numpy as np

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
    print('get', len(feature_combinations), 'feature_combinations.')
    return feature_combinations


def apply_feature_combinations(df, feature_combinations):
    """
    根据特征组合规则，应用组合特征到 DataFrame 中。
    返回一个新的 DataFrame，包含原始特征和组合特征。
    每个样本只保留一行，组合特征拼接成一个字符串。
    """
    #print(list(df.columns))
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
            #print(n, combined_feature_name, combined_features_str)
            n += 1
        
        # 创建新行，包含原始特征和组合特征
        new_row = row.to_dict()  # 将 row 转换为字典，确保是独立副本
        new_row.update(combined_features_dict)  # 更新字典，添加组合特征
        combined_rows.append(new_row)  # 将新行添加到列表中
    
    # 将所有组合后的行转换为 DataFrame
    df_combined = pd.DataFrame(combined_rows)
    #print(list(df_combined.columns))
    return df_combined


def parse_model_file(model_file):
    """
    解析模型文件，返回一个字典，键为特征名称（格式为 name1=value1\001name2=value2\001...），值为对应的权重。
    """
    print('model:', model_file)
    feature_weights = {}
    
    with open(model_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 去除行末的换行符，并按 '\002' 分割
            parts = line.strip().split('\x02')
            
            if len(parts) != 2:
                #print(f"Warning: Invalid line format in {model_file}: {line}")
                continue
            
            feature_name, weight_str = parts
            try:
                weight = float(weight_str)
                feature_weights[feature_name] = weight
            except ValueError:
                #print(f"Warning: Invalid weight value in {model_file}: {weight_str}")
                pass
    
    return feature_weights

def figure_pivr(sample_df, model):
    score = 0
    for feature in sample_df.columns:
        if 'combined_' not in feature:
            continue
        #print(feature, sample_df[feature].iloc[0])
        feature_value_list = sample_df[feature].iloc[0].split('\003')
        for feature_value in feature_value_list:
            weight = 0.0
            if feature_value in model:
                weight = model.get(feature_value, 0.0)
                score += weight
            #print(n, feature_value, '\t', weight, '\t', score)
    return 1 / (1 + np.exp(-score))

def sample_process(df):
    print('sample count before:', df.shape)
    #print(list(df.columns))
    #df = df[(df['ts'] >= '2025-06-12 05:10:00') & (df['ts'] <= '2025-06-12 05:48:00')]
    df = df[df['abtestkey'] == 'yeahdsp_union_ftrl_purchase_ruf_v7_shein_stat']
    df = df[df['demand_pkgname'] == 'COM.ZZKKO']
    #df = df[df['ymdh'] == '2025010406']
    df['online_pivr'] = df['predictScore']
    df = df.sample(frac=0.1, random_state=42)
    
    print('sample count after:', df.shape)
    return df

def output(df_combined):
    # 保存带有 score 的数据
    output_csv = 'sample_scores.csv'
    columns = ['requestid', 'ts', 'modelVersion', 'online_pivr', 'score_1', 'score_1_diff', 'gandalfIpAddress']
    #columns = ['demand_pkgname', 'abtestkey', 'ymdh', 'modelVersion', 'online_pivr', 'score_1', 'score_1_diff', 'score_2', 'score_2_diff']
    #columns = ['demand_pkgname', 'abtestkey', 'ymdh', 'modelVersion', 'online_pivr', 'score_1', 'score_1_diff']

    df_combined[columns].to_csv(output_csv, index=False, sep='\t')
    print(f"Sample scores saved to {output_csv}")

def main():
    # 读取 Parquet 样本文件
    parquet_file = './data/2025-07-17-00/'
    df = pd.read_parquet(parquet_file)
    df = sample_process(df)
    
    # 解析特征配置文件
    config_file = './conf/combine_schema'
    feature_combinations = parse_feature_config(config_file)
    
    # 应用特征组合规则
    df_combined = apply_feature_combinations(df, feature_combinations)
    
    # 读取模型文件
    model_file = './data/online_model_2025-07-08'
    model_1 = parse_model_file(model_file)
    
    #model_file = './data/online_model_2024-12-26'
    #model_2 = parse_model_file(model_file)
    
    # 计算每个样本的 score
    df_combined['score_1'] = df_combined.apply(lambda row: figure_pivr(pd.DataFrame([row]), model_1), axis=1)
    df_combined['score_1_diff'] = round(100 * (df_combined['score_1'] / df_combined['online_pivr'] - 1), 0)
    #df_combined['score_2'] = df_combined.apply(lambda row: figure_pivr(pd.DataFrame([row]), model_2), axis=1)
    #df_combined['score_2_diff'] = round(100 * (df_combined['score_2'] / df_combined['online_pivr'] - 1), 0)
    
    output(df_combined)

if __name__ == "__main__":
    main()