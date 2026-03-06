import csv
from collections import defaultdict

def parse_model_file(file_path):
    """
    解析模型文件，返回一个字典 {feature_name: weight}
    """
    feature_weights = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去除行末的换行符，并按 '\002' 分割
            parts = line.strip().split('\x02')

            feature_name, weight_str, _, pos, neg = parts
            try:
                weight = float(weight_str)
                pos = int(pos)
                neg = int(neg)
                feature_weights[feature_name] = [weight, pos, neg]
            except ValueError:
                print(f"Warning: Invalid weight value in {file_path}: {weight_str}")
    
    return feature_weights

def compare_weights(model1, model2):
    """
    比较两个模型文件中的权重差异，处理只存在于一个文件中的特征，并过滤掉 diff 为 0 的特征
    """
    diff_results = []
    
    # 创建一个集合来跟踪所有出现过的 feature_name
    all_features = set(model1.keys()).union(set(model2.keys()))
    
    for feature_name in all_features:
        info1 = model1.get(feature_name, [0.0000001, 0, 0])  # 如果 feature_name 不在 model1 中，默认 weight 为 0
        info2 = model2.get(feature_name, [0.0000001, 0, 0])  # 如果 feature_name 不在 model2 中，默认 weight 为 0
        diff = abs(info1[0] - info2[0])*100/info2[0]
        
        # 只保留 diff 不为 0 的特征
        if diff > 10000:
            diff_results.append((feature_name, info1[0], info2[0], diff, info1[1], info2[1], info1[2], info2[2]))
    
    # 按 diff 降序排序
    diff_results.sort(key=lambda x: x[5], reverse=False)
    
    return diff_results

def save_diff_results(diff_results, output_file):
    """
    将差异结果保存到 CSV 文件中
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['feature_name', 'weight_model1', 'weight_model2', 'diff', 'pos_model1', 'pos_model2', 'neg_model1', 'neg_model2'])
        writer.writerows(diff_results)
    print(f"Difference results saved to {output_file}")

if __name__ == "__main__":
    # 文件路径
    model1_file = '/home/ec2-user/SageMaker/movas/ruf/validation_v1/train_output/model_file_2025-06-02.filt'
    model2_file = '/home/ec2-user/SageMaker/movas/ruf/validation_v1/train_output/model_file_2025-06-04.filt'
    output_file = 'weight_diff.csv'

    # 解析两个模型文件
    model1 = parse_model_file(model1_file)
    model2 = parse_model_file(model2_file)

    # 比较权重差异
    diff_results = compare_weights(model1, model2)

    # 保存差异结果到 CSV 文件
    save_diff_results(diff_results, output_file)