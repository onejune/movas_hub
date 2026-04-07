from collections import defaultdict

def parse_model_file(file_path):
    feature_stats = defaultdict(lambda: {"positive": 0, "negative": 0, "values": set()})

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 按 '\002' 分割行数据
            parts = line.strip().split('\002')
            if len(parts) < 3:
                continue  # 跳过格式不正确的行

            # 提取特征部分
            feature_part = parts[0]
            features = [f.split('=', 1) for f in feature_part.split('\001')]  # 分离 feature_name 和 value
            feature_dict = {f[0]: f[1] for f in features}  # 构建 feature_name 到 value 的映射

            # 提取正负样本数
            positive_count = int(parts[-2])
            negative_count = int(parts[-1])

            # 统计单个特征和组合特征
            feature_names = list(feature_dict.keys())
            for i in range(len(feature_names)):
                for j in range(i, len(feature_names)):
                    feature_name = "#".join(sorted([feature_names[i], feature_names[j]]))
                    combined_values = "#".join(sorted([feature_dict[feature_names[i]], feature_dict[feature_names[j]]]))
                    
                    # 更新统计信息
                    feature_stats[feature_name]["positive"] += positive_count
                    feature_stats[feature_name]["negative"] += negative_count
                    feature_stats[feature_name]["values"].add(combined_values)

    return feature_stats


def print_feature_stats(feature_stats):
    for feature_name, stats in feature_stats.items():
        unique_value_count = len(stats["values"])
        print(f"{feature_name}, {stats['positive']}, {stats['negative']}, {unique_value_count}")


# 示例用法
if __name__ == "__main__":
    file_path = "./train_output/base"
    stats = parse_model_file(file_path)
    print_feature_stats(stats)
