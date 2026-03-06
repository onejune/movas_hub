import sys
import numpy as np
from collections import defaultdict

def scoreRegressionAUC(label, score):
    label = np.asarray(label)
    score = np.asarray(score)
    
    sorted_indices = np.argsort(score)[::-1]  
    sorted_labels = label[sorted_indices]
    
    def merge_sort_count(lst):
        if len(lst) <= 1:
            return lst, 0, 0
        
        mid = len(lst) // 2
        left, left_correct, left_same = merge_sort_count(lst[:mid])
        right, right_correct, right_same = merge_sort_count(lst[mid:])
        
        merged, split_correct, split_same = merge_count(left, right)
        
        return merged, left_correct + right_correct + split_correct, left_same + right_same + split_same
    
    def merge_count(left, right):
        merged = []
        i = j = 0
        correct = 0
        same = 0
        len_left = len(left)
        
        while i < len(left) and j < len(right):
            if left[i] > right[j]:
                merged.append(left[i])
                correct += len(right) - j  
                i += 1
            elif left[i] < right[j]:
                merged.append(right[j])
                j += 1
            else:
                same_count_left = 1
                same_count_right = 1
                
                while i + 1 < len(left) and left[i] == left[i+1]:
                    same_count_left += 1
                    i += 1
                
                while j + 1 < len(right) and right[j] == right[j+1]:
                    same_count_right += 1
                    j += 1
                
                same += same_count_left * same_count_right
                correct += same_count_left * (len(right) - j)
                
                merged.extend([left[i]] * same_count_left)
                merged.extend([right[j]] * same_count_right)
                
                i += 1
                j += 1
        
        merged.extend(left[i:])
        merged.extend(right[j:])
        
        return merged, correct, same
    
    _, correct_pairs, same_pairs = merge_sort_count(sorted_labels.tolist())
    n = len(sorted_labels)
    total_pairs = n * (n - 1) // 2  
    
    if total_pairs == same_pairs:  
        return 1.0
    
    valid_pairs = total_pairs - same_pairs
    ranking_accuracy = correct_pairs / valid_pairs
    
    return ranking_accuracy

def PCOC(labels, predictions):
    if not labels or not predictions:
        return 0.0
    
    assert len(labels) == len(predictions), "标签和预测值数量必须相同"
    
    sum_predictions = sum(predictions)
    sum_labels = sum(labels)
    
    if sum_labels == 0:
        return float('nan')  
    
    return sum_predictions / sum_labels

def calculate_regression_metrics(labels, predictions):
    if not labels or not predictions:
        return (0.0, 0.0, 0.0)
    
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    
    errors = predictions - labels
    
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    
    non_zero_labels = labels[labels != 0]
    non_zero_predictions = predictions[labels != 0]
    
    if len(non_zero_labels) == 0:
        mape = 0.0
    else:
        mape = np.mean(np.abs((non_zero_predictions - non_zero_labels) / non_zero_labels)) * 100.0
    
    return (rmse, mae, mape)

def main():
    if len(sys.argv) != 3:
        print(f"用法: python {sys.argv[0]} <输入文件> <输出文件>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # 从文件名提取DATE
    dtm = 'date'
    try:
        file_name = input_file.split('/')[-1]
        dtm = '_'.join(file_name.split('_')[1:-3])
    except:
        pass

    data_dict = defaultdict(lambda: {'labels': [], 'predictions': []})

    try:
        with open(input_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        key = parts[0]
                        label = float(parts[1])
                        pred = float(parts[2])
                        data_dict[key]['labels'].append(label)
                        data_dict[key]['predictions'].append(pred)
                    except ValueError:
                        print(f"警告: 无法解析行: {line.strip()}", file=sys.stderr)
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}", file=sys.stderr)
        sys.exit(1)

    if not data_dict:
        print("错误: 没有提供数据!", file=sys.stderr)
        sys.exit(1)

    # 计算全体样本
    all_labels = []
    all_predictions = []
    for key_data in data_dict.values():
        all_labels.extend(key_data['labels'])
        all_predictions.extend(key_data['predictions'])
    
    # 计算sum(pred)/sum(label)
    sum_labels = sum(all_labels)
    sum_predictions = sum(all_predictions)
    
    if sum_labels == 0:
        print("错误: 标签总和为零，无法计算比例", file=sys.stderr)
        sys.exit(1)
    
    ratio = sum_predictions / sum_labels
    print(f"预测值总和/标签总和 = {ratio:.4f}")
    
    # 如果比例小于0.6，则调整所有预测值
    if ratio < 0.6:
        print("比例小于0.6，调整预测值")
        for key_data in data_dict.values():
            key_data['predictions'] = [pred / (1 - pred) for pred in key_data['predictions']]
        
        # 更新全体预测值
        all_predictions = []
        for key_data in data_dict.values():
            all_predictions.extend(key_data['predictions'])
        
        # 重新计算调整后的比例
        sum_predictions = sum(all_predictions)
        new_ratio = sum_predictions / sum_labels
        print(f"调整后预测值总和/标签总和 = {new_ratio:.4f}")

    # 计算所有指标
    results = []
    # 添加全体结果
    auc_all = scoreRegressionAUC(all_labels, all_predictions)
    pcoc_all = PCOC(all_labels, all_predictions)
    rmse_all, mae_all, _ = calculate_regression_metrics(all_labels, all_predictions)  # 忽略MAPE
    results.append({
        'date': dtm,
        'key': 'metric-all',
        'label_sum': sum(all_labels),
        'sample_count': len(all_labels),
        'auc': auc_all,
        'pcoc': pcoc_all,
        'rmse': rmse_all,
        'mae': mae_all
    })

    # 添加各分组结果
    for key, data in data_dict.items():
        labels = data['labels']
        predictions = data['predictions']
        auc = scoreRegressionAUC(labels, predictions)
        pcoc = PCOC(labels, predictions)
        rmse, mae, _ = calculate_regression_metrics(labels, predictions)  # 忽略MAPE
        results.append({
            'date': dtm,
            'key': key,
            'label_sum': sum(labels),
            'sample_count': len(labels),
            'auc': auc,
            'pcoc': pcoc,
            'rmse': rmse,
            'mae': mae
        })

    # 按label_sum降序排序
    results.sort(key=lambda x: x['sample_count'], reverse=True)

    # 计算每列的最大宽度
    column_widths = {
        'date': max(len('DATE'), max(len(str(r['date'])) for r in results)),
        'key': max(len('Key'), max(len(str(r['key'])) for r in results)),
        'label_sum': max(len('SUM(label)'), 12),
        'sample_count': max(len('Count'), 8),
        'auc': max(len('AUC'), 10),
        'pcoc': max(len('PCOC'), 10),
        'rmse': max(len('RMSE'), 10),
        'mae': max(len('MAE'), 10)
    }

    total_width = sum(column_widths.values()) + 8 * 3 - 1

    # 生成表格内容
    table_content = []
    table_content.append('+' + '-' * total_width + '+')
    table_content.append(f"| {'DATE':^{column_widths['date']}} | {'Key':^{column_widths['key']}} | {'SUM(label)':^{column_widths['label_sum']}} | {'Count':^{column_widths['sample_count']}} | {'AUC':^{column_widths['auc']}} | {'PCOC':^{column_widths['pcoc']}} | {'RMSE':^{column_widths['rmse']}} | {'MAE':^{column_widths['mae']}} |")
    table_content.append('+' + '-' * total_width + '+')

    for i, result in enumerate(results):
        table_content.append(
            f"| {result['date']:{column_widths['date']}} "
            f"| {result['key']:{column_widths['key']}} "
            f"| {result['label_sum']:^{column_widths['label_sum']}.4f} "
            f"| {result['sample_count']:^{column_widths['sample_count']}} "
            f"| {result['auc']:^{column_widths['auc']}.4f} "
            f"| {result['pcoc']:^{column_widths['pcoc']}.4f} "
            f"| {result['rmse']:^{column_widths['rmse']}.4f} "
            f"| {result['mae']:^{column_widths['mae']}.4f} |"
        )
        if i < len(results) - 1:
            table_content.append('+' + '-' * total_width + '+')

    table_content.append('+' + '-' * total_width + '+')

    # 写入文件
    try:
        with open(output_file, 'w') as f:
            f.write('\n'.join(table_content))
        print(f"结果已成功写入 {output_file}")
    except Exception as e:
        print(f"错误: 写入文件失败 - {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()