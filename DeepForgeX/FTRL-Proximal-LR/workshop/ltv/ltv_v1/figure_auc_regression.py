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
    data_dict = defaultdict(lambda: {'labels': [], 'predictions': []})
    
    for line in sys.stdin:
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
    
    if not data_dict:
        print("错误: 没有提供数据!", file=sys.stderr)
        sys.exit(1)
    
    # 计算全体样本（合并所有key的数据）
    all_labels = []
    all_predictions = []
    for key_data in data_dict.values():
        all_labels.extend(key_data['labels'])
        all_predictions.extend(key_data['predictions'])
    
    # 计算所有指标
    results = []
    # 添加全体结果
    auc_all = scoreRegressionAUC(all_labels, all_predictions)
    pcoc_all = PCOC(all_labels, all_predictions)
    rmse_all, mae_all, mape_all = calculate_regression_metrics(all_labels, all_predictions)
    results.append({
        'key': 'metric-all',
        'label_sum': sum(all_labels),
        'pred_sum': sum(all_predictions),
        'auc': auc_all,
        'pcoc': pcoc_all,
        'rmse': rmse_all,
        'mae': mae_all,
        'mape': mape_all
    })
    
    # 添加各分组结果
    for key, data in data_dict.items():
        labels = data['labels']
        predictions = data['predictions']
        auc = scoreRegressionAUC(labels, predictions)
        pcoc = PCOC(labels, predictions)
        rmse, mae, mape = calculate_regression_metrics(labels, predictions)
        results.append({
            'key': key,
            'label_sum': sum(labels),
            'pred_sum': sum(predictions),
            'auc': auc,
            'pcoc': pcoc,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        })
    
    # 计算每列的最大宽度
    column_widths = {
        'key': max(len('Key'), max(len(str(r['key'])) for r in results)),
        'label_sum': max(len('label_sum'), 14),  # 调整为容纳6位小数
        'pred_sum': max(len('pred_sum'), 14),   # 调整为容纳6位小数
        'auc': max(len('AUC'), 10),             # 调整为容纳6位小数
        'pcoc': max(len('PCOC'), 10),           # 调整为容纳6位小数
        'rmse': max(len('RMSE'), 10),           # 调整为容纳6位小数
        'mae': max(len('MAE'), 10),             # 调整为容纳6位小数
        'mape': max(len('MAPE'), 10)            # 调整为容纳6位小数
    }
    
    # 计算总宽度
    total_width = sum(column_widths.values()) + 7 * 3 + 2
    
    # 打印表头
    print('+' + '-' * total_width + '+')
    print(f"| {'Key':^{column_widths['key']}} | {'SUM(lable)':^{column_widths['label_sum']}} | {'SUM(pred)':^{column_widths['pred_sum']}} | {'AUC':^{column_widths['auc']}} | {'PCOC':^{column_widths['pcoc']}} | {'RMSE':^{column_widths['rmse']}} | {'MAE':^{column_widths['mae']}} | {'MAPE':^{column_widths['mape']}} |")
    print('+' + '-' * total_width + '+')
    
    # 打印数据行和行分割线
    for i, result in enumerate(results):
        print(f"| {result['key']:{column_widths['key']}} | {result['label_sum']:^{column_widths['label_sum']}.6f} | {result['pred_sum']:^{column_widths['pred_sum']}.6f} | {result['auc']:^{column_widths['auc']}.6f} | {result['pcoc']:^{column_widths['pcoc']}.6f} | {result['rmse']:^{column_widths['rmse']}.6f} | {result['mae']:^{column_widths['mae']}.6f} | {result['mape']:^{column_widths['mape']}.6f} |")
        if i < len(results) - 1:  # 最后一行不打印分割线
            print('+' + '-' * total_width + '+')
    
    # 打印表尾
    print('+' + '-' * total_width + '+')

if __name__ == "__main__":
    main()