import sys
import numpy as np

def scoreRegressionAUC(label, score):
    """
    计算回归任务的排序AUC（基于逆序对比例）
    归并版本
    参数:
    labels (list): 真实值列表（连续值）
    predictions (list): 预测值列表
    
    返回:
    float: 排序AUC值
    """
    # 确保输入是numpy数组
    # print("calculate regression AUC...")
    label = np.asarray(label)
    score = np.asarray(score)
    
    # 按预测值排序
    sorted_indices = np.argsort(score)[::-1]  # 降序排列
    sorted_labels = label[sorted_indices]
    
    # 使用归并排序计算顺序对、逆序对和相同对
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
                correct += len(right) - j  # 当前元素与右侧所有剩余元素形成顺序对
                i += 1
            elif left[i] < right[j]:
                merged.append(right[j])
                j += 1
            else:
                # 相等的情况
                same_count_left = 1
                same_count_right = 1
                
                # 计算左侧连续相同元素数量
                while i + 1 < len(left) and left[i] == left[i+1]:
                    same_count_left += 1
                    i += 1
                
                # 计算右侧连续相同元素数量
                while j + 1 < len(right) and right[j] == right[j+1]:
                    same_count_right += 1
                    j += 1
                
                # 相同元素对数量
                same += same_count_left * same_count_right
                
                # 顺序对数量
                correct += same_count_left * (len(right) - j)
                
                # 将这些相同元素添加到合并数组
                merged.extend([left[i]] * same_count_left)
                merged.extend([right[j]] * same_count_right)
                
                i += 1
                j += 1
        
        # 添加剩余元素
        merged.extend(left[i:])
        merged.extend(right[j:])
        
        return merged, correct, same
    
    _, correct_pairs, same_pairs = merge_sort_count(sorted_labels.tolist())
    n = len(sorted_labels)
    total_pairs = n * (n - 1) // 2  # 总对数
    
    # 计算排序准确率
    if total_pairs == same_pairs:  # 所有标签都相同的情况
        return 1.0
    
    # 排除相同标签对的影响
    valid_pairs = total_pairs - same_pairs
    ranking_accuracy = correct_pairs / valid_pairs
    
    return ranking_accuracy

    """
    计算回归任务的排序AUC（基于逆序对比例）
    非归并版本
    参数:
    labels (list): 真实值列表（连续值）
    predictions (list): 预测值列表
    
    返回:
    float: 排序AUC值
    """
    # # 确保输入是numpy数组
    # label = np.asarray(label)
    # score = np.asarray(score)
    
    # # 按预测值排序
    # sorted_indices = np.argsort(score)[::-1]  # 降序排列
    # sorted_labels = label[sorted_indices]
    
    # # 计算逆序对数量
    # n = len(sorted_labels)
    # total_pairs = n * (n - 1) // 2  # 总对数
    
    # # 处理相同标签的情况
    # same_pairs = 0
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         if sorted_labels[i] == sorted_labels[j]:
    #             same_pairs += 1
    
    # # 计算顺序对数量（正确排序的对数）
    # correct_pairs = 0
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         if sorted_labels[i] > sorted_labels[j]:
    #             correct_pairs += 1
    
    # # 计算排序准确率（类似于AUC）
    # if total_pairs == same_pairs:  # 所有标签都相同的情况
    #     return 1.0
    
    # # 排除相同标签对的影响
    # valid_pairs = total_pairs - same_pairs
    # ranking_accuracy = correct_pairs / valid_pairs
    
    # return ranking_accuracy


def PCOC(labels, predictions):
    """
    计算回归任务的PCOC（预测值与真实值的比例）
    
    参数:
    labels (list): 真实值列表
    predictions (list): 预测值列表
    
    返回:
    float: PCOC值
    """
    if not labels or not predictions:
        return 0.0
    
    # 确保输入长度相同
    assert len(labels) == len(predictions), "标签和预测值数量必须相同"
    
    # 计算预测值总和与真实值总和的比例
    sum_predictions = sum(predictions)
    sum_labels = sum(labels)
    
    if sum_labels == 0:
        return float('nan')  # 避免除以零
    
    return sum_predictions / sum_labels



def calculate_regression_metrics(labels, predictions):
    """
    计算回归任务的RMSE, MAE和MAPE指标
    
    参数:
    labels (list): 真实值列表
    predictions (list): 预测值列表
    
    返回:
    tuple: (rmse, mae, mape)
    """
    if not labels or not predictions:
        return (0.0, 0.0, 0.0)
    
    # 确保输入是numpy数组
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    
    # 计算误差
    errors = predictions - labels
    
    # 计算RMSE
    rmse = np.sqrt(np.mean(errors ** 2))
    
    # 计算MAE
    mae = np.mean(np.abs(errors))
    
    # 计算MAPE (避免除以零)
    non_zero_labels = labels[labels != 0]
    non_zero_predictions = predictions[labels != 0]
    
    if len(non_zero_labels) == 0:
        mape = 0.0
    else:
        mape = np.mean(np.abs((non_zero_predictions - non_zero_labels) / non_zero_labels)) * 100.0
    
    return (rmse, mae, mape)



def main():
    # 从标准输入读取数据
    labels = []
    predictions = []
    
    for line in sys.stdin:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                label = float(parts[0])
                pred = float(parts[1])
                labels.append(label)
                predictions.append(pred)
            except ValueError:
                print(f"警告: 无法解析行: {line.strip()}", file=sys.stderr)
    
    # 检查是否有数据
    if not labels or not predictions:
        print("错误: 没有提供数据!", file=sys.stderr)
        sys.exit(1)
    
    # 计算评估指标
    auc = scoreRegressionAUC(labels, predictions)
    pcoc = PCOC(labels, predictions)
    rmse, mae, mape = calculate_regression_metrics(labels, predictions)

    # 输出结果
    # print()
    # print('*' * 20, 'Regression Validation Result', '*' * 20)
    # print(f'Total samples: {len(labels)}')
    # print(f'Labels sum: {sum(labels):.4f}, Predictions sum: {sum(predictions):.4f}')
    # print(f'Regression AUC: {auc:.4f}, PCOC: {pcoc:.4f}')
    print('*' * 60)
    print(f"AUC: {auc:.4f} PCOC: {pcoc:.4f} RMSE: {rmse:.4f}")
    print('*' * 60) 

if __name__ == "__main__":
    main()    