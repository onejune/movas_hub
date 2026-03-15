# metrics_eval.py

from typing import List, Tuple
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F
import numpy as np
from typing import List, Tuple
import math
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_auc_and_pcoc_regression(label_pred_list):
    if not label_pred_list:
        return {"auc": float('nan'), "pcoc": float('nan')}
    
    labels, preds = zip(*label_pred_list)
    labels = np.asarray(labels, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    
    # Compute PCOC
    sum_labels = np.sum(labels)
    if sum_labels == 0:
        pcoc = float('nan')
    else:
        pcoc = np.sum(preds) / sum_labels

    # Compute AUC
    auc = scoreRegressionAUC_optimized(labels, preds)
    return auc, pcoc

def scoreRegressionAUC_optimized(label, score):
    label = np.asarray(label, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    
    if len(label) <= 1:
        return 1.0

    # Sort by score in descending order
    sorted_idx = np.argsort(score)[::-1]
    sorted_labels = label[sorted_idx]
    
    n = len(sorted_labels)
    total_pairs = n * (n - 1) // 2
    
    # Count same-label pairs
    unique_vals, counts = np.unique(sorted_labels, return_counts=True)
    same_pairs = np.sum(counts * (counts - 1) // 2)

    if same_pairs == total_pairs:
        return 1.0

    # Compute correct pairs using cumulative logic
    correct_pairs = 0
    cum_count = 0  # Count of elements processed so far (higher scores)
    cum_sum = 0    # Sum of labels processed so far

    i = 0
    while i < n:
        j = i
        while j < n and sorted_labels[j] == sorted_labels[i]:
            j += 1
        current_label = sorted_labels[i]
        group_size = j - i

        # Count how many previous labels are > current_label
        # This is sum of (number of labels > current_label in previous groups)
        num_greater = np.sum(sorted_labels[:i] > current_label)
        correct_pairs += group_size * num_greater
        i = j

    valid_pairs = total_pairs - same_pairs
    return correct_pairs / valid_pairs if valid_pairs > 0 else 1.0

def calculate_logloss(label_prediction_list: List[Tuple[float, float]]) -> float:
    """
    计算二分类 LogLoss
    
    Args:
        label_prediction_list: [(label, prediction_score), ...]
                              - label: 真实标签 (0.0 或 1.0)
                              - prediction_score: 预测为正类的概率 (0.0~1.0)
    
    Returns:
        float: LogLoss 值，越小越好
    """
    if not label_prediction_list:
        return 0.0  # 空列表返回 0
    
    total_loss = 0.0
    n_samples = len(label_prediction_list)
    
    for label, pred_prob in label_prediction_list:
        # 防止 log(0)：将概率限制在 [eps, 1-eps] 范围内
        eps = 1e-15
        pred_prob = max(min(pred_prob, 1.0 - eps), eps)
        
        # 确保标签是 0 或 1
        if label not in (0.0, 1.0):
            raise ValueError(f"Label must be 0.0 or 1.0, got {label}")
        
        # 计算单个样本的 logloss
        if label == 1.0:
            # 真实为正类：-log(预测正类概率)
            sample_loss = -math.log(pred_prob)
        else:
            # 真实为负类：-log(预测负类概率) = -log(1 - 预测正类概率)
            sample_loss = -math.log(1.0 - pred_prob)
        
        total_loss += sample_loss
    
    # 平均损失
    return total_loss / n_samples

def calculate_logloss2(label_prediction_list, eps=1e-15):
    """
    计算二分类 LogLoss
    
    :param y_true: 真实标签列表 [0, 1, 0, ...]
    :param y_pred_proba: 预测概率列表 [0.1, 0.9, 0.3, ...]
    :param eps: 防止 log(0) 的极小值
    :return: LogLoss 值
    参数:
        label_prediction_list: List of (label, prediction_score)
    """
    # 限制概率在 [eps, 1-eps] 范围内，避免 log(0)
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    
    # 计算 LogLoss
    loss = -(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
    return np.mean(loss)

def compute_auc_pcoc2(result_df):
    # 创建两个评估器，分别用于计算 AUC 和 AUPR
    auc_evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",  # 默认值，也可以是 "probability"
        labelCol="label",
        metricName="areaUnderROC"  # 计算 AUC
    )

    aupr_evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="label",
        metricName="areaUnderPR"  # 计算 AUPR
    )

    # 分别计算 AUC 和 AUPR
    test_auc = auc_evaluator.evaluate(result_df)
    test_aupr = aupr_evaluator.evaluate(result_df)
    
    # 计算 rawPrediction 和 label 的总和
    sum_raw_prediction = F.sum("rawPrediction").alias("sum_raw")
    sum_label = F.sum("label").alias("sum_label")

    # 聚合并计算 PCOC
    pcoc_df = result_df.agg(sum_raw_prediction, sum_label)
    pcoc_df = pcoc_df.withColumn("pcoc", F.col("sum_raw") / F.col("sum_label"))

    # 提取 PCOC 值
    pcoc = pcoc_df.select("pcoc").first()[0]
    return test_auc, test_aupr, pcoc

def compute_auc_pcoc(label_prediction_list):
    """
    根据 label 和 raw prediction 列表计算 AUC 和 PCOC

    参数:
        label_prediction_list: List of (label, prediction_score)

    返回:
        auc, pcoc
    """
    auc_dict = {}
    for label, score in label_prediction_list:
        score_str = str(score)
        auc_dict.setdefault(score_str, [0, 0])  # [负样本数, 正样本数]
        if float(label) == 1:
            auc_dict[score_str][1] += 1
        auc_dict[score_str][0] += 1

    predicted_ctr = []
    num_impressions = []
    num_clicks = []
    for score_str in auc_dict:
        predicted_ctr.append(float(score_str))
        num_impressions.append(auc_dict[score_str][0])
        num_clicks.append(auc_dict[score_str][1])

    auc, _ = scoreClickAUC(num_clicks, num_impressions, predicted_ctr)
    pcoc = PCOC(num_clicks, num_impressions, predicted_ctr)
    return auc, pcoc


def PCOC(num_clicks: List[int], num_impressions: List[int], predicted_ctr: List[float]) -> float:
    """
    计算 PCOC（预测点击率的加权平均 / 实际点击数）

    参数:
        num_clicks: 每组的点击数
        num_impressions: 每组的曝光数（负样本数）
        predicted_ctr: 每组的预测 CTR

    返回:
        pcoc 值
    """
    sum_pctr = 0.0
    for i in range(len(num_clicks)):
        sum_pctr += predicted_ctr[i] * num_impressions[i]
    sum_clicks = sum(num_clicks)
    return round(sum_pctr / sum_clicks, 4) if sum_clicks != 0 else 0.0


def scoreClickAUC(num_clicks: List[int], num_impressions: List[int], predicted_ctr: List[float]) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    计算 click-based AUC

    参数:
        num_clicks: 每个预测分数 bucket 中的点击数
        num_impressions: 每个 bucket 中的曝光数
        predicted_ctr: 每个 bucket 的预测CTR

    返回:
        auc 值 和 bucket 信息
    """
    i_sorted = sorted(range(len(predicted_ctr)), key=lambda i: predicted_ctr[i], reverse=True)

    auc_temp = 0.0
    click_sum = 0.0
    old_click_sum = 0.0
    no_click = 0.0
    no_click_sum = 0.0

    last_ctr = predicted_ctr[i_sorted[0]] + 1.0
    bucket = []

    for i in range(len(predicted_ctr)):
        if last_ctr != predicted_ctr[i_sorted[i]]:
            bucket.append((click_sum, no_click_sum, last_ctr))
            auc_temp += (click_sum + old_click_sum) * no_click / 2.0
            old_click_sum = click_sum
            no_click = 0.0
            last_ctr = predicted_ctr[i_sorted[i]]
        no_click += num_impressions[i_sorted[i]] - num_clicks[i_sorted[i]]
        no_click_sum += num_impressions[i_sorted[i]] - num_clicks[i_sorted[i]]
        click_sum += num_clicks[i_sorted[i]]

    auc_temp += (click_sum + old_click_sum) * no_click / 2.0

    if click_sum == 0 or no_click_sum == 0:
        auc = 1.0
    else:
        auc = round(auc_temp / (click_sum * no_click_sum), 5)

    bucket.append((click_sum, no_click_sum, last_ctr))
    return auc, bucket
