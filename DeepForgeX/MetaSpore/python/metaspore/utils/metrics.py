#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metaspore.utils.metrics - 评估指标模块

包含 AUC、PCOC、LogLoss 等评估指标的计算函数。
"""

import math
import numpy as np
from typing import List, Tuple
from sklearn.metrics import roc_auc_score


def compute_auc_and_pcoc_regression(label_pred_list: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    计算回归任务的 AUC 和 PCOC
    
    Args:
        label_pred_list: [(label, prediction), ...]
    
    Returns:
        (auc, pcoc)
    """
    if not label_pred_list:
        return float('nan'), float('nan')
    
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
    auc = _score_regression_auc_optimized(labels, preds)
    return auc, pcoc


def _score_regression_auc_optimized(label: np.ndarray, score: np.ndarray) -> float:
    """优化版回归 AUC 计算"""
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
    i = 0
    while i < n:
        j = i
        while j < n and sorted_labels[j] == sorted_labels[i]:
            j += 1
        current_label = sorted_labels[i]
        group_size = j - i

        # Count how many previous labels are > current_label
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
        return 0.0
    
    total_loss = 0.0
    n_samples = len(label_prediction_list)
    
    for label, pred_prob in label_prediction_list:
        # 防止 log(0)
        eps = 1e-15
        pred_prob = max(min(pred_prob, 1.0 - eps), eps)
        
        # 计算单个样本的 logloss
        if label == 1.0:
            sample_loss = -math.log(pred_prob)
        else:
            sample_loss = -math.log(1.0 - pred_prob)
        
        total_loss += sample_loss
    
    return total_loss / n_samples


def compute_auc_pcoc(label_prediction_list: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    根据 label 和 raw prediction 列表计算 AUC 和 PCOC

    Args:
        label_prediction_list: List of (label, prediction_score)

    Returns:
        (auc, pcoc)
    """
    auc_dict = {}
    for label, score in label_prediction_list:
        score_str = str(score)
        auc_dict.setdefault(score_str, [0, 0])  # [总数, 正样本数]
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

    Args:
        num_clicks: 每组的点击数
        num_impressions: 每组的曝光数
        predicted_ctr: 每组的预测 CTR

    Returns:
        pcoc 值
    """
    sum_pctr = 0.0
    for i in range(len(num_clicks)):
        sum_pctr += predicted_ctr[i] * num_impressions[i]
    sum_clicks = sum(num_clicks)
    return round(sum_pctr / sum_clicks, 4) if sum_clicks != 0 else 0.0


def scoreClickAUC(
    num_clicks: List[int], 
    num_impressions: List[int], 
    predicted_ctr: List[float]
) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    计算 click-based AUC

    Args:
        num_clicks: 每个预测分数 bucket 中的点击数
        num_impressions: 每个 bucket 中的曝光数
        predicted_ctr: 每个 bucket 的预测CTR

    Returns:
        (auc, bucket_info)
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
