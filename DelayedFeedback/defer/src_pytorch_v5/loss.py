"""
Defer 损失函数 - PyTorch v4 版本

v4 严格对齐 TF 原版 delay_win_time_loss + delay_win_select_loss:
1. 多窗口设计: 24h/48h/72h 三个窗口独立建模
2. 每个窗口有独立的 loss_win_now + loss_win_spm
3. 全局 loss_cv_spm 建模最终转化
4. TF 原版的 stop_gradient 策略

损失函数类型:
- cross_entropy_loss: 标准二分类交叉熵 (Vanilla, Oracle)
- fnw_loss: 假负加权损失 (FNW)
- fnc_loss: 假负复制损失 (FNC)
- dfm_loss: 指数延迟损失 (DFM)
- esdfm_loss: ES-DFM 损失
- winadapt_loss_v4: 多窗口自适应损失 v4 - 严格对齐 TF 原版
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 时间窗口 (小时)
TIME_WINDOWS = [24, 48, 72, 168]


def stable_log1pex(x):
    """稳定计算 log(1 + exp(x)) = softplus"""
    return F.softplus(x)


def stable_sigmoid(x):
    """数值稳定的 sigmoid，对齐 TF 原版 stable_sigmoid_pos"""
    return torch.sigmoid(x)


# ============================================================================
# 基础损失函数
# ============================================================================

def cross_entropy_loss(logits, labels, reduction='mean'):
    """
    标准二分类交叉熵损失
    用于: Vanilla, Oracle
    """
    return F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)


# ============================================================================
# FNW / FNC 损失函数
# ============================================================================

def fnw_loss(logits, labels, elapsed_time, C=0.25):
    """
    假负加权损失 (Fake Negative Weighted)
    """
    z = labels.float()
    e = elapsed_time.float()
    
    w = e / (e + C + 1e-8)
    
    pos_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='none')
    neg_loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none')
    
    loss = z * pos_loss + (1 - z) * w * neg_loss
    
    return loss.mean()


def fnc_loss(logits, labels, elapsed_time, C=0.25):
    """假负复制损失 (Fake Negative Calibration)"""
    return fnw_loss(logits, labels, elapsed_time, C)


# ============================================================================
# DFM 损失函数
# ============================================================================

def dfm_loss(cv_logits, log_lamb, labels, delay_time):
    """指数延迟损失 (DFM - Delayed Feedback Model)"""
    z = labels.float()
    d = delay_time.float()
    
    lamb = F.softplus(log_lamb)
    log_lamb_stable = torch.log(lamb + 1e-8)
    
    p = torch.sigmoid(cv_logits)
    
    pos_loss = stable_log1pex(-cv_logits) - log_lamb_stable + lamb * d
    
    survival = torch.exp(-lamb * d)
    neg_loss = -torch.log(1 - p + p * survival + 1e-8)
    
    loss = z * pos_loss + (1 - z) * neg_loss
    
    return loss.mean()


# ============================================================================
# ES-DFM 损失函数
# ============================================================================

def esdfm_pretrain_loss(tn_logits, dp_logits, tn_labels, dp_labels, pos_labels):
    """ES-DFM 预训练损失函数"""
    tn_mask = (1 - pos_labels) + dp_labels
    
    tn_loss_raw = F.binary_cross_entropy_with_logits(tn_logits, tn_labels.float(), reduction='none')
    tn_loss = (tn_loss_raw * tn_mask).sum() / (tn_mask.sum() + 1e-8)
    
    dp_loss = F.binary_cross_entropy_with_logits(dp_logits, dp_labels.float(), reduction='mean')
    
    return tn_loss + dp_loss


def esdfm_importance_weight_loss(cv_logits, tn_prob, dp_prob, labels, pos_labels):
    """ES-DFM 重要性加权损失"""
    z = labels.float()
    
    w_neg = tn_prob / (tn_prob + dp_prob + 1e-8)
    w = z + (1 - z) * w_neg
    
    loss_raw = F.binary_cross_entropy_with_logits(cv_logits, z, reduction='none')
    loss = (loss_raw * w).mean()
    
    return loss


# ============================================================================
# WinAdapt v4 损失函数 - 严格对齐 TF 原版 delay_win_time_loss + delay_win_select_loss
# ============================================================================

def winadapt_loss_v4(outputs, labels_dict, time_windows=TIME_WINDOWS, loss_scale=3.0):
    """
    WinAdapt v4 多窗口自适应损失 - 严格对齐 TF 原版
    
    核心设计 (来自 TF 原版 delay_win_time_loss + delay_win_select_loss):
    
    1. 每个窗口独立建模:
       - P(窗口w内转化) = P(cv) * P(t<=w | cv)
       - loss_win_now: 窗口内观察 (label_01 vs label_00)，使用 stop_gradient
       - loss_win_spm: 窗口后建模 (label_01 vs label_10 vs label_11)
    
    2. 全局 CVR 建模:
       - loss_cv_spm: 最终转化 (label_01 + label_11) vs 真负 (label_10)
    
    标签定义 (TF 原版):
    - label_01: 窗口内已转化 (window positive)
    - label_00: 窗口内未转化但窗口未结束 (unobservable)
    - label_11: 窗口后转化 (delayed positive / SPM)
    - label_10: 真正未转化 (true negative)
    
    TF 原版权重 (delay_win_select_loss subloss=1):
    - loss_win_now: 0.1 * (24h) + 0.05 * (48h) + 0.05 * (72h) = 0.2
    - loss_win_spm: 0.1 * (24h) + 0.05 * (48h) + 0.05 * (72h) = 0.2
    - loss_cv_spm: 0.6
    
    Args:
        outputs: dict with
            - 'cv_logits': [batch_size] 最终转化 logits
            - 'time_24h_logits': [batch_size] 24h 条件概率 logits
            - 'time_48h_logits': [batch_size] 48h 条件概率 logits
            - 'time_72h_logits': [batch_size] 72h 条件概率 logits
        labels_dict: dict with
            - 'label_{w}h': 窗口 w 内是否转化 (0/1)
            - 'observable_{w}h': 窗口 w 是否可观察 (0/1)
            - 'label_168h' or 'label_oracle': 最终转化标签
        time_windows: 时间窗口列表 (小时)
        loss_scale: loss 缩放因子 (TF 原版用 3)
    """
    cv_logits = outputs['cv_logits']
    
    # 时间窗口条件概率 logits
    time_logits_dict = {
        24: outputs['time_24h_logits'],
        48: outputs['time_48h_logits'],
        72: outputs['time_72h_logits'],
    }
    
    # 稳定 sigmoid
    cv_prob = stable_sigmoid(cv_logits)
    time_probs = {w: stable_sigmoid(time_logits_dict[w]) for w in [24, 48, 72]}
    
    # stop_gradient 版本
    cv_prob_stopped = cv_prob.detach()
    time_probs_stopped = {w: tp.detach() for w, tp in time_probs.items()}
    
    # 获取最终转化标签
    label_oracle = labels_dict.get('label_168h', labels_dict.get('label_oracle'))
    
    eps = 1e-7
    
    # ========== 多窗口 Loss ==========
    # TF 原版 delay_win_select_loss subloss=1 的权重:
    # 24h: 0.1 (now) + 0.1 (spm) = 0.2
    # 48h: 0.05 (now) + 0.05 (spm) = 0.1
    # 72h: 0.05 (now) + 0.05 (spm) = 0.1
    window_weights = {24: 0.1, 48: 0.05, 72: 0.05}
    
    total_loss_win_now = 0.0
    total_loss_win_spm = 0.0
    
    # 用于 loss_cv_spm 的累积标签
    # 注意: 使用 168h (最终窗口) 的标签来计算 label_01, label_11, label_10
    # 但每个窗口的 loss_win 使用各自窗口的标签
    
    for w in [24, 48, 72]:
        weight = window_weights[w]
        
        # 当前窗口标签
        label_w = labels_dict[f'label_{w}h']  # 窗口 w 内是否转化
        observable_w = labels_dict[f'observable_{w}h']  # 窗口 w 是否可观察
        
        # 构建 4 类标签 (TF 原版定义)
        # label_01: 窗口内已转化 = label_w & observable_w
        # label_00: 窗口内未转化且窗口未结束 = ~label_w & ~observable_w
        # label_11: 窗口后转化 (延迟正) = ~label_w & label_oracle & observable_w
        # label_10: 真负 = ~label_oracle & observable_w
        
        label_01 = label_w * observable_w
        label_00 = (1 - label_w) * (1 - observable_w)
        label_11 = (1 - label_w) * label_oracle * observable_w
        label_10 = (1 - label_oracle) * observable_w
        
        # 联合概率: P(窗口w内转化) = P(cv) * P(t <= w | cv)
        win_prob_1 = cv_prob * time_probs[w]
        win_prob_0 = 1 - win_prob_1
        
        # stop_gradient 版本 (用于 loss_win_now)
        # TF 原版: stable_win_prop_1_stop = stable_cv_prop * stable_time_prop_stop
        win_prob_1_stopped = cv_prob * time_probs_stopped[w]
        win_prob_0_stopped = 1 - win_prob_1_stopped
        
        # 数值稳定
        win_prob_1 = torch.clamp(win_prob_1, eps, 1 - eps)
        win_prob_0 = torch.clamp(win_prob_0, eps, 1 - eps)
        win_prob_1_stopped = torch.clamp(win_prob_1_stopped, eps, 1 - eps)
        win_prob_0_stopped = torch.clamp(win_prob_0_stopped, eps, 1 - eps)
        
        # ===== Loss: loss_win_now (窗口内观察) =====
        # TF 原版: loss_win = -mean(log(win_1_stop) * label_01 + log(win_0_stop) * (1-label_01))
        # 但 TF 原版用的是 label_01_sum (累积标签)，这里我们用当前窗口标签
        loss_win_now = -(
            torch.log(win_prob_1_stopped) * label_01 + 
            torch.log(win_prob_0_stopped) * (1 - label_01)
        ).mean()
        
        # ===== Loss: loss_win_spm (窗口后建模) =====
        # TF 原版: loss_win_spm = -mean(log(win_1) * label_01 + log(win_0) * label_15_0)
        # 其中 label_15_0 = min(label_11_15 + label_11_30 + label_11_60 + label10, 1)
        # 这里简化为: label_01 正样本, (label_10 + label_11) 负样本
        label_neg = label_10 + label_11  # 窗口内未转化的样本
        loss_win_spm = -(
            torch.log(win_prob_1) * label_01 + 
            torch.log(win_prob_0) * label_neg
        ).mean()
        
        total_loss_win_now += weight * loss_win_now
        total_loss_win_spm += weight * loss_win_spm
    
    # ========== Loss: loss_cv_spm (最终 CVR) ==========
    # TF 原版: loss_cv_spm = -mean(log(cv) * label01 + log(cv) * label11 + log(1-cv) * label10)
    # 使用 168h (最终) 的标签
    label_168h = labels_dict.get('label_168h', label_oracle)
    observable_168h = labels_dict.get('observable_168h', torch.ones_like(label_oracle))
    
    # 对于 168h 窗口，所有样本应该都是可观察的
    # label_01_final: 最终转化
    # label_10_final: 最终未转化 (真负)
    label_01_final = label_168h * observable_168h
    label_10_final = (1 - label_168h) * observable_168h
    
    cv_prob_clamped = torch.clamp(cv_prob, eps, 1 - eps)
    
    loss_cv_spm = -(
        torch.log(cv_prob_clamped) * label_01_final + 
        torch.log(1 - cv_prob_clamped) * label_10_final
    ).mean()
    
    # ========== 组合 Loss ==========
    # TF 原版 delay_win_select_loss subloss=1:
    # loss = 0.1*loss_win_15 + 0.05*loss_win_30 + 0.05*loss_win_60 
    #      + 0.1*loss_win_15_spm + 0.05*loss_win_30_spm + 0.05*loss_win_60_spm 
    #      + 0.6*loss_cv_spm
    total_loss = total_loss_win_now + total_loss_win_spm + 0.6 * loss_cv_spm
    
    # 缩放 (TF 原版用 *3 或 *1)
    total_loss = total_loss * loss_scale
    
    return total_loss


def winadapt_loss(outputs, labels_dict, time_windows=TIME_WINDOWS):
    """
    WinAdapt 损失函数 - 默认使用 v4 实现
    """
    return winadapt_loss_v4(outputs, labels_dict, time_windows)


# ============================================================================
# 损失函数工厂
# ============================================================================

LOSS_REGISTRY = {
    'vanilla': cross_entropy_loss,
    'oracle': cross_entropy_loss,
    'fnw': fnw_loss,
    'fnc': fnc_loss,
    'dfm': dfm_loss,
    'esdfm': esdfm_importance_weight_loss,
    'winadapt': winadapt_loss,
    'winadapt_v4': winadapt_loss_v4,
}


def get_loss_fn(method):
    """获取损失函数"""
    method = method.lower()
    if method not in LOSS_REGISTRY:
        raise ValueError(f"Unknown method: {method}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[method]
