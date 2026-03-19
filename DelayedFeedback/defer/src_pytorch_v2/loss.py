"""
Defer 损失函数 - PyTorch v2 版本

损失函数类型:
- cross_entropy_loss: 标准二分类交叉熵 (Vanilla, Oracle)
- fnw_loss: 假负加权损失 (FNW)
- fnc_loss: 假负复制损失 (FNC)
- dfm_loss: 指数延迟损失 (DFM)
- esdfm_loss: ES-DFM 损失
- winadapt_loss: 多窗口自适应损失 (WinAdapt)
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
    
    对假负样本（短窗口内未转化但可能后续转化）进行加权
    
    Args:
        logits: [batch_size] CVR logits
        labels: [batch_size] 是否转化 (0/1)
        elapsed_time: [batch_size] 已过时间 (小时)
        C: 窗口参数 (小时)
    
    假负权重: w = e / (e + C)
    时间越短，权重越小（因为更可能是假负）
    """
    z = labels.float()
    e = elapsed_time.float()
    
    # 假负权重
    w = e / (e + C + 1e-8)
    
    pos_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='none')
    neg_loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none')
    
    # 正样本用标准损失，负样本用加权损失
    loss = z * pos_loss + (1 - z) * w * neg_loss
    
    return loss.mean()


def fnc_loss(logits, labels, elapsed_time, C=0.25):
    """
    假负复制损失 (Fake Negative Calibration)
    类似 FNW，使用相同的加权策略
    """
    return fnw_loss(logits, labels, elapsed_time, C)


# ============================================================================
# DFM 损失函数
# ============================================================================

def dfm_loss(cv_logits, log_lamb, labels, delay_time):
    """
    指数延迟损失 (DFM - Delayed Feedback Model)
    
    假设转化延迟服从指数分布 Exp(λ)
    
    正样本损失: -log P(转化) - log λ + λ * delay
    负样本损失: -log(1 - P(转化) + P(转化) * exp(-λ * elapsed))
    
    Args:
        cv_logits: [batch_size] CVR logits
        log_lamb: [batch_size] log(λ) 延迟分布参数
        labels: [batch_size] 是否转化 (0/1)
        delay_time: [batch_size] 延迟/已过时间 (小时)
    """
    z = labels.float()
    d = delay_time.float()
    
    # λ = softplus(log_lamb) 确保 λ > 0
    lamb = F.softplus(log_lamb)
    log_lamb_stable = torch.log(lamb + 1e-8)
    
    # 转化概率
    p = torch.sigmoid(cv_logits)
    
    # 正样本损失: -log P(转化) - log λ + λ * delay
    pos_loss = stable_log1pex(-cv_logits) - log_lamb_stable + lamb * d
    
    # 负样本损失: -log(1 - P + P * exp(-λ * e))
    survival = torch.exp(-lamb * d)
    neg_loss = -torch.log(1 - p + p * survival + 1e-8)
    
    loss = z * pos_loss + (1 - z) * neg_loss
    
    return loss.mean()


# ============================================================================
# ES-DFM 损失函数
# ============================================================================

def esdfm_pretrain_loss(tn_logits, dp_logits, tn_labels, dp_labels, pos_labels):
    """
    ES-DFM 预训练损失函数
    
    学习两个辅助任务:
    - tn_logits: 预测是否为真负样本
    - dp_logits: 预测是否为延迟正样本
    
    Args:
        tn_logits: [batch_size] 真负预测 logits
        dp_logits: [batch_size] 延迟正预测 logits
        tn_labels: [batch_size] 真负标签
        dp_labels: [batch_size] 延迟正标签
        pos_labels: [batch_size] 窗口内正样本标签
    """
    # tn_mask: 只在非正样本或延迟正样本上计算 tn 损失
    tn_mask = (1 - pos_labels) + dp_labels
    
    # tn 损失
    tn_loss_raw = F.binary_cross_entropy_with_logits(tn_logits, tn_labels.float(), reduction='none')
    tn_loss = (tn_loss_raw * tn_mask).sum() / (tn_mask.sum() + 1e-8)
    
    # dp 损失
    dp_loss = F.binary_cross_entropy_with_logits(dp_logits, dp_labels.float(), reduction='mean')
    
    return tn_loss + dp_loss


def esdfm_importance_weight_loss(cv_logits, tn_prob, dp_prob, labels, pos_labels):
    """
    ES-DFM 重要性加权损失
    
    核心思想: 用 tn_prob 和 dp_prob 计算重要性权重来纠正假负偏差
    
    Args:
        cv_logits: [batch_size] CVR logits
        tn_prob: [batch_size] 真负概率
        dp_prob: [batch_size] 延迟正概率
        labels: [batch_size] 观察标签 (窗口内是否转化)
        pos_labels: [batch_size] 窗口内正样本标签
    """
    z = labels.float()
    
    # 重要性权重
    # 正样本: w = 1
    # 负样本: w = tn_prob / (tn_prob + dp_prob)
    w_neg = tn_prob / (tn_prob + dp_prob + 1e-8)
    w = z + (1 - z) * w_neg
    
    # 加权交叉熵
    loss_raw = F.binary_cross_entropy_with_logits(cv_logits, z, reduction='none')
    loss = (loss_raw * w).mean()
    
    return loss


# ============================================================================
# WinAdapt 损失函数
# ============================================================================

def stable_sigmoid(x):
    """数值稳定的 sigmoid"""
    return torch.sigmoid(x)


def winadapt_loss(outputs, labels_dict, time_windows=TIME_WINDOWS):
    """
    WinAdapt 多窗口自适应损失 - 与原始 TF 版本对齐
    
    核心思想:
    - 模型输出 4 个 logits: cv_logit + 3 个时间窗口条件概率 logit
    - P(窗口内转化) = P(cv) * P(t<=window | cv)
    - 对每个窗口的联合概率计算 BCE loss
    - 同时对最终转化标签 (oracle) 计算 CVR loss
    
    Args:
        outputs: dict with
            - 'cv_logits': [batch_size] 最终转化 logits
            - 'time_24h_logits': [batch_size] 24h 条件概率 logits
            - 'time_48h_logits': [batch_size] 48h 条件概率 logits
            - 'time_72h_logits': [batch_size] 72h 条件概率 logits
        labels_dict: dict with
            - 'label_{w}h': 窗口 w 的标签 (0/1)
            - 'observable_{w}h': 窗口 w 的可观察标记 (0/1)
            - 'label_oracle': 最终转化标签 (可选)
        time_windows: 时间窗口列表 (小时), 前 3 个用于时间条件概率
    """
    cv_logits = outputs['cv_logits']
    cv_prob = stable_sigmoid(cv_logits)
    
    # 时间窗口条件概率
    time_logits_list = [
        outputs['time_24h_logits'],
        outputs['time_48h_logits'],
        outputs['time_72h_logits'],
    ]
    time_probs = [stable_sigmoid(tl) for tl in time_logits_list]
    
    # 用于 stop_gradient 的 cv_prob (防止时间窗口 loss 影响 cv_prob 学习)
    cv_prob_stopped = cv_prob.detach()
    
    total_loss = 0.0
    
    # 窗口 loss: 对前 3 个窗口 (24h, 48h, 72h) 使用联合概率建模
    for i, w in enumerate(time_windows[:3]):
        win_label = labels_dict[f'label_{w}h']
        observable = labels_dict[f'observable_{w}h']
        
        # 联合概率: P(win) = P(cv) * P(t<=w | cv)
        # 使用 stop_gradient 的 cv_prob，让梯度主要流向 time_prob
        win_prob = cv_prob_stopped * time_probs[i]
        
        # 数值稳定的 BCE loss
        # loss = -[y * log(p) + (1-y) * log(1-p)]
        win_prob_clamp = torch.clamp(win_prob, 1e-7, 1 - 1e-7)
        loss_per_sample = -(win_label * torch.log(win_prob_clamp) + 
                           (1 - win_label) * torch.log(1 - win_prob_clamp))
        
        # 只在可观察样本上计算
        weighted_loss = (loss_per_sample * observable).sum() / (observable.sum() + 1e-8)
        total_loss += weighted_loss
    
    # CVR loss: 对最终转化标签 (168h / oracle) 计算
    # 这里用 168h 标签作为 oracle (因为数据集中 observable_168h = 100%)
    cv_label = labels_dict.get('label_168h', labels_dict.get('label_oracle'))
    if cv_label is not None:
        cv_loss = F.binary_cross_entropy_with_logits(cv_logits, cv_label.float())
        # 权重: 窗口 loss 和 CVR loss 的平衡
        # 原始 TF 版本用 0.2 * win_loss + 0.8 * cv_loss
        total_loss = 0.2 * total_loss / 3 + 0.8 * cv_loss
    else:
        total_loss = total_loss / 3
    
    return total_loss


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
}


def get_loss_fn(method):
    """获取损失函数"""
    method = method.lower()
    if method not in LOSS_REGISTRY:
        raise ValueError(f"Unknown method: {method}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[method]
