"""
Defer 损失函数 - PyTorch 版本

损失函数类型:
- cross_entropy_loss: 标准二分类交叉熵 (Baseline, Vanilla, Oracle)
- fnw_loss: 假负加权损失 (FNW)
- fnc_loss: 假负复制损失 (FNC)  
- exp_delay_loss: 指数延迟损失 (DFM)
- delay_tn_dp_loss: 真负/延迟正损失 (ES-DFM)
- fsiw_loss: 重要性加权损失 (FSIW)
- delay_3class_loss: 三分类损失
- delay_win_time_loss: 窗口时间损失
- delay_win_select_loss: 自适应窗口损失 (Defer 核心)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def stable_log1pex(x):
    """
    稳定计算 log(1 + exp(x))
    等价于 softplus，但数值更稳定
    """
    return F.softplus(x)


def stable_sigmoid(x):
    """稳定的 sigmoid"""
    return torch.sigmoid(x)


# ============================================================================
# 基础损失函数
# ============================================================================

def cross_entropy_loss(targets, outputs, params=None):
    """
    标准二分类交叉熵损失
    
    用于: Baseline, Vanilla, Oracle
    
    Args:
        targets: dict with "label" [batch_size, 1]
        outputs: dict with "logits" [batch_size, 1]
    """
    logits = outputs["logits"]
    labels = targets["label"].float()
    
    if labels.dim() > 1:
        labels = labels[:, 0:1]
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)
    
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return {"loss": loss}


# ============================================================================
# FNW / FNC 损失函数
# ============================================================================

def fnw_loss(targets, outputs, params=None):
    """
    假负加权损失 (Fake Negative Weighted)
    
    对假负样本（短窗口内未转化但可能后续转化）进行加权
    
    Args:
        targets: dict with 
            - "label": [batch_size, 2] -> [label, elapsed_time]
        outputs: dict with "logits"
    """
    logits = outputs["logits"]
    labels = targets["label"]
    
    z = labels[:, 0:1].float()  # 是否转化
    e = labels[:, 1:2].float()  # 已过时间
    
    # 假负权重: 时间越短，权重越小（因为更可能是假负）
    # w = e / (e + C), C 是窗口大小
    C = params.get("C", 0.25) * 3600 if params else 900  # 默认 15 分钟
    w = e / (e + C)
    
    pos_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='none')
    neg_loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none')
    
    # 正样本用标准损失，负样本用加权损失
    loss = z * pos_loss + (1 - z) * w * neg_loss
    
    return {"loss": loss.mean()}


def fnc_loss(targets, outputs, params=None):
    """
    假负复制损失 (Fake Negative Calibration)
    
    类似 FNW，但使用不同的加权策略
    """
    return fnw_loss(targets, outputs, params)


# ============================================================================
# DFM 损失函数
# ============================================================================

def exp_delay_loss(targets, outputs, params=None):
    """
    指数延迟损失 (DFM - Delayed Feedback Model)
    
    假设转化延迟服从指数分布 Exp(λ)
    
    正样本损失: -log P(转化) - log λ + λ * delay
    负样本损失: -log(1 - P(转化) + P(转化) * exp(-λ * elapsed))
    
    Args:
        targets: dict with
            - "label": [batch_size, 2] -> [is_converted, delay_time]
        outputs: dict with
            - "logits": CVR logits
            - "log_lamb": log(λ) 延迟分布参数
    """
    x = outputs["logits"]  # CVR logits
    log_lamb = outputs["log_lamb"]
    
    labels = targets["label"]
    z = labels[:, 0:1].float()  # 是否转化
    d = labels[:, 1:2].float()  # 延迟时间 (秒)
    e = d  # elapsed time = delay for converted samples
    
    # λ = softplus(log_lamb) 确保 λ > 0
    lamb = F.softplus(log_lamb)
    log_lamb_stable = torch.log(lamb + 1e-8)
    
    # 转化概率
    p = torch.sigmoid(x)
    
    # 正样本损失: -log P(转化) - log λ + λ * delay
    pos_loss = stable_log1pex(-x) - log_lamb_stable + lamb * d
    
    # 负样本损失: -log(1 - P + P * exp(-λ * e))
    # = -log(1 - P * (1 - exp(-λ * e)))
    survival = torch.exp(-lamb * e)
    neg_loss = -torch.log(1 - p + p * survival + 1e-8)
    
    loss = z * pos_loss + (1 - z) * neg_loss
    
    return {"loss": loss.mean()}


# ============================================================================
# ES-DFM 损失函数
# ============================================================================

def delay_tn_dp_loss(targets, outputs, params=None):
    """
    ES-DFM 预训练损失函数
    
    学习两个辅助任务:
    - tn_logits: 预测是否为真负样本
    - dp_logits: 预测是否为延迟正样本
    
    Args:
        targets: dict with "label" [batch_size, 3] -> [tn_label, dp_label, pos_label]
        outputs: dict with "tn_logits", "dp_logits"
    """
    tn_logits = outputs["tn_logits"]
    dp_logits = outputs["dp_logits"]
    
    labels = targets["label"]
    tn_label = labels[:, 0:1].float()   # 真负样本
    dp_label = labels[:, 1:2].float()   # 延迟正样本
    pos_label = labels[:, 2:3].float()  # 窗口内正样本
    
    # tn_mask: 只在非正样本或延迟正样本上计算 tn 损失
    tn_mask = (1 - pos_label) + dp_label
    
    # tn 损失
    tn_loss_raw = F.binary_cross_entropy_with_logits(tn_logits, tn_label, reduction='none')
    tn_loss = (tn_loss_raw * tn_mask).sum() / (tn_mask.sum() + 1e-8)
    
    # dp 损失
    dp_loss = F.binary_cross_entropy_with_logits(dp_logits, dp_label, reduction='mean')
    
    loss = tn_loss + dp_loss
    
    return {
        "loss": loss,
        "tn_loss": tn_loss,
        "dp_loss": dp_loss
    }


def esdfm_importance_weight_loss(targets, outputs, params=None):
    """
    ES-DFM 重要性加权损失 (原版核心)
    
    核心思想: 用 tn_prob 和 dp_prob 计算重要性权重来纠正假负偏差
    
    - 正样本权重: 1 + dp_prob (补偿延迟正样本)
    - 负样本权重: (1 + dp_prob) * tn_prob (只信任真负样本)
    
    Args:
        targets: dict with "label" [batch_size, 3] -> [tn_label, dp_label, pos_label]
                 或 [batch_size, 1] 简化标签 (窗口内是否转化)
        outputs: dict with "cv_logits", "tn_logits", "dp_logits"
    """
    cv_logits = outputs["cv_logits"]
    tn_logits = outputs["tn_logits"]
    dp_logits = outputs["dp_logits"]
    
    labels = targets["label"]
    # 标签: 窗口内是否转化 (pos_label)
    if labels.shape[1] >= 3:
        z = labels[:, 2:3].float()  # pos_label
    else:
        z = labels[:, 0:1].float()
    
    # 计算辅助概率 (stop gradient)
    tn_prob = torch.sigmoid(tn_logits).detach()
    dp_prob = torch.sigmoid(dp_logits).detach()
    
    # 重要性权重
    pos_weight = 1 + dp_prob  # 正样本权重: 补偿延迟正
    neg_weight = (1 + dp_prob) * tn_prob  # 负样本权重: 只信任真负
    
    # 加权交叉熵
    cv_prob = torch.sigmoid(cv_logits)
    pos_loss = -torch.log(cv_prob + 1e-8)
    neg_loss = -torch.log(1 - cv_prob + 1e-8)
    
    loss = (pos_loss * pos_weight * z + neg_loss * neg_weight * (1 - z)).mean()
    
    return {
        "loss": loss,
    }


def esdfm_loss(targets, outputs, params=None):
    """
    ES-DFM 重要性加权损失
    
    使用预训练的 tn/dp 模型计算重要性权重，纠正假负样本的分布偏移
    """
    logits = outputs["logits"]
    labels = targets["label"]
    
    z = labels[:, 0:1].float()  # 是否转化
    
    # 如果有重要性权重
    if "weight" in targets:
        weight = targets["weight"]
    else:
        weight = torch.ones_like(z)
    
    loss = F.binary_cross_entropy_with_logits(logits, z, reduction='none')
    loss = (loss * weight).mean()
    
    return {"loss": loss}


# ============================================================================
# FSIW 损失函数
# ============================================================================

def fsiw_loss(targets, outputs, params=None):
    """
    FSIW (Fake negative Sample Importance Weighting) 损失
    """
    logits = outputs["logits"]
    labels = targets["label"]
    
    z = labels[:, 0:1].float()
    
    # FSIW 权重
    if "fsiw_weight" in targets:
        weight = targets["fsiw_weight"]
    else:
        weight = torch.ones_like(z)
    
    loss = F.binary_cross_entropy_with_logits(logits, z, reduction='none')
    loss = (loss * weight).mean()
    
    return {"loss": loss}


# ============================================================================
# 三分类损失
# ============================================================================

def delay_3class_loss(targets, outputs, params=None):
    """
    三分类损失
    
    类别:
    - 0: 真负样本 (确认未转化)
    - 1: 假负样本 (可能后续转化)
    - 2: 正样本 (已转化)
    """
    logits = outputs["logits"]  # [batch_size, 3]
    labels = targets["label"]  # [batch_size, 3] one-hot
    
    loss = F.cross_entropy(logits, labels.argmax(dim=1))
    
    return {"loss": loss}


# ============================================================================
# 窗口时间损失
# ============================================================================

def delay_win_time_loss(targets, outputs, params=None):
    """
    窗口时间损失
    
    同时学习:
    - 转化概率
    - 窗口内转化概率
    """
    cv_logits = outputs["cv_logits"]
    time_logits = outputs["time_logits"]
    
    labels = targets["label"]
    cv_label = labels[:, 0:1].float()
    win_label = labels[:, 1:2].float()
    
    cv_loss = F.binary_cross_entropy_with_logits(cv_logits, cv_label)
    time_loss = F.binary_cross_entropy_with_logits(time_logits, win_label)
    
    loss = cv_loss + time_loss
    
    return {
        "loss": loss,
        "cv_loss": cv_loss,
        "time_loss": time_loss
    }


# ============================================================================
# Defer 核心: 自适应窗口损失
# ============================================================================

def delay_win_select_loss(targets, outputs, params=None):
    """
    自适应窗口选择损失 (Defer 核心方法) - 对齐 TF 原版 delay_win_select_loss

    TF 原版三套 loss 并行:
      1. stop_grad 版 (loss_win_15/30/60): time_logits 梯度截断，只更新 cv_logits
      2. spm 版 (loss_win_15/30/60_spm): 使用精确负样本 (label_11_xx + label10)
      3. cv_spm: 延迟正样本 (label_11) 也监督 cv_logits

    最终 loss (subloss=1, 原版最优):
      0.1*loss_win_15 + 0.05*loss_win_30 + 0.05*loss_win_60
    + 0.1*loss_win_15_spm + 0.05*loss_win_30_spm + 0.05*loss_win_60_spm
    + 0.6*loss_cv_spm

    标签格式 (14 维):
        0:  label_11      - 延迟正样本 (窗口外转化)
        1:  label_10      - 真负样本
        2:  label_01_15   - win1 内转化
        3:  label_01_30   - win1~win2 内转化 (增量)
        4:  label_01_60   - win2~win3 内转化 (增量)
        5:  label_01_30_sum - win2 内累计转化
        6:  label_01_60_sum - win3 内累计转化
        7:  label_01_30_mask - win2 是否可观察
        8:  label_01_60_mask - win3 是否可观察
        9:  label_00      - 窗口内负样本
        10: label_01      - 窗口内正样本 (win3 内所有转化)
        11: label_11_15   - 延迟正且 delay <= win1+24h
        12: label_11_30   - 延迟正且 delay <= win2+24h
        13: label_11_60   - 延迟正且 delay <= win3+24h
    """
    cv_logits = outputs["cv_logits"]
    time15_logits = outputs["time15_logits"]
    time30_logits = outputs["time30_logits"]
    time60_logits = outputs["time60_logits"]

    labels = targets["label"]
    eps = 1e-7

    # ==================== 解析 14 维标签 ====================
    label_11        = labels[:, 0:1].float()   # 延迟正样本 (窗口外)
    label_10        = labels[:, 1:2].float()   # 真负样本
    label_01_15     = labels[:, 2:3].float()   # win1 内转化
    # label_01_30   = labels[:, 3:4]           # win1~win2 增量 (不直接用)
    # label_01_60   = labels[:, 4:5]           # win2~win3 增量 (不直接用)
    label_01_30_sum = labels[:, 5:6].float()   # win2 内累计
    label_01_60_sum = labels[:, 6:7].float()   # win3 内累计
    label_01_30_mask= labels[:, 7:8].float()   # win2 可观察 mask
    label_01_60_mask= labels[:, 8:9].float()   # win3 可观察 mask
    label_00        = labels[:, 9:10].float()  # 窗口内负样本
    label_01        = labels[:, 10:11].float() # 窗口内正样本
    label_11_15     = labels[:, 11:12].float() # 延迟正细分 win1
    label_11_30     = labels[:, 12:13].float() # 延迟正细分 win2
    label_11_60     = labels[:, 13:14].float() # 延迟正细分 win3

    # ==================== 计算概率 ====================
    cv_prob     = torch.sigmoid(cv_logits)
    time15_prob = torch.sigmoid(time15_logits)
    time30_prob = torch.sigmoid(time30_logits)
    time60_prob = torch.sigmoid(time60_logits)

    # stop_gradient 版 (只更新 cv_logits，time_logits 梯度截断)
    time15_prob_stop = time15_prob.detach()
    time30_prob_stop = time30_prob.detach()
    time60_prob_stop = time60_prob.detach()

    # 联合概率: P(win 内转化) = P(转化) * P(win 内 | 转化)
    win15_prob      = cv_prob * time15_prob
    win15_prob_stop = cv_prob * time15_prob_stop
    win30_prob      = cv_prob * time30_prob
    win30_prob_stop = cv_prob * time30_prob_stop
    win60_prob      = cv_prob * time60_prob
    win60_prob_stop = cv_prob * time60_prob_stop

    # ==================== 第一套: stop_grad 版 ====================
    # 对应 TF: loss_win_15/30/60 (time_logits stop_gradient)
    # 负样本 = 1 - label (全量负样本，包含假负)
    loss_win_15 = -(
        torch.log(win15_prob_stop + eps) * label_01_15
        + torch.log(1 - win15_prob_stop + eps) * (1 - label_01_15)
    ).mean()

    loss_win_30 = -(
        torch.log(win30_prob_stop + eps) * label_01_30_sum
        + torch.log(1 - win30_prob_stop + eps) * (1 - label_01_30_sum)
    ).mean()

    loss_win_60 = -(
        torch.log(win60_prob_stop + eps) * label_01_60_sum
        + torch.log(1 - win60_prob_stop + eps) * (1 - label_01_60_sum)
    ).mean()

    # ==================== 第二套: spm 版 (精确负样本) ====================
    # 对应 TF: loss_win_15/30/60_spm
    # 精确负样本: 确定不在该窗口内转化的样本
    #   label_15_0 = label_11_15 + label_11_30 + label_11_60 + label_10
    #   (延迟正样本中 delay > win1 的，以及真负样本)
    label_15_0 = torch.clamp(label_11_15 + label_11_30 + label_11_60 + label_10, 0, 1)
    label_30_0 = torch.clamp(label_11_30 + label_11_60 + label_10, 0, 1)
    label_60_0 = torch.clamp(label_11_60 + label_10, 0, 1)

    loss_win_15_spm = -(
        torch.log(win15_prob + eps) * label_01_15
        + torch.log(1 - win15_prob + eps) * label_15_0
    ).mean()

    loss_win_30_spm = -(
        torch.log(win30_prob + eps) * label_01_30_sum
        + torch.log(1 - win30_prob + eps) * label_30_0
    ).mean()

    loss_win_60_spm = -(
        torch.log(win60_prob + eps) * label_01_60_sum
        + torch.log(1 - win60_prob + eps) * label_60_0
    ).mean()

    # ==================== 第三套: cv_spm ====================
    # 对应 TF: loss_cv_spm
    # 关键: 延迟正样本 (label_11) 也监督 cv_logits 学到"会转化"
    # 真负样本 (label_10) 监督 cv_logits 学到"不转化"
    loss_cv_spm = -(
        torch.log(cv_prob + eps) * label_01        # 窗口内正样本
        + torch.log(cv_prob + eps) * label_11      # 延迟正样本 (重要!)
        + torch.log(1 - cv_prob + eps) * label_10  # 真负样本
    ).mean()

    # ==================== 总损失 (subloss=1, TF 原版最优) ====================
    loss = (
        0.1  * loss_win_15
        + 0.05 * loss_win_30
        + 0.05 * loss_win_60
        + 0.1  * loss_win_15_spm
        + 0.05 * loss_win_30_spm
        + 0.05 * loss_win_60_spm
        + 0.6  * loss_cv_spm
    )

    return {
        "loss": loss,
        "loss_win_15": loss_win_15,
        "loss_win_30": loss_win_30,
        "loss_win_60": loss_win_60,
        "loss_win_15_spm": loss_win_15_spm,
        "loss_win_30_spm": loss_win_30_spm,
        "loss_win_60_spm": loss_win_60_spm,
        "loss_cv_spm": loss_cv_spm,
    }


# ============================================================================
# 损失函数工厂
# ============================================================================

LOSS_FUNCTIONS = {
    "cross_entropy_loss": cross_entropy_loss,
    "fnw_loss": fnw_loss,
    "fnc_loss": fnc_loss,
    "exp_delay_loss": exp_delay_loss,
    "delay_tn_dp_loss": delay_tn_dp_loss,
    "esdfm_loss": esdfm_loss,
    "esdfm_importance_weight_loss": esdfm_importance_weight_loss,
    "fsiw_loss": fsiw_loss,
    "delay_3class_loss": delay_3class_loss,
    "delay_win_time_loss": delay_win_time_loss,
    "delay_win_select_loss": delay_win_select_loss,
}


def get_loss_fn(name):
    """获取损失函数"""
    if name not in LOSS_FUNCTIONS:
        raise NotImplementedError(f"损失函数 {name} 未实现")
    return LOSS_FUNCTIONS[name]


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    batch_size = 32
    
    # 测试 cross_entropy_loss
    outputs = {"logits": torch.randn(batch_size, 1)}
    targets = {"label": torch.randint(0, 2, (batch_size, 1)).float()}
    loss = cross_entropy_loss(targets, outputs)
    print(f"cross_entropy_loss: {loss['loss'].item():.4f}")
    
    # 测试 exp_delay_loss
    outputs = {
        "logits": torch.randn(batch_size, 1),
        "log_lamb": torch.randn(batch_size, 1)
    }
    targets = {"label": torch.rand(batch_size, 2) * torch.tensor([1, 3600])}
    loss = exp_delay_loss(targets, outputs)
    print(f"exp_delay_loss: {loss['loss'].item():.4f}")
    
    # 测试 delay_win_select_loss
    outputs = {
        "cv_logits": torch.randn(batch_size, 1),
        "time15_logits": torch.randn(batch_size, 1),
        "time30_logits": torch.randn(batch_size, 1),
        "time60_logits": torch.randn(batch_size, 1),
    }
    targets = {"label": torch.rand(batch_size, 14)}
    loss = delay_win_select_loss(targets, outputs)
    print(f"delay_win_select_loss: {loss['loss'].item():.4f}")
