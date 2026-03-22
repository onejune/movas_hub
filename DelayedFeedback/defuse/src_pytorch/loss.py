"""
DEFUSE 损失函数

包含:
- DEFUSE loss: 4-component label correction (IP/FN/RN/DP)
- ES-DFM loss: Simple importance weighting
- Pretrain loss: BCE for tn/dp heads
"""

import torch
import torch.nn.functional as F


def stable_log1pex(x: torch.Tensor) -> torch.Tensor:
    """
    计算 log(1 + exp(-x)) = softplus(-x) = -log(sigmoid(x))
    
    数值稳定版本
    """
    return F.softplus(-x)


def pretrain_loss(tn_logits: torch.Tensor,
                  dp_logits: torch.Tensor,
                  tn_label: torch.Tensor,
                  dp_label: torch.Tensor) -> torch.Tensor:
    """
    Stage 1: 预训练 tn/dp 分类器
    
    使用 BCE loss 训练 tn 和 dp 头
    """
    tn_loss = F.binary_cross_entropy_with_logits(tn_logits, tn_label)
    dp_loss = F.binary_cross_entropy_with_logits(dp_logits, dp_label)
    return tn_loss + dp_loss


def defuse_loss(cv_logits: torch.Tensor,
                tn_prob: torch.Tensor,
                dp_prob: torch.Tensor,
                label: torch.Tensor,
                eps: float = 1e-7) -> torch.Tensor:
    """
    DEFUSE loss: 4-component label correction
    
    样本类型:
    - IP (In-window Positive): 窗口内转化, label=1
    - FN (Fake Negative): 假负样本 (实际会转化但还没观察到), label=0
    - RN (Real Negative): 真负样本, label=0
    - DP (Delayed Positive): 延迟正样本 (窗口外转化), label=0
    
    Loss 公式:
    - loss1 (IP): -log(σ(x)) * (1 + dp_prob)
    - loss2 (FN): zi * -log(σ(x)) * dp_prob
    - loss3 (RN): (1-zi) * -log(1-σ(x)) * (1 + dp_prob)
    - loss4 (DP): -log(σ(x)) * 1
    
    其中 zi = 1 - tn_prob = P(fake negative)
    
    Args:
        cv_logits: CVR 预测 logits
        tn_prob: True Negative 概率 (detached)
        dp_prob: Delayed Positive 概率 (detached)
        label: 观察到的标签 (0 或 1)
        eps: 数值稳定性
    
    Returns:
        标量 loss
    """
    # zi = P(fake negative) = 1 - P(true negative)
    zi = torch.clamp(1.0 - tn_prob, eps, 1.0 - eps)
    dp_prob = torch.clamp(dp_prob, eps, 1.0 - eps)
    
    # -log(σ(x)) = log(1 + exp(-x)) = stable_log1pex(x)
    neg_log_sigmoid = stable_log1pex(cv_logits)
    
    # -log(1 - σ(x)) = x + log(1 + exp(-x)) = x + stable_log1pex(x)
    neg_log_1_minus_sigmoid = cv_logits + stable_log1pex(cv_logits)
    
    # 正样本 (label=1): loss1 + loss4
    # loss1: -log(σ(x)) * (1 + dp_prob)
    # loss4: -log(σ(x)) * 1
    pos_loss = neg_log_sigmoid * (1.0 + dp_prob) + neg_log_sigmoid
    
    # 负样本 (label=0): loss2 + loss3
    # loss2: zi * -log(σ(x)) * dp_prob
    # loss3: (1-zi) * -log(1-σ(x)) * (1 + dp_prob)
    neg_loss = (zi * neg_log_sigmoid * dp_prob + 
                (1.0 - zi) * neg_log_1_minus_sigmoid * (1.0 + dp_prob))
    
    # 组合
    loss = label * pos_loss + (1.0 - label) * neg_loss
    
    return loss.mean()


def esdfm_loss(cv_logits: torch.Tensor,
               tn_prob: torch.Tensor,
               dp_prob: torch.Tensor,
               label: torch.Tensor,
               eps: float = 1e-7) -> torch.Tensor:
    """
    ES-DFM loss: Simple importance weighting
    
    公式:
    - pos_weight = 1 + dp_prob
    - neg_weight = (1 + dp_prob) * tn_prob
    - loss = pos_loss * pos_weight * z + neg_loss * neg_weight * (1-z)
    
    Args:
        cv_logits: CVR 预测 logits
        tn_prob: True Negative 概率 (detached)
        dp_prob: Delayed Positive 概率 (detached)
        label: 观察到的标签 (0 或 1)
        eps: 数值稳定性
    
    Returns:
        标量 loss
    """
    tn_prob = torch.clamp(tn_prob, eps, 1.0 - eps)
    dp_prob = torch.clamp(dp_prob, eps, 1.0 - eps)
    
    # 权重
    pos_weight = 1.0 + dp_prob
    neg_weight = (1.0 + dp_prob) * tn_prob
    
    # 正负样本 loss
    neg_log_sigmoid = stable_log1pex(cv_logits)  # -log(σ(x))
    neg_log_1_minus_sigmoid = cv_logits + stable_log1pex(cv_logits)  # -log(1-σ(x))
    
    pos_loss = neg_log_sigmoid * pos_weight
    neg_loss = neg_log_1_minus_sigmoid * neg_weight
    
    # 组合
    loss = label * pos_loss + (1.0 - label) * neg_loss
    
    return loss.mean()
