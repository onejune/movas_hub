"""
DEFER 损失函数

完全对齐 src_tf_github/loss.py 实现

修正版本:
1. stop_gradient 放在 time_prop 上 (不是 cv_prop)
2. SPM 损失的负样本 mask 正确构造
3. 接口与 MetaSpore PyTorchEstimator 兼容: loss_fn(output, target) -> scalar
"""
import torch
import torch.nn.functional as F
from typing import Dict, Optional


# ============================================================================
# 数值稳定的 Sigmoid
# ============================================================================

def stable_sigmoid(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """数值稳定的 sigmoid"""
    return torch.clamp(torch.sigmoid(x), eps, 1.0 - eps)


# ============================================================================
# WinAdapt 损失 (核心) - 完全对齐 TF 原版
# ============================================================================

def delay_win_select_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    minibatch=None,
    **kwargs
) -> tuple:
    """
    WinAdapt 损失函数 - 适配 MetaSpore PyTorchEstimator
    
    Args:
        logits: (batch, 4) - 模型输出 [cv_logit, time_w1, time_w2, time_w3]
        labels: (batch, 14) - 14 维标签 (从 input_label_column 解析)
        minibatch: MetaSpore minibatch DataFrame，包含:
            - cv_label: 原始二分类标签 (是否最终转化)
            - 其他特征列
        **kwargs: 其他参数 (task_to_id_map, task_weights_dict 等)
    
    Returns:
        (loss, task_loss): tuple of scalar tensors (MetaSpore 要求返回两个值)
    
    标签格式 (14 维):
    - z[:, 0]: label_11 - 延迟正样本 (delayed positive)
    - z[:, 1]: label_10 - 真负样本 (true negative)
    - z[:, 2]: label_01_15 - 窗口1内正样本
    - z[:, 3]: label_01_30 - 窗口2内正样本 (增量)
    - z[:, 4]: label_01_60 - 窗口3内正样本 (增量)
    - z[:, 5]: label_01_30_sum - 窗口2累积
    - z[:, 6]: label_01_60_sum - 窗口3累积
    - z[:, 7]: label_01_30_mask - 窗口2 mask
    - z[:, 8]: label_01_60_mask - 窗口3 mask
    - z[:, 9]: label_00 - 窗口内负样本 (未观测完)
    - z[:, 10]: label_01 - 窗口内正样本 (所有)
    - z[:, 11]: label_11_15 - 延迟正 w1-w2
    - z[:, 12]: label_11_30 - 延迟正 w2-w3
    - z[:, 13]: label_11_60 - 延迟正 >w3
    
    关键点:
    1. stop_gradient 放在 time_prop 上，不是 cv_prop
    2. SPM 损失的负样本 = 延迟正样本 + 真负样本 (排除 label_00)
    3. CV 损失: 正例=label01+label11, 负例=label10 (排除 label_00)
    4. 可选：从 minibatch['cv_label'] 获取原始二分类标签
    """
    # DEBUG: 打印输入形状，确认函数被调用
    print(f"[DEBUG delay_win_select_loss] logits.shape={logits.shape}, labels.shape={labels.shape}")
    
    # 可选：从 minibatch 中提取原始 cv_label (单值标签)
    cv_label_from_minibatch = None
    if minibatch is not None and 'cv_label' in minibatch.columns:
        cv_label_from_minibatch = torch.tensor(
            minibatch['cv_label'].values.astype('float32'),
            device=logits.device
        )
        print(f"[DEBUG delay_win_select_loss] cv_label from minibatch: {cv_label_from_minibatch.shape}")
    
    z = labels  # (batch, 14)
    
    eps = 1e-7
    
    # ========== 解析标签 ==========
    label_11 = z[:, 0]           # delayed positive (>w3)
    label_10 = z[:, 1]           # true negative
    label_01_15 = z[:, 2]        # window 1 positive
    label_01_30_sum = z[:, 5]    # window 2 cumulative (w1 + w2)
    label_01_60_sum = z[:, 6]    # window 3 cumulative (w1 + w2 + w3)
    label_01 = z[:, 10]          # window positive (all)
    label_11_15 = z[:, 11]       # delayed positive w1-w2
    label_11_30 = z[:, 12]       # delayed positive w2-w3
    label_11_60 = z[:, 13]       # delayed positive >w3
    
    # ========== 解析 logits ==========
    cv_logit = logits[:, 0]
    time_15_logit = logits[:, 1]
    time_30_logit = logits[:, 2]
    time_60_logit = logits[:, 3]
    
    # ========== 计算概率 ==========
    cv_prob = stable_sigmoid(cv_logit)
    time_15_prob = stable_sigmoid(time_15_logit)
    time_30_prob = stable_sigmoid(time_30_logit)
    time_60_prob = stable_sigmoid(time_60_logit)
    
    # ========== stop_gradient 版本 (关键：stop 在 time_prop 上) ==========
    time_15_prob_stop = time_15_prob.detach()
    time_30_prob_stop = time_30_prob.detach()
    time_60_prob_stop = time_60_prob.detach()
    
    # ========== 窗口概率 ==========
    # 正常版本 (用于 SPM loss)
    win_prob_15 = cv_prob * time_15_prob
    win_prob_30 = cv_prob * time_30_prob
    win_prob_60 = cv_prob * time_60_prob
    
    # stop_gradient 版本 (用于窗口 loss，只更新 cv_head)
    win_prob_15_stop = cv_prob * time_15_prob_stop
    win_prob_30_stop = cv_prob * time_30_prob_stop
    win_prob_60_stop = cv_prob * time_60_prob_stop
    
    # ========== 窗口损失 (stop_gradient on time_prop) ==========
    loss_win_15 = -torch.mean(
        label_01_15 * torch.log(win_prob_15_stop + eps) +
        (1 - label_01_15) * torch.log(1 - win_prob_15_stop + eps)
    )
    
    loss_win_30 = -torch.mean(
        label_01_30_sum * torch.log(win_prob_30_stop + eps) +
        (1 - label_01_30_sum) * torch.log(1 - win_prob_30_stop + eps)
    )
    
    loss_win_60 = -torch.mean(
        label_01_60_sum * torch.log(win_prob_60_stop + eps) +
        (1 - label_01_60_sum) * torch.log(1 - win_prob_60_stop + eps)
    )
    
    # ========== SPM 损失 (关键：正确的负样本 mask) ==========
    # 窗口1的负样本: 所有延迟正样本 + 真负样本
    label_15_neg = torch.clamp(label_11_15 + label_11_30 + label_11_60 + label_10, max=1.0)
    
    # 窗口2的负样本: w2之后的延迟正样本 + 真负样本
    label_30_neg = torch.clamp(label_11_30 + label_11_60 + label_10, max=1.0)
    
    # 窗口3的负样本: w3之后的延迟正样本 + 真负样本
    label_60_neg = torch.clamp(label_11_60 + label_10, max=1.0)
    
    loss_win_15_spm = -torch.mean(
        label_01_15 * torch.log(win_prob_15 + eps) +
        label_15_neg * torch.log(1 - win_prob_15 + eps)
    )
    
    loss_win_30_spm = -torch.mean(
        label_01_30_sum * torch.log(win_prob_30 + eps) +
        label_30_neg * torch.log(1 - win_prob_30 + eps)
    )
    
    loss_win_60_spm = -torch.mean(
        label_01_60_sum * torch.log(win_prob_60 + eps) +
        label_60_neg * torch.log(1 - win_prob_60 + eps)
    )
    
    # ========== CV 损失 ==========
    # 正样本: label_01 (窗口内正) + label_11 (延迟正)
    # 负样本: label_10 (真负)
    loss_cv_spm = -torch.mean(
        label_01 * torch.log(cv_prob + eps) +
        label_11 * torch.log(cv_prob + eps) +
        label_10 * torch.log(1 - cv_prob + eps)
    )
    
    # ========== 总损失 (subloss=1 配置) ==========
    loss = (
        0.10 * loss_win_15 +
        0.05 * loss_win_30 +
        0.05 * loss_win_60 +
        0.10 * loss_win_15_spm +
        0.05 * loss_win_30_spm +
        0.05 * loss_win_60_spm +
        0.60 * loss_cv_spm
    )
    
    # MetaSpore 要求返回 (loss, task_loss) tuple
    return loss, loss


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试 DEFER 损失函数")
    print("=" * 60)
    
    batch_size = 32
    
    # 构造测试数据
    torch.manual_seed(42)
    
    labels = torch.zeros(batch_size, 14)
    
    # 随机分配样本类型
    for i in range(batch_size):
        r = torch.rand(1).item()
        if r < 0.7:  # 70% 真负样本
            labels[i, 1] = 1  # label_10
        elif r < 0.8:  # 10% 窗口1正样本
            labels[i, 2] = 1  # label_01_15
            labels[i, 10] = 1  # label_01
        elif r < 0.85:  # 5% 窗口2正样本
            labels[i, 3] = 1  # label_01_30
            labels[i, 5] = 1  # label_01_30_sum
            labels[i, 10] = 1  # label_01
        elif r < 0.9:  # 5% 窗口3正样本
            labels[i, 4] = 1  # label_01_60
            labels[i, 6] = 1  # label_01_60_sum
            labels[i, 10] = 1  # label_01
        else:  # 10% 延迟正样本
            labels[i, 0] = 1  # label_11
            if torch.rand(1).item() < 0.33:
                labels[i, 11] = 1  # label_11_15
            elif torch.rand(1).item() < 0.5:
                labels[i, 12] = 1  # label_11_30
            else:
                labels[i, 13] = 1  # label_11_60
    
    # 设置 mask
    labels[:, 7] = 1.0  # label_01_30_mask
    labels[:, 8] = 1.0  # label_01_60_mask
    
    # 累积值
    labels[:, 5] = torch.clamp(labels[:, 2] + labels[:, 3], max=1.0)  # label_01_30_sum
    labels[:, 6] = torch.clamp(labels[:, 2] + labels[:, 3] + labels[:, 4], max=1.0)  # label_01_60_sum
    
    logits = torch.randn(batch_size, 4, requires_grad=True)
    
    print("\n标签分布:")
    print(f"  label_11 (延迟正): {labels[:, 0].sum().item():.0f}")
    print(f"  label_10 (真负): {labels[:, 1].sum().item():.0f}")
    print(f"  label_01_15 (窗口1正): {labels[:, 2].sum().item():.0f}")
    print(f"  label_01 (所有窗口正): {labels[:, 10].sum().item():.0f}")
    
    print("\n损失计算:")
    loss = delay_win_select_loss(logits, labels)
    print(f"  loss: {loss.item():.4f}")
    
    # 测试梯度
    print("\n梯度测试:")
    loss.backward()
    print(f"  logits.grad shape: {logits.grad.shape}")
    print(f"  logits.grad mean: {logits.grad.mean().item():.6f}")
    
    print("\n测试通过!")
