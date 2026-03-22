"""
DEFER 损失函数 (Delayed Feedback Modeling)

================================================================================
背景
================================================================================
广告转化存在延迟反馈问题: 用户点击广告后，转化可能在几小时甚至几天后发生。
传统做法是等待足够长时间 (如7天) 才能确定标签，但这会导致数据延迟。
DEFER 方法通过时间窗口对样本分类，结合概率建模来处理延迟反馈。

================================================================================
核心思想
================================================================================
将转化概率分解为两部分:
    P(转化) = P(最终会转化) × P(在窗口内转化 | 最终会转化)
            = cv_prob × time_prob

模型输出 4 个 logit:
- cv_logit: 最终转化概率 (是否会转化)
- time_w1_logit: 在窗口1内转化的概率
- time_w2_logit: 在窗口2内转化的概率
- time_w3_logit: 在窗口3内转化的概率

================================================================================
提供两个版本
================================================================================
1. delay_win_select_loss_v1:
   - 完全对齐 src_tf_github/loss.py 实现
   - 14 维标签，包含 SPM loss
   - stop_gradient 放在 time_prop 上
   - 适用于: defer_v1 (短窗口 15/30/60min)

2. delay_win_select_loss_v2:
   - 简化版，8 维标签
   - 支持 observable mask (处理样本年龄不足的情况)
   - stop_gradient 放在 cv_prob 上
   - 适用于: defer_v2 (长窗口 24/48/72h)

接口统一: loss_fn(logits, labels, minibatch=None) -> (loss, loss)
"""
import torch
import torch.nn.functional as F
from typing import Dict, Optional


# ============================================================================
# 数值稳定的 Sigmoid
# ============================================================================

def stable_sigmoid(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """数值稳定的 sigmoid，避免 log(0) 和 log(1) 的数值问题"""
    return torch.clamp(torch.sigmoid(x), eps, 1.0 - eps)


# ============================================================================
# WinAdapt 损失 (核心) - 完全对齐 TF 原版
# ============================================================================

def delay_win_select_loss_v1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    minibatch=None,
    **kwargs
) -> tuple:
    """
    WinAdapt 损失函数 - 适配 MetaSpore PyTorchEstimator
    
    ============================================================================
    输入参数
    ============================================================================
    Args:
        logits: (batch, 4) - 模型输出
            - logits[:, 0]: cv_logit (最终转化概率)
            - logits[:, 1]: time_w1_logit (窗口1内转化概率)
            - logits[:, 2]: time_w2_logit (窗口2内转化概率)
            - logits[:, 3]: time_w3_logit (窗口3内转化概率)
        
        labels: (batch, 14) - 14 维标签 (fallback，通常不使用)
        
        minibatch: MetaSpore minibatch DataFrame，包含:
            - defer_label: 14 维标签的 JSON 字符串 (优先使用)
            - label: 原始 1 维标签 (用于 AUC 评估)
            - 其他特征列
        
        **kwargs: 其他参数 (task_to_id_map, task_weights_dict 等)
    
    Returns:
        (loss, task_loss): tuple of scalar tensors (MetaSpore 要求返回两个值)
    
    ============================================================================
    14 维标签格式 (从 defer_label JSON 解析)
    ============================================================================
    索引  名称              含义                                    用途
    ----  ----              ----                                    ----
    [0]   label_11          延迟正样本 (label=1, diff>delay)        SPM loss 正样本
    [1]   label_10          真负样本 (label=0)                      CV/SPM loss 负样本
    [2]   label_01_win1     窗口1内正样本 (diff<=win1)              窗口1 loss
    [3]   label_01_win2     窗口2内正样本 (win1<diff<=win2)         窗口2 loss (增量)
    [4]   label_01_win3     窗口3内正样本 (win2<diff<=win3)         窗口3 loss (增量)
    [5]   label_01_sum12    窗口1+2累积正样本                       窗口2 loss
    [6]   label_01_sum123   窗口1+2+3累积正样本                     窗口3 loss
    [7]   mask_win2         窗口2 mask (恒为1)                      保留
    [8]   mask_win3         窗口3 mask (恒为1)                      保留
    [9]   reserved          保留位 (恒为0)                          未使用
    [10]  label_01          所有窗口内正样本 (同[6])                CV loss 正样本
    [11]  label_11_w1       延迟正细分1 (delay<diff<=delay+win3)   SPM loss 细分
    [12]  label_11_w2       延迟正细分2                             SPM loss 细分
    [13]  label_11_w3       延迟正细分3 (diff>delay+2*win3)        SPM loss 细分
    
    ============================================================================
    损失函数组成
    ============================================================================
    总损失 = 0.10 × loss_win_15      (窗口1 loss，stop_gradient on time_prob)
           + 0.05 × loss_win_30      (窗口2 loss，stop_gradient on time_prob)
           + 0.05 × loss_win_60      (窗口3 loss，stop_gradient on time_prob)
           + 0.10 × loss_win_15_spm  (窗口1 SPM loss，无 stop_gradient)
           + 0.05 × loss_win_30_spm  (窗口2 SPM loss，无 stop_gradient)
           + 0.05 × loss_win_60_spm  (窗口3 SPM loss，无 stop_gradient)
           + 0.60 × loss_cv_spm      (CV loss，主任务)
    
    ============================================================================
    关键设计点
    ============================================================================
    1. stop_gradient 放在 time_prob 上，不是 cv_prob
       - 窗口 loss 只更新 cv_head，不更新 time_head
       - 这样 time_head 只通过 SPM loss 更新
    
    2. SPM 损失的负样本定义
       - 窗口1负样本 = 所有延迟正样本 + 真负样本
       - 窗口2负样本 = w2之后的延迟正样本 + 真负样本
       - 窗口3负样本 = w3之后的延迟正样本 + 真负样本
       - 注意：排除 label_00 (未观测完样本)
    
    3. CV 损失的样本定义
       - 正样本 = label_01 (窗口内正) + label_11 (延迟正)
       - 负样本 = label_10 (真负)
       - 排除 label_00 (未观测完样本)
    
    4. 标签来源
       - 优先从 minibatch['defer_label'] 解析 JSON 获取 14 维标签
       - fallback 到 labels 参数 (通常不使用)
    """
    # DEBUG: 打印输入形状，确认函数被调用
    #print(f"[DEBUG delay_win_select_loss] logits.shape={logits.shape}, labels.shape={labels.shape}")
    
    # ========== 从 minibatch 中提取 defer_label (14 维) ==========
    # 优先使用 minibatch['defer_label']，如果没有则使用 labels 参数
    if minibatch is not None and 'defer_label' in minibatch.columns:
        import json
        defer_labels_raw = minibatch['defer_label'].values
        # defer_label 是 JSON 字符串，需要解析
        z = torch.tensor(
            [json.loads(v) if isinstance(v, str) else v for v in defer_labels_raw],
            dtype=torch.float32,
            device=logits.device
        )
    else:
        z = labels  # (batch, 14) - fallback to labels parameter
    
    eps = 1e-7
    
    # ==========================================================================
    # Step 1: 解析 14 维标签
    # ==========================================================================
    # 标签含义参见 docstring 中的表格
    label_11 = z[:, 0]           # 延迟正样本: label=1, diff>delay
    label_10 = z[:, 1]           # 真负样本: label=0
    label_01_15 = z[:, 2]        # 窗口1内正样本: label=1, diff<=win1
    label_01_30_sum = z[:, 5]    # 窗口1+2累积: label=1, diff<=win2
    label_01_60_sum = z[:, 6]    # 窗口1+2+3累积: label=1, diff<=win3
    label_01 = z[:, 10]          # 所有窗口内正样本 (同 label_01_60_sum)
    label_11_15 = z[:, 11]       # 延迟正细分1: delay<diff<=delay+win3
    label_11_30 = z[:, 12]       # 延迟正细分2: delay+win3<diff<=delay+2*win3
    label_11_60 = z[:, 13]       # 延迟正细分3: diff>delay+2*win3
    
    # ==========================================================================
    # Step 2: 解析模型输出 (4 个 logit)
    # ==========================================================================
    cv_logit = logits[:, 0]       # 最终转化概率 logit
    time_15_logit = logits[:, 1]  # 窗口1内转化概率 logit
    time_30_logit = logits[:, 2]  # 窗口2内转化概率 logit
    time_60_logit = logits[:, 3]  # 窗口3内转化概率 logit
    
    # ==========================================================================
    # Step 3: 计算概率 (sigmoid)
    # ==========================================================================
    cv_prob = stable_sigmoid(cv_logit)           # P(最终会转化)
    time_15_prob = stable_sigmoid(time_15_logit) # P(在win1内转化 | 会转化)
    time_30_prob = stable_sigmoid(time_30_logit) # P(在win2内转化 | 会转化)
    time_60_prob = stable_sigmoid(time_60_logit) # P(在win3内转化 | 会转化)
    
    # ==========================================================================
    # Step 4: stop_gradient 版本 (关键设计点)
    # ==========================================================================
    # 对 time_prob 做 stop_gradient，这样窗口 loss 只更新 cv_head
    # time_head 只通过 SPM loss 更新
    time_15_prob_stop = time_15_prob.detach()
    time_30_prob_stop = time_30_prob.detach()
    time_60_prob_stop = time_60_prob.detach()
    
    # ==========================================================================
    # Step 5: 计算窗口概率 P(在窗口内转化) = cv_prob × time_prob
    # ==========================================================================
    # 正常版本 (用于 SPM loss，梯度同时流向 cv_head 和 time_head)
    win_prob_15 = cv_prob * time_15_prob
    win_prob_30 = cv_prob * time_30_prob
    win_prob_60 = cv_prob * time_60_prob
    
    # stop_gradient 版本 (用于窗口 loss，梯度只流向 cv_head)
    win_prob_15_stop = cv_prob * time_15_prob_stop
    win_prob_30_stop = cv_prob * time_30_prob_stop
    win_prob_60_stop = cv_prob * time_60_prob_stop
    
    # ==========================================================================
    # Step 6: 窗口损失 (Binary Cross Entropy, stop_gradient on time_prob)
    # ==========================================================================
    # 这些 loss 只更新 cv_head，让 cv_head 学习窗口内转化的模式
    # 公式: -[y*log(p) + (1-y)*log(1-p)]
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
    
    # ==========================================================================
    # Step 7: SPM 损失 (关键：正确的负样本 mask)
    # ==========================================================================
    # SPM (Selective Prediction Mask) loss 的负样本需要仔细设计:
    # - 窗口1负样本 = 所有延迟正样本 + 真负样本 (排除 label_00)
    # - 窗口2负样本 = w2之后的延迟正样本 + 真负样本
    # - 窗口3负样本 = w3之后的延迟正样本 + 真负样本
    # 
    # 为什么要这样设计？
    # - 延迟正样本在窗口内被误判为负样本，但实际是正样本
    # - 我们希望模型预测这些样本的窗口概率较低 (因为它们确实不在窗口内转化)
    # - 所以把它们作为 SPM loss 的负样本
    
    # 窗口1的负样本: 所有延迟正样本 + 真负样本
    label_15_neg = torch.clamp(label_11_15 + label_11_30 + label_11_60 + label_10, max=1.0)
    
    # 窗口2的负样本: w2之后的延迟正样本 + 真负样本
    label_30_neg = torch.clamp(label_11_30 + label_11_60 + label_10, max=1.0)
    
    # 窗口3的负样本: w3之后的延迟正样本 + 真负样本
    label_60_neg = torch.clamp(label_11_60 + label_10, max=1.0)
    
    # SPM loss 公式: -[正样本*log(p) + 负样本*log(1-p)]
    # 注意：这里正样本和负样本不互斥，可能有样本两者都不是 (label_00)
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
    
    # ==========================================================================
    # Step 8: CV 损失 (主任务损失)
    # ==========================================================================
    # CV (Conversion) loss 预测最终是否会转化
    # - 正样本 = label_01 (窗口内正) + label_11 (延迟正)
    # - 负样本 = label_10 (真负)
    # - 排除 label_00 (未观测完样本)，因为它们的最终标签不确定
    loss_cv_spm = -torch.mean(
        label_01 * torch.log(cv_prob + eps) +          # 窗口内正样本
        label_11 * torch.log(cv_prob + eps) +          # 延迟正样本
        label_10 * torch.log(1 - cv_prob + eps)        # 真负样本
    )
    
    # ==========================================================================
    # Step 9: 总损失 (加权组合)
    # ==========================================================================
    # 权重配置 (subloss=1):
    # - CV loss 权重最大 (0.60)，因为这是主任务
    # - 窗口 loss 权重较小，起辅助作用
    # - win1 权重 > win2 > win3，因为短窗口的标签更可靠
    loss = (
        0.10 * loss_win_15 +      # 窗口1 loss (stop_gradient)
        0.05 * loss_win_30 +      # 窗口2 loss (stop_gradient)
        0.05 * loss_win_60 +      # 窗口3 loss (stop_gradient)
        0.10 * loss_win_15_spm +  # 窗口1 SPM loss
        0.05 * loss_win_30_spm +  # 窗口2 SPM loss
        0.05 * loss_win_60_spm +  # 窗口3 SPM loss
        0.60 * loss_cv_spm        # CV loss (主任务)
    )
    
    # MetaSpore 要求返回 (loss, task_loss) tuple
    return loss, loss


def delay_win_select_loss_v2(
    logits: torch.Tensor,
    labels: torch.Tensor,
    minibatch=None,
    cv_loss_weight: float = 0.8,
    win_loss_weight: float = 0.2,
    **kwargs
) -> tuple:
    """
    WinAdapt 多窗口自适应损失 v2 - 简化版，与 MetaSpore 接口对齐
    
    ============================================================================
    核心思想
    ============================================================================
    - 模型输出 4 个 logits: cv_logit + 3 个时间窗口条件概率 logit
    - P(窗口内转化) = P(cv) × P(t ≤ window | cv)
    - 对每个窗口的联合概率计算 BCE loss
    - 同时对最终转化标签 (oracle) 计算 CVR loss
    
    ============================================================================
    与 v1 (delay_win_select_loss) 的区别
    ============================================================================
    | 维度          | v1 (14维)                    | v2 (8维)                      |
    |---------------|------------------------------|-------------------------------|
    | 标签格式      | 14维，包含延迟正样本细分     | 8维，简化为窗口标签+mask      |
    | 窗口标签      | 增量式 (各窗口不重叠)        | 累积式 (各窗口包含之前窗口)   |
    | stop_gradient | 在 time_prob 上              | 在 cv_prob 上                 |
    | SPM loss      | 有 (复杂的负样本 mask)       | 无 (简化为 BCE)               |
    | 权重配置      | 7 个子 loss 独立权重         | 2 个权重: cv + win            |
    
    ============================================================================
    输入参数
    ============================================================================
    Args:
        logits: (batch, 4) - 模型输出
            - logits[:, 0]: cv_logit (最终转化概率)
            - logits[:, 1]: time_w1_logit (窗口1内转化概率)
            - logits[:, 2]: time_w2_logit (窗口2内转化概率)
            - logits[:, 3]: time_w3_logit (窗口3内转化概率)
        
        labels: (batch, 8) - 8 维标签 (fallback，通常不使用)
        
        minibatch: MetaSpore minibatch DataFrame，包含:
            - defer_label: 8 维标签的 JSON 字符串 (优先使用)
            - label: 原始 1 维标签 (用于 AUC 评估)
        
        cv_loss_weight: CVR loss 权重，默认 0.8
        win_loss_weight: 窗口 loss 权重，默认 0.2
    
    Returns:
        (loss, loss): tuple of scalar tensors (MetaSpore 要求返回两个值)
    
    ============================================================================
    8 维标签格式 (从 defer_label JSON 解析)
    ============================================================================
    索引  名称              含义                                    用途
    ----  ----              ----                                    ----
    [0]   label_win1        label=1 AND diff<=win1 (累积)           窗口1 loss
    [1]   label_win2        label=1 AND diff<=win2 (累积)           窗口2 loss
    [2]   label_win3        label=1 AND diff<=win3 (累积)           窗口3 loss
    [3]   label_oracle      label=1 AND diff<=delay                 CV loss (主任务)
    [4]   observable_win1   sample_age >= win1                      窗口1 mask
    [5]   observable_win2   sample_age >= win2                      窗口2 mask
    [6]   observable_win3   sample_age >= win3                      窗口3 mask
    [7]   observable_oracle 1.0 (恒定)                              oracle mask
    
    ============================================================================
    损失函数组成
    ============================================================================
    总损失 = cv_loss_weight × loss_cv + win_loss_weight × mean(loss_win)
           = 0.8 × BCE(cv_logit, label_oracle)
           + 0.2 × mean([BCE(win_prob_i, label_i) * observable_i])
    
    其中:
    - win_prob_i = cv_prob.detach() × time_prob_i
    - stop_gradient 在 cv_prob 上，所以窗口 loss 只更新 time head
    - CV loss 更新 cv head
    
    ============================================================================
    关键设计点
    ============================================================================
    1. stop_gradient 在 cv_prob 上 (与 v1 相反)
       - 窗口 loss 只更新 time_head
       - CV loss 更新 cv_head
       - 这样两个 head 有明确的分工
    
    2. 累积式窗口标签 (与 v1 的增量式不同)
       - label_win1 = 1 if diff <= win1
       - label_win2 = 1 if diff <= win2 (包含 win1 内的正样本)
       - label_win3 = 1 if diff <= win3 (包含 win1, win2 内的正样本)
    
    3. observable mask 处理实时数据
       - 如果样本年龄 < 窗口时间，则该样本不可观察
       - 不可观察的样本不参与该窗口的 loss 计算
    
    4. 简化的权重配置
       - 只有 cv_loss_weight 和 win_loss_weight 两个参数
       - 默认 0.8 + 0.2 = 1.0
    """
    eps = 1e-7
    
    # ==========================================================================
    # Step 1: 从 minibatch 中提取 defer_label (8 维)
    # ==========================================================================
    if minibatch is not None and 'defer_label' in minibatch.columns:
        import json
        defer_labels_raw = minibatch['defer_label'].values
        z = torch.tensor(
            [json.loads(v) if isinstance(v, str) else v for v in defer_labels_raw],
            dtype=torch.float32,
            device=logits.device
        )
    else:
        z = labels  # fallback to labels parameter
    
    # ==========================================================================
    # Step 2: 解析模型输出 (4 个 logit)
    # ==========================================================================
    cv_logit = logits[:, 0]       # 最终转化概率 logit
    time_logits = [
        logits[:, 1],             # 窗口1 条件概率 logit
        logits[:, 2],             # 窗口2 条件概率 logit
        logits[:, 3],             # 窗口3 条件概率 logit
    ]
    
    # ==========================================================================
    # Step 3: 解析 8 维标签
    # ==========================================================================
    label_windows = [
        z[:, 0],                  # label_win1 (累积)
        z[:, 1],                  # label_win2 (累积)
        z[:, 2],                  # label_win3 (累积)
    ]
    label_oracle = z[:, 3]        # 最终转化标签
    
    observable_windows = [
        z[:, 4],                  # observable_win1
        z[:, 5],                  # observable_win2
        z[:, 6],                  # observable_win3
    ]
    
    # ==========================================================================
    # Step 4: 计算概率
    # ==========================================================================
    cv_prob = stable_sigmoid(cv_logit)
    time_probs = [stable_sigmoid(tl) for tl in time_logits]
    
    # stop_gradient 在 cv_prob 上: 窗口 loss 只更新 time head
    cv_prob_stopped = cv_prob.detach()
    
    # ==========================================================================
    # Step 5: 窗口 loss (更新 time head)
    # ==========================================================================
    win_losses = []
    for i in range(3):
        win_label = label_windows[i]
        observable = observable_windows[i]
        time_prob = time_probs[i]
        
        # 联合概率: P(win) = P(cv).detach() × P(t ≤ w | cv)
        win_prob = cv_prob_stopped * time_prob
        win_prob_clamp = torch.clamp(win_prob, eps, 1.0 - eps)
        
        # BCE loss (逐样本)
        loss_per_sample = -(
            win_label * torch.log(win_prob_clamp) +
            (1 - win_label) * torch.log(1 - win_prob_clamp)
        )
        
        # 只在可观察样本上计算 (加权平均)
        observable_sum = observable.sum() + eps
        weighted_loss = (loss_per_sample * observable).sum() / observable_sum
        win_losses.append(weighted_loss)
    
    # 窗口 loss 平均
    loss_win = sum(win_losses) / len(win_losses)
    
    # ==========================================================================
    # Step 6: CV loss (更新 cv head)
    # ==========================================================================
    # 使用 BCE with logits (数值更稳定)
    loss_cv = F.binary_cross_entropy_with_logits(cv_logit, label_oracle)
    
    # ==========================================================================
    # Step 7: 总 loss
    # ==========================================================================
    total_loss = cv_loss_weight * loss_cv + win_loss_weight * loss_win
    
    # MetaSpore 要求返回 (loss, task_loss) tuple
    return total_loss, total_loss


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
