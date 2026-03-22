#
# Copyright 2022 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import json

def nansum(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x).sum()

def log_loss(yhat, y, eps=1e-12):
    # Clamp yhat to [eps, 1 - eps] for numerical stability
    yhat = torch.clamp(yhat, eps, 1 - eps)
    loss = -(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat))
    return nansum(loss)

class LossUtils:
    @staticmethod
    # minibach: pandas.core.frame.DataFrame
    def log_loss1(yhat, y, minibatch, task_to_id_map = None, task_weights_dict = None, eps=1e-12):
        if isinstance(yhat, dict) and 'prediction' in yhat:
            yhat = yhat['prediction']

        loss = log_loss(yhat, y, eps)
        return loss, loss
    
    @staticmethod
    def log_loss(predictions, labels, minibatch, **kwargs):
        labels = labels.float().view(-1)
        output = predictions.view(-1)
        if torch.isnan(output).any():
            print("WARNING: NaN detected in model predictions during loss calculation.")
            output = torch.nan_to_num(output, nan=0.5, posinf=1.0 - 1e-7, neginf=1e-7)
        output = torch.clamp(output, min=1e-7, max=1 - 1e-7)
        total_loss = F.binary_cross_entropy(output, labels, reduction='mean')

        return total_loss, total_loss
    
    @staticmethod
    def nansum(x):
        return nansum(x)

    @staticmethod
    def ziln_loss(yhat, y_true, minibatch, reduction='sum'):
        """
        ZILN模型的负对数似然损失函数
        Args:
            logit_pi: 零膨胀部分的logit，形状 [batch_size]
            mu: 对数正态部分的均值，形状 [batch_size]
            log_sigma: 对数正态部分的对数标准差，形状 [batch_size]
            y_true: 真实LTV值，形状 [batch_size]
            reduction: 损失聚合方式
        Returns:
            loss: 标量损失值
        """
        logit_pi, mu, log_sigma = torch.split(yhat, [1, 1, 1], dim=1)

        pi = torch.sigmoid(logit_pi)
        sigma = torch.exp(log_sigma)
        
        # 识别零值和正值样本
        zero_mask = (y_true == 0)
        pos_mask = ~zero_mask
        
        loss = torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
        eps = 1e-8
        
        # 零值部分的损失: log(pi + (1-pi)*P(y=0|lognormal)) = log(pi) (因为lognormal在y=0处概率为0)
        if zero_mask.any():
            loss_zero = -torch.log(pi[zero_mask] + eps)
            if reduction == 'mean':
                loss = loss + loss_zero.mean()
            elif reduction == 'sum':
                loss = loss + loss_zero.sum()
            else:
                loss = loss + loss_zero
        
        # 正值部分的损失
        if pos_mask.any():
            y_pos = y_true[pos_mask]
            mu_pos = mu[pos_mask]
            sigma_pos = sigma[pos_mask]
            log_y_pos = torch.log(y_pos)
            
            # 对数正态PDF的对数
            log_normal_pdf = -0.5 * ((log_y_pos - mu_pos) / sigma_pos)**2 - torch.log(sigma_pos) - 0.5 * np.log(2 * np.pi)
            
            # 完整的正值部分似然: (1-pi) * lognormal_pdf
            loss_pos = -torch.log((1 - pi[pos_mask]) + eps) - log_normal_pdf + log_y_pos
            if reduction == 'mean':
                loss = loss + loss_pos.mean()
            elif reduction == 'sum':
                loss = loss + loss_pos.sum()
        return nansum(loss)

    @staticmethod
    def censored_regression_loss(yhat, y, minibatch):
        #print("Executor PyTorch version:", torch.__version__)
        #print('yhat:', yhat.shape, yhat.dim(), yhat)
        if yhat.shape[1] != 2:
            raise ValueError("yhat must have shape (batch_size, 2)")

        mu = yhat[:, 0]
        log_sigma = yhat[:, 1]
        sigma = torch.exp(log_sigma).clamp(min=1e-6, max=10.0)  # 防止 sigma 过小/过大

        # 获取并裁剪 bid_price
        bid_price_series = minibatch['bidPrice']
        bid_price_numeric = pd.to_numeric(bid_price_series, errors='coerce')
        bid_price = torch.tensor(bid_price_numeric.fillna(0.0).values, dtype=torch.float32)

        labels = y.squeeze()
        log_bid = torch.log(bid_price + 1e-8)

        # 创建标准正态分布用于计算 log_cdf
        std_normal = torch.distributions.Normal(0, 1)

        # 竞胜样本损失：负对数概率密度
        if labels.sum() > 0:
            log_win_price = torch.log(bid_price[labels == 1] + 1e-8)
            mu_win = mu[labels == 1]
            sigma_win = sigma[labels == 1]
            dist_win = torch.distributions.Normal(mu_win, sigma_win)
            loss_win = -dist_win.log_prob(log_win_price)
        else:
            loss_win = torch.tensor(0.0)

        # 竞败样本损失：-log_survival = -log(1 - CDF) = -log_cdf(-z)
        if (labels == 0).sum() > 0:
            z_bid = (log_bid - mu) / sigma
            cdf_bid = 0.5 * (1 + torch.erf(z_bid / np.sqrt(2)))
            log_survival = torch.log(torch.clamp(1 - cdf_bid, min=1e-8))
            loss_lose = -log_survival
        else:
            loss_lose = torch.tensor(0.0)

        # 合并损失
        total_loss = torch.cat([
            loss_win.flatten(),
            loss_lose.flatten()
        ])

        # 清理异常值并返回 mean
        total_loss = torch.where(
            torch.isnan(total_loss) | torch.isinf(total_loss),
            torch.zeros_like(total_loss),
            total_loss
        )
        total_loss = nansum(total_loss)
        #test
        #total_loss = torch.tensor(1.0, requires_grad=True)
        #print("total_loss:", total_loss)
        return total_loss

    @staticmethod
    def wce_loss(y_hat, y, minibatch):
        """
        WCE loss for revenue/LTV prediction.
        y_hat: logits (raw model output), shape (N,)
        y:     true revenue (>=0), shape (N,)
        Returns mean loss to minimize.
        """
        p = torch.sigmoid(y_hat)
        eps = 1e-8
        p = torch.clamp(p, eps, 1 - eps)
        loss = - (y * torch.log(p) + torch.log(1 - p))
        return nansum(loss)

    @staticmethod
    def mse_loss(y_hat, y, minibatch):
        loss = (y - y_hat) ** 2
        return nansum(loss)

    @staticmethod
    def focal_loss(y_hat, y, minibatch, alpha=1.0, gamma=2.0):
        p = y_hat
        eps = 1e-8
        p = torch.clamp(p, eps, 1 - eps)  # Clamp to prevent log(0)
        
        # Calculate cross entropy
        ce_loss = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))
        
        # Calculate focal weight
        # For positive samples (y=1): (1-p)^gamma
        # For negative samples (y=0): p^gamma
        p_t = p * y + (1 - p) * (1 - y)  # Probability of the true class
        focal_weight = (1 - p_t) ** gamma
        
        # Apply focal weight and alpha
        loss = alpha * focal_weight * ce_loss
        return nansum(loss)

    @staticmethod
    def multi_label_loss_func(y_pred, y_true, minibatch, task_to_id_map = None, task_weights_dict = None):
        if task_to_id_map is None:
            raise ValueError("task_to_id_map must be provided.")
    
        y_true_list = minibatch['mul_labels']
        batch_size = y_pred.size(0)
        num_tasks = len(task_to_id_map)
        device = y_pred.device

        # === 1. 构建目标张量 [batch_size, num_tasks] ===
        target_tensor = torch.full((batch_size, num_tasks), -1.0, dtype=torch.float32, device=device)
        for batch_idx, label_map in enumerate(y_true_list):
            if isinstance(label_map, str):
                try:
                    label_dict = json.loads(label_map)
                except (json.JSONDecodeError, TypeError):
                    continue
            elif isinstance(label_map, dict):
                label_dict = label_map
            else:
                continue

            if isinstance(label_dict, dict):
                for task_name, label_val in label_dict.items():
                    if task_name in task_to_id_map:
                        task_id = task_to_id_map[task_name]
                        target_tensor[batch_idx, task_id] = float(label_val)

        # === 2. 计算 BCE loss and mask ===
        bce_loss = F.binary_cross_entropy(y_pred, target_tensor, reduction='none')  # [B, T]
        mask = (target_tensor != -1.0)  # [B, T]

        if mask.sum() == 0:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            per_task_avg_bce = torch.zeros(num_tasks, device=device)
            return total_loss, per_task_avg_bce

        # === 3. 准备任务权重（默认为 1.0）===
        if task_weights_dict is None:
            task_weights_dict = {}
        # 构建 task_id -> weight 映射
        task_id_to_weight = {}
        for task_name, task_id in task_to_id_map.items():
            weight = task_weights_dict.get(task_name, 1.0)
            task_id_to_weight[task_id] = float(weight)

        # === 4. 计算 total_loss 和 per_task_avg_bce ===
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        per_task_avg_bce = []

        for task_id in range(num_tasks):
            task_mask = mask[:, task_id]
            num_valid = task_mask.sum()

            if num_valid > 0:
                avg_bce = bce_loss[:, task_id][task_mask].mean()
                per_task_avg_bce.append(avg_bce.detach())  # for logging, no grad needed

                weight = task_id_to_weight.get(task_id, 1.0)
                weighted_task_loss = weight * avg_bce
                total_loss = total_loss + weighted_task_loss
            else:
                per_task_avg_bce.append(torch.tensor(0.0, device=device))
                # 无有效样本，该 task 不贡献 loss

        per_task_avg_bce = torch.stack(per_task_avg_bce)  # [num_tasks]
        return total_loss, per_task_avg_bce

    @staticmethod
    def multi_label_uncertainty_loss_func(y_pred, y_true, minibatch, task_to_id_map=None, model=None):
        """
        支持不确定性权重的多标签损失函数
        Args:
            y_pred: 模型预测结果 [batch_size, num_tasks]
            y_true: 真实标签（未使用，从mini_batch中获取）
            minibatch: 批次数据
            task_to_id_map: 任务名到ID的映射
            model: 模型对象，用于获取不确定性权重参数
        """
        if task_to_id_map is None:
            raise ValueError("task_to_id_map must be provided.")
        
        y_true_list = minibatch['mul_labels']  # list of dict or str
        batch_size = y_pred.size(0)
        num_tasks = len(task_to_id_map)
        device = y_pred.device

        # === 1. 构建目标张量 [batch_size, num_tasks] ===
        target_tensor = torch.full((batch_size, num_tasks), -1.0, dtype=torch.float32, device=device)
        for batch_idx, label_map in enumerate(y_true_list):
            if isinstance(label_map, str):
                try:
                    label_dict = json.loads(label_map)
                except (json.JSONDecodeError, TypeError):
                    continue
            elif isinstance(label_map, dict):
                label_dict = label_map
            else:
                continue

            if isinstance(label_dict, dict):
                for task_name, label_val in label_dict.items():
                    if task_name in task_to_id_map:
                        task_id = task_to_id_map[task_name]
                        target_tensor[batch_idx, task_id] = float(label_val)

        # === 2. 计算 BCE loss (no reduction) ===
        bce_loss = F.binary_cross_entropy(y_pred, target_tensor, reduction='none')  # [B, T]
        mask = (target_tensor != -1.0)  # [B, T], bool

        if mask.sum() == 0:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            per_task_avg_bce = torch.zeros(num_tasks, device=device)
            return total_loss, per_task_avg_bce

        # === 3. 初始化返回值 ===
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        per_task_avg_bce = []

        # === 4. 遍历每个 task ===
        use_uncertainty = (
            model is not None
            and hasattr(model, 'use_uncertainty_weighting')
            and model.use_uncertainty_weighting
            and hasattr(model, 'log_vars')
        )

        for task_id in range(num_tasks):
            task_mask = mask[:, task_id]          # [B]
            num_valid = task_mask.sum()

            if num_valid > 0:
                # 原始 BCE 平均（用于监控）
                avg_bce = bce_loss[:, task_id][task_mask].mean()
                per_task_avg_bce.append(avg_bce.detach())  # detach to avoid grad graph for logging

                if use_uncertainty and task_id < len(model.log_vars):
                    log_var = model.log_vars[task_id]      # scalar
                    precision = torch.exp(-log_var)        # exp(-s_i)
                    task_loss = precision * avg_bce + log_var
                else:
                    task_loss = avg_bce

                total_loss = total_loss + task_loss
            else:
                # 无有效样本：BCE 为 0，loss 贡献为 0
                per_task_avg_bce.append(torch.tensor(0.0, device=device))
                # total_loss += 0 → do nothing

        per_task_avg_bce = torch.stack(per_task_avg_bce)  # [num_tasks]loss
        return total_loss, per_task_avg_bce

    @staticmethod
    def delf_loss(y_pred, y_true, minibatch, task_to_id_map=None, model=None):
        """
        DELF损失函数（支持 per-sample observation_time）
        联合建模转化与否 + 转化时间分布 + IPW纠偏
        Args:
            model: DELFWideDeep 模型实例
            y_pred: 模型输出 (propensity_prob, k_param, lambda_param, wide_contribution)
            y: 真实标签（占位符，未使用）
            minibatch: 批量数据字典，必须包含：
                - 'observed_conversion': [B] 观测到的转化标签 (0/1)
                - 'observation_time': [B] 每个样本的观察窗口（单位：天，Tensor）
        """
        lambda_time = 1.0
        # === 安全提取并转为 torch.Tensor ===
        def to_tensor(series_or_tensor, dtype=torch.float32):
            if isinstance(series_or_tensor, pd.Series):
                return torch.tensor(series_or_tensor.values, dtype=dtype, device=y_pred[0].device)
            elif isinstance(series_or_tensor, torch.Tensor):
                return series_or_tensor.to(dtype).to(y_pred[0].device)
            else:
                raise TypeError(f"Unsupported type: {type(series_or_tensor)}")

        observed_conversion = to_tensor(minibatch['observed_conversion'], dtype=torch.int32)  # [B]
        obs_time = to_tensor(minibatch['observation_time'], dtype=torch.float32)             # [B]
        p_prop, k, lam, _ = y_pred  # 各为 [B, 1] 或 [B]
        p_prop = torch.clamp(p_prop, 1e-7, 1.0 - 1e-7)

        # 确保维度对齐：squeeze 到 [B]
        if p_prop.dim() > 1:
            p_prop = p_prop.squeeze(-1)  # [B]
        if k.dim() > 1:
            k = k.squeeze(-1)
        if lam.dim() > 1:
            lam = lam.squeeze(-1)
        if obs_time.dim() > 1:
            obs_time = obs_time.squeeze(-1)
        obs_time = torch.clamp(obs_time, min=1e-6)

        # 计算观察时间内的 Weibull CDF（用于 IPW 和理解，但 loss 不直接用）
        cdf_obs = model.weibull_cdf(k, lam, obs_time)  # [B]

        # 计算 IPW 权重（注意：IPW 内部也需使用 per-sample obs_window）
        ipw_weights = model.calculate_ipw_weights(p_prop, k, lam, obs_window=obs_time)  # [B]

        # 构造样本权重，只用来加权损失，不需要梯度回传
        sample_weights = torch.where(
            observed_conversion == 1,
            torch.ones_like(ipw_weights),  # 已观测转化：权重为1
            ipw_weights                   # 未观测转化：使用权重
        ).detach()  # [B]

        #print('p_prop:', p_prop, 'observed_conversion:', observed_conversion, 'sample_weights:', sample_weights, 'obs_time:', obs_time, 'k:', k, 'lam:', lam)
        # 转化倾向损失（加权 BCE）
        prop_loss = torch.nn.functional.binary_cross_entropy(
            p_prop,
            observed_conversion.float(),
            weight=sample_weights
        )

        # 时间分布损失（仅对 observed_conversion == 1 的样本计算）
        time_mask = (observed_conversion == 1).float()  # [B]
        if time_mask.sum() > 0:
            # Weibull 负对数似然（per-sample）
            # 注意：这里使用每个样本自己的 obs_time（即真实转化时间）
            # 前提：obs_time 对于 observed_conversion=1 的样本 应该是真实转化时间（而非截断窗口）
            loglik = torch.log(k / lam + 1e-8) + (k - 1) * torch.log(obs_time / lam + 1e-8) - (obs_time / lam) ** k
            time_loss = -torch.mean(loglik * time_mask)
        else:
            time_loss = torch.tensor(0.0, device=k.device)

        total_loss = prop_loss + lambda_time * time_loss
        return total_loss, prop_loss

# DEFER 损失函数
from metaspore.algos.delay_feedback.defer_loss import (
    delay_win_select_loss_v1,
    delay_win_select_loss_v2,
)

# 定义损失函数映射字典
LOSS_FUNCTIONS = {
    'log_loss': LossUtils.log_loss,
    'mse_loss': LossUtils.mse_loss,
    'wce_loss': LossUtils.wce_loss,                        
    'ce_loss': F.cross_entropy,
    'focal_loss': LossUtils.focal_loss,
    'l1_loss': F.l1_loss,
    'huber_loss': F.smooth_l1_loss,
    'ziln_loss' : LossUtils.ziln_loss,
    'mtl_loss': LossUtils.multi_label_loss_func,
    'uw_mtl_loss': LossUtils.multi_label_uncertainty_loss_func,
    'delf_loss': LossUtils.delf_loss,
    # DEFER 损失函数
    'defer_loss_v1': delay_win_select_loss_v1,     # v1: 14 维标签
    'defer_loss_v2': delay_win_select_loss_v2,     # v2: 8 维标签
}

def get_loss_function(loss_name: str):
    fn = LOSS_FUNCTIONS.get(loss_name)
    if fn is None:
        raise ValueError(f"Unknown loss function: {loss_name}")
    return fn
    


# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    N = 2560
    yhat = torch.randn(N, 2, requires_grad=True)  # 确保初始张量需要梯度
    y = torch.randint(0, 2, (N, 1), dtype=torch.float32)
    
    # 创建示例 minibatch DataFrame
    df = pd.DataFrame({
        'bidPrice': np.random.uniform(1, 100, N),
    })
    
    # 计算损失
    loss = LossUtils.censored_regression_loss(yhat, y, df)
    print(f"Loss value: {loss.item()} \
            \nLoss shape: {loss.shape}  \
            \nLoss requires_grad: {loss.requires_grad}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    
    # 测试反向传播
    try:
        loss.backward()
        print("Backward pass successful!")
        print(f"yhat.grad shape: {yhat.grad.shape}")
    except Exception as e:
        print(f"Backward pass failed: {e}")

