"""
Defer 评估指标 - PyTorch 版本
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


def cal_auc(labels, probs):
    """计算 AUC"""
    try:
        # 确保有正负样本
        if len(np.unique(labels)) < 2:
            return 0.5
        return roc_auc_score(labels, probs)
    except:
        return 0.5


def cal_prauc(labels, probs):
    """计算 PR-AUC"""
    try:
        if len(np.unique(labels)) < 2:
            return labels.mean()
        return average_precision_score(labels, probs)
    except:
        return labels.mean() if len(labels) > 0 else 0.0


def cal_llloss(labels, probs):
    """计算 LogLoss (交叉熵)"""
    try:
        # 裁剪概率避免 log(0)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return log_loss(labels, probs)
    except:
        return 1.0


def cal_ece(labels, probs, n_bins=10):
    """
    计算 Expected Calibration Error (ECE)
    
    ECE 衡量预测概率与实际频率的偏差
    """
    try:
        labels = np.array(labels).flatten()
        probs = np.array(probs).flatten()
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            low, high = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (probs >= low) & (probs < high)
            
            if mask.sum() > 0:
                bin_acc = labels[mask].mean()
                bin_conf = probs[mask].mean()
                bin_size = mask.sum() / len(labels)
                ece += bin_size * abs(bin_acc - bin_conf)
        
        return ece
    except:
        return 0.5


def cal_pcoc(labels, probs):
    """
    计算 PCOC (Predicted Clicks Over Observed Clicks)
    
    PCOC = sum(probs) / sum(labels)
    
    衡量预测总量与实际总量的偏差:
    - PCOC = 1.0: 完美校准
    - PCOC > 1.0: 预测偏高 (高估)
    - PCOC < 1.0: 预测偏低 (低估)
    """
    try:
        labels = np.array(labels).flatten()
        probs = np.array(probs).flatten()
        
        observed = labels.sum()
        predicted = probs.sum()
        
        if observed == 0:
            return float('inf') if predicted > 0 else 1.0
        
        return predicted / observed
    except:
        return 1.0


class ScalarMovingAverage:
    """标量移动平均"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.values = []
    
    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values = self.values[-self.window_size:]
    
    def get(self):
        if len(self.values) == 0:
            return 0.0
        return np.mean(self.values)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 测试指标
    np.random.seed(42)
    
    n = 1000
    labels = np.random.randint(0, 2, n)
    probs = np.random.random(n) * 0.5 + labels * 0.3  # 有一定相关性
    probs = np.clip(probs, 0, 1)
    
    print(f"AUC:     {cal_auc(labels, probs):.4f}")
    print(f"PR-AUC:  {cal_prauc(labels, probs):.4f}")
    print(f"LogLoss: {cal_llloss(labels, probs):.4f}")
    print(f"ECE:     {cal_ece(labels, probs):.4f}")
