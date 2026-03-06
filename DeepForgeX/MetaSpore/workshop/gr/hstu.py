import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class PointwiseAggregatedAttention(nn.Module):
    """
    HSTU架构中的点积聚合注意力（PAA）
    不使用softmax，保留原始点积权重以建模绝对偏好强度
    """
    def __init__(self, d_model: int, n_heads: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # 线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # 相对位置偏置（简化版）
        self.relative_bias = nn.Parameter(torch.zeros(2 * max_len - 1, n_heads))
        
        # 缩放因子（可选，用于数值稳定）
        self.scale = nn.Parameter(torch.ones(1) * math.sqrt(self.d_head))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model) 输入序列
            mask: (batch_size, seq_len) 掩码，用于屏蔽填充位置
        Returns:
            output: (batch_size, seq_len, d_model) 输出序列
        """
        batch_size, seq_len, _ = x.shape
        
        # 生成Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, L, D)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # 计算点积得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L, L)
        
        # 添加相对位置偏置
        seq_range = torch.arange(seq_len, device=x.device)
        relative_positions = seq_range[None, :] - seq_range[:, None] + seq_len - 1  # (L, L)
        relative_bias = self.relative_bias[relative_positions]  # (L, L, H)
        relative_bias = relative_bias.permute(2, 0, 1).unsqueeze(0)  # (1, H, L, L)
        scores = scores + relative_bias

        # 应用掩码（可选）
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            mask = mask * mask.transpose(-1, -2)  # (B, 1, L, L) - 双向掩码
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 核心：不使用softmax！直接聚合
        # 使用scores作为权重对V进行加权求和
        output = torch.matmul(scores, V)  # (B, H, L, D)

        # 重塑输出
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        output = self.dropout(output)
        
        return output


class TargetAwareCrossAttention(nn.Module):
    """
    目标感知交叉注意力，用于排序任务
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.scale = nn.Parameter(torch.ones(1) * math.sqrt(self.d_head))
        self.dropout = nn.Dropout(dropout)

    def forward(self, target: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            target: (batch_size, num_targets, d_model) 候选物品表示
            context: (batch_size, seq_len, d_model) 用户历史序列
            mask: (batch_size, seq_len) 上下文掩码
        Returns:
            output: (batch_size, num_targets, d_model) 聚合后的表示
        """
        batch_size, num_targets, _ = target.shape
        seq_len = context.size(1)
        
        Q = self.W_q(target).view(batch_size, num_targets, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, D)
        K = self.W_k(context).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)    # (B, H, L, D)
        V = self.W_v(context).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)   # (B, H, L, D)

        # 计算点积得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, L)
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 点积聚合
        output = torch.matmul(scores, V)  # (B, H, T, D)

        # 重塑输出
        output = output.transpose(1, 2).contiguous().view(batch_size, num_targets, self.d_model)
        output = self.W_o(output)
        output = self.dropout(output)
        
        return output


class HSTUBlock(nn.Module):
    """
    HSTU架构中的单个块，包含PAA、残差连接和前馈网络
    """
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = PointwiseAggregatedAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),  # 使用GELU激活函数
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 残差连接 + 层归一化
        attn_out = self.attn(x, mask)
        x = self.norm1(x + attn_out)
        
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class HSTU(nn.Module):
    """
    完整的HSTU架构模型
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        ff_dim: int = 2048,
        n_layers: int = 6,
        max_len: int = 1024,
        dropout: float = 0.1,
        task: str = 'generation'  # 'generation', 'retrieval', 'ranking'
    ):
        super().__init__()
        self.task = task
        self.d_model = d_model
        
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        
        # HSTU块堆叠
        self.layers = nn.ModuleList([
            HSTUBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)
        ])
        
        # 目标感知交叉注意力（用于排序任务）
        if task == 'ranking':
            self.target_attn = TargetAwareCrossAttention(d_model, n_heads, dropout)
            self.rank_output = nn.Linear(d_model, 1)  # 输出单个得分
        
        # 最终输出层
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) 输入序列
            targets: (batch_size, num_targets, d_model) 候选物品表示（仅在ranking任务中使用）
            mask: (batch_size, seq_len) 掩码
        Returns:
            output: 根据任务类型返回不同输出
        """
        batch_size, seq_len = x.shape
        
        # 嵌入和位置编码
        x = self.token_embedding(x) * math.sqrt(self.d_model)  # 缩放嵌入
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # 通过HSTU层
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        if self.task == 'ranking' and targets is not None:
            # 使用目标感知交叉注意力进行排序
            target_scores = self.target_attn(targets, x, mask)
            scores = self.rank_output(target_scores).squeeze(-1)  # (batch_size, num_targets)
            return scores
        elif self.task == 'retrieval':
            # 为每个位置预测下一个token
            return self.fc_out(x)
        else:  # generation
            # 生成任务：预测下一个token
            return self.fc_out(x)


class HSTUWithStochasticLength(nn.Module):
    """
    带有随机长度（Stochastic Length）的HSTU变体
    用于提高训练效率
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        ff_dim: int = 2048,
        n_layers: int = 6,
        max_len: int = 1024,
        dropout: float = 0.1,
        sl_prob: float = 0.1  # 随机截断概率
    ):
        super().__init__()
        self.hstu = HSTU(vocab_size, d_model, n_heads, ff_dim, n_layers, max_len, dropout)
        self.sl_prob = sl_prob  # Stochastic Length概率

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 随机截断序列以提高效率（模拟SL机制）
        if self.training and self.sl_prob > 0:
            batch_size, seq_len = x.shape
            # 随机决定截断长度
            max_trunc_len = max(1, int(seq_len * (1 - self.sl_prob)))
            trunc_len = torch.randint(1, max_trunc_len + 1, (1,)).item()
            
            # 随机选择截断位置
            start_idx = torch.randint(0, seq_len - trunc_len + 1, (1,)).item()
            x = x[:, start_idx:start_idx + trunc_len]
            if mask is not None:
                mask = mask[:, start_idx:start_idx + trunc_len]
        
        return self.hstu(x, targets, mask)


# 示例：构建和测试HSTU模型
def test_hstu():
    # 生成式任务
    print("Testing Generation HSTU...")
    gen_model = HSTU(vocab_size=10000, d_model=256, n_heads=4, ff_dim=512, n_layers=2, task='generation')
    
    batch_size, seq_len = 256, 100
    x = torch.randint(0, 9999, (batch_size, seq_len))
    mask = torch.ones_like(x)
    
    output = gen_model(x, mask=mask)
    print(f"Generation - Input: {x.shape}, Output: {output.shape}")
    
    # 排序任务
    print("\nTesting Ranking HSTU...")
    rank_model = HSTU(vocab_size=10000, d_model=256, n_heads=4, ff_dim=512, n_layers=2, task='ranking')
    
    x = torch.randint(0, 9999, (batch_size, seq_len))
    num_targets = 10
    targets = torch.randn(batch_size, num_targets, 256)  # 候选物品表示
    
    scores = rank_model(x, targets=targets, mask=mask)
    print(f"Ranking - Input: {x.shape}, Targets: {targets.shape}, Scores: {scores.shape}")
    
    # 检查输出维度
    assert scores.shape == (batch_size, num_targets), f"Expected {(batch_size, num_targets)}, got {scores.shape}"
    print("All tests passed!")


if __name__ == "__main__":
    test_hstu()



