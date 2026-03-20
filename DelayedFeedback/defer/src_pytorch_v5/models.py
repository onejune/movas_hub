"""
Defer 模型 - PyTorch v2 版本

模型类型:
- MLP_SIG: 基础二分类模型 (Vanilla, Oracle)
- MLP_winadapt: 自适应窗口模型 (WinAdapt)

适配 249 个类别特征 (parquet 格式数据)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 默认配置
DEFAULT_EMBED_DIM = 8
DEFAULT_HIDDEN_DIMS = [128, 64]


class DeferModel(nn.Module):
    """
    Defer 模型 - 动态 Embedding + MLP
    
    与 TF v2 版本结构对齐:
    - Embedding 层: 每个特征独立 embedding
    - MLP: Dense(128, relu) -> Dense(64, relu) -> Dense(1)
    
    Args:
        vocab_sizes: list, 每个特征的词表大小
        embed_dim: embedding 维度
        hidden_dims: MLP 隐藏层维度
        output_dim: 输出维度 (1 for vanilla/oracle, 4 for winadapt)
    """
    def __init__(self, vocab_sizes, embed_dim=DEFAULT_EMBED_DIM, 
                 hidden_dims=DEFAULT_HIDDEN_DIMS, output_dim=1):
        super().__init__()
        
        self.vocab_sizes = vocab_sizes
        self.embed_dim = embed_dim
        self.num_features = len(vocab_sizes)
        
        # Embedding 层
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for vocab_size in vocab_sizes
        ])
        
        # 计算 MLP 输入维度
        input_dim = self.num_features * embed_dim
        
        # MLP 层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # 统计参数量
        total_embed_params = sum(v * embed_dim for v in vocab_sizes)
        mlp_params = sum(p.numel() for p in self.mlp.parameters())
        output_params = sum(p.numel() for p in self.output_layer.parameters())
        
        print(f"模型构建完成:")
        print(f"  特征数: {self.num_features}")
        print(f"  Embedding 参数: {total_embed_params:,}")
        print(f"  MLP 参数: {mlp_params:,}")
        print(f"  输出层参数: {output_params:,}")
        print(f"  总参数: {total_embed_params + mlp_params + output_params:,}")
    
    def forward(self, cate_feats):
        """
        Args:
            cate_feats: [batch_size, num_features] 类别特征 (整数索引)
        
        Returns:
            logits: [batch_size, output_dim]
        """
        # Embedding
        embeds = []
        for i, emb_layer in enumerate(self.embeddings):
            # 确保索引在有效范围内
            feat_idx = cate_feats[:, i] % self.vocab_sizes[i]
            embeds.append(emb_layer(feat_idx))
        
        # 拼接: [batch_size, num_features * embed_dim]
        x = torch.cat(embeds, dim=-1)
        
        # MLP
        x = self.mlp(x)
        logits = self.output_layer(x)
        
        return logits


class VanillaModel(DeferModel):
    """Vanilla/Oracle 模型 - 单输出"""
    def __init__(self, vocab_sizes, embed_dim=DEFAULT_EMBED_DIM, hidden_dims=DEFAULT_HIDDEN_DIMS):
        super().__init__(vocab_sizes, embed_dim, hidden_dims, output_dim=1)


class WinAdaptModel(DeferModel):
    """
    WinAdapt 模型 - 4 输出头
    
    与原始 TF 版本对齐:
    - logits[:, 0]: cv_logit - 最终转化概率
    - logits[:, 1]: time_24h_logit - 24h 内转化的条件概率 P(t<=24h | cv=1)
    - logits[:, 2]: time_48h_logit - 48h 内转化的条件概率 P(t<=48h | cv=1)
    - logits[:, 3]: time_72h_logit - 72h 内转化的条件概率 P(t<=72h | cv=1)
    
    联合概率: P(窗口内转化) = P(cv) * P(t<=window | cv)
    """
    def __init__(self, vocab_sizes, embed_dim=DEFAULT_EMBED_DIM, hidden_dims=DEFAULT_HIDDEN_DIMS):
        super().__init__(vocab_sizes, embed_dim, hidden_dims, output_dim=4)
    
    def forward(self, cate_feats):
        """
        Returns:
            dict with cv_logits and time window logits
        """
        logits = super().forward(cate_feats)
        return {
            'cv_logits': logits[:, 0],
            'time_24h_logits': logits[:, 1],
            'time_48h_logits': logits[:, 2],
            'time_72h_logits': logits[:, 3],
            'logits': logits,  # 保留原始 logits 供 loss 使用
        }


class DFMModel(DeferModel):
    """DFM 模型 - 2 输出 (cv_logits, log_lamb)"""
    def __init__(self, vocab_sizes, embed_dim=DEFAULT_EMBED_DIM, hidden_dims=DEFAULT_HIDDEN_DIMS):
        super().__init__(vocab_sizes, embed_dim, hidden_dims, output_dim=2)
    
    def forward(self, cate_feats):
        """
        Returns:
            dict with cv_logits and log_lamb
        """
        logits = super().forward(cate_feats)
        return {
            'cv_logits': logits[:, 0],
            'log_lamb': logits[:, 1],
        }


class ESDFMModel(DeferModel):
    """ES-DFM 模型 - 3 输出 (cv_logits, tn_logits, dp_logits)"""
    def __init__(self, vocab_sizes, embed_dim=DEFAULT_EMBED_DIM, hidden_dims=DEFAULT_HIDDEN_DIMS):
        super().__init__(vocab_sizes, embed_dim, hidden_dims, output_dim=3)
    
    def forward(self, cate_feats):
        """
        Returns:
            dict with cv_logits, tn_logits, dp_logits
        """
        logits = super().forward(cate_feats)
        return {
            'cv_logits': logits[:, 0],
            'tn_logits': logits[:, 1],
            'dp_logits': logits[:, 2],
        }


def get_model(method, vocab_sizes, embed_dim=DEFAULT_EMBED_DIM, hidden_dims=DEFAULT_HIDDEN_DIMS):
    """
    获取模型实例
    
    Args:
        method: 'vanilla', 'oracle', 'fnw', 'fnc', 'dfm', 'esdfm', 'winadapt'
        vocab_sizes: list, 每个特征的词表大小
        embed_dim: embedding 维度
        hidden_dims: MLP 隐藏层维度
    """
    method = method.lower()
    if method in ['vanilla', 'oracle', 'fnw', 'fnc']:
        return VanillaModel(vocab_sizes, embed_dim, hidden_dims)
    elif method == 'dfm':
        return DFMModel(vocab_sizes, embed_dim, hidden_dims)
    elif method == 'esdfm':
        return ESDFMModel(vocab_sizes, embed_dim, hidden_dims)
    elif method == 'winadapt':
        return WinAdaptModel(vocab_sizes, embed_dim, hidden_dims)
    else:
        raise ValueError(f"Unknown method: {method}. Available: vanilla, oracle, fnw, fnc, dfm, esdfm, winadapt")


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 模拟 249 个特征的 vocab_sizes
    vocab_sizes = [100] * 249
    
    batch_size = 32
    cate_feats = torch.randint(0, 100, (batch_size, 249))
    
    # 测试 Vanilla 模型
    print("\n=== Vanilla Model ===")
    model = get_model('vanilla', vocab_sizes)
    output = model(cate_feats)
    print(f"Output shape: {output.shape}")
    
    # 测试 WinAdapt 模型
    print("\n=== WinAdapt Model ===")
    model = get_model('winadapt', vocab_sizes)
    output = model(cate_feats)
    for k, v in output.items():
        print(f"  {k}: {v.shape}")
