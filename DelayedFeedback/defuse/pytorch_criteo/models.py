#!/usr/bin/env python3
"""
DEFUSE Criteo Benchmark - Models

Implements all model variants for delayed feedback methods.
Aligned with TensorFlow implementation in src_tf/models.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from config import DataConfig


# Feature bin sizes (from TF implementation)
NUM_BIN_SIZES = (64, 16, 128, 64, 128, 64, 512, 512)
CAT_BIN_SIZES = (512, 128, 256, 256, 64, 256, 256, 16, 256)


class FeatureEncoder(nn.Module):
    """
    Feature encoder for Criteo dataset.
    
    - Numeric features: bucketized and embedded
    - Categorical features: hashed and embedded
    """
    
    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Numeric feature embeddings (bucketized)
        self.num_embeddings = nn.ModuleList([
            nn.Embedding(bin_size, embed_dim)
            for bin_size in NUM_BIN_SIZES
        ])
        
        # Categorical feature embeddings (hashed)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(bin_size, embed_dim)
            for bin_size in CAT_BIN_SIZES
        ])
        
        # Total feature dimension
        self.output_dim = (len(NUM_BIN_SIZES) + len(CAT_BIN_SIZES)) * embed_dim
    
    def forward(self, num_features: torch.Tensor, cat_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            num_features: (batch, 8) normalized numeric features in [0, 1]
            cat_features: (batch, 9) hashed categorical feature indices
        
        Returns:
            (batch, output_dim) embedded features
        """
        embeddings = []
        
        # Numeric features: bucketize and embed
        for i, (embed, bin_size) in enumerate(zip(self.num_embeddings, NUM_BIN_SIZES)):
            # Bucketize: [0, 1] -> [0, bin_size-1]
            bucket_idx = (num_features[:, i] * (bin_size - 1)).long().clamp(0, bin_size - 1)
            embeddings.append(embed(bucket_idx))
        
        # Categorical features: embed with modulo to handle out-of-range
        for i, (embed, bin_size) in enumerate(zip(self.cat_embeddings, CAT_BIN_SIZES)):
            idx = cat_features[:, i] % bin_size
            embeddings.append(embed(idx))
        
        return torch.cat(embeddings, dim=-1)


class MLPBlock(nn.Module):
    """MLP block with optional batch norm."""
    
    def __init__(self, in_dim: int, out_dim: int, use_bn: bool = True, l2_reg: float = 1e-5):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        self.activation = nn.LeakyReLU()
        
        # L2 regularization is handled by weight_decay in optimizer
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class BaseMLP(nn.Module):
    """Base MLP model."""
    
    def __init__(self, 
                 hidden_dims: List[int] = [256, 256, 128],
                 embed_dim: int = 8,
                 use_bn: bool = True,
                 l2_reg: float = 1e-5):
        super().__init__()
        
        self.encoder = FeatureEncoder(embed_dim)
        
        # MLP layers
        layers = []
        in_dim = self.encoder.output_dim
        for out_dim in hidden_dims:
            layers.append(MLPBlock(in_dim, out_dim, use_bn, l2_reg))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        
        self.hidden_dim = hidden_dims[-1]
    
    def encode(self, num_features: torch.Tensor, cat_features: torch.Tensor) -> torch.Tensor:
        """Encode features through MLP."""
        x = self.encoder(num_features, cat_features)
        x = self.mlp(x)
        return x


class MLPSig(BaseMLP):
    """
    MLP with single sigmoid output.
    Used for: Vanilla, FNW, FNC, ES-DFM finetune, DEFUSE finetune, DEFER finetune
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, num_features: torch.Tensor, cat_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.encode(num_features, cat_features)
        logits = self.output(x)
        return {"logits": logits}


class MLPExpDelay(BaseMLP):
    """
    MLP with exponential delay modeling.
    Used for: DFM (Delayed Feedback Model)
    
    Outputs:
    - logits: conversion probability
    - log_lamb: log of delay rate parameter
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output = nn.Linear(self.hidden_dim, 2)
    
    def forward(self, num_features: torch.Tensor, cat_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.encode(num_features, cat_features)
        out = self.output(x)
        return {
            "logits": out[:, 0:1],
            "log_lamb": out[:, 1:2]
        }


class MLPTnDp(BaseMLP):
    """
    MLP with tn/dp dual outputs.
    Used for: ES-DFM pretrain, DEFUSE pretrain
    
    Outputs:
    - tn_logits: true negative probability
    - dp_logits: delayed positive probability
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output = nn.Linear(self.hidden_dim, 2)
    
    def forward(self, num_features: torch.Tensor, cat_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.encode(num_features, cat_features)
        out = self.output(x)
        return {
            "tn_logits": out[:, 0:1],
            "dp_logits": out[:, 1:2]
        }


class MLPDp(BaseMLP):
    """
    MLP with single dp output.
    Used for: DEFER pretrain
    
    Outputs:
    - logits: delayed positive probability
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, num_features: torch.Tensor, cat_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.encode(num_features, cat_features)
        logits = self.output(x)
        return {"logits": logits}


class BiDefuseModel(BaseMLP):
    """
    Bi-DEFUSE model with in-window and out-window heads.
    Uses attention mechanism to combine expert outputs.
    
    Aligned with TF Bi-DEFUSE_inoutw implementation.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Separate branches for in-window and out-window
        self.inw_branch = MLPBlock(self.hidden_dim, 128, use_bn=True)
        self.outw_branch = MLPBlock(self.hidden_dim, 128, use_bn=True)
        
        # Attention weights (learnable)
        self.attention_inw = nn.Parameter(torch.ones(1, 2, 1))
        self.attention_outw = nn.Parameter(torch.ones(1, 2, 1))
        
        # Output heads
        self.inw_output = nn.Linear(128, 1)
        self.outw_output = nn.Linear(128, 1)
    
    def forward(self, num_features: torch.Tensor, cat_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shared encoding
        x = self.encode(num_features, cat_features)
        
        # Branch outputs
        x_inw = self.inw_branch(x)   # (batch, 128)
        x_outw = self.outw_branch(x)  # (batch, 128)
        
        # Attention mechanism
        # Combine branch output with shared representation
        # expert_tensor shape: (batch, 2, 128)
        expert_inw = torch.stack([x_inw, x[:, :128] if x.size(1) >= 128 else F.pad(x, (0, 128 - x.size(1)))], dim=1)
        expert_outw = torch.stack([x_outw, x[:, :128] if x.size(1) >= 128 else F.pad(x, (0, 128 - x.size(1)))], dim=1)
        
        # Weighted sum with attention
        att_inw = (expert_inw * self.attention_inw).sum(dim=1)   # (batch, 128)
        att_outw = (expert_outw * self.attention_outw).sum(dim=1)  # (batch, 128)
        
        # Final outputs
        logits_inw = self.inw_output(att_inw)
        logits_outw = self.outw_output(att_outw)
        
        return {
            "logits_inw": logits_inw,
            "logits_outw": logits_outw
        }
    
    def predict(self, num_features: torch.Tensor, cat_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict with combined probability.
        
        TF original: pred = inw_pred + outw_pred (simple sum)
        This allows prob > 1, but matches TF implementation.
        """
        outputs = self.forward(num_features, cat_features)
        inw_prob = torch.sigmoid(outputs["logits_inw"])
        outw_prob = torch.sigmoid(outputs["logits_outw"])
        
        # TF original: simple sum (can exceed 1)
        prob = inw_prob + outw_prob
        
        return {
            "prob": prob,
            "inw_prob": inw_prob,
            "outw_prob": outw_prob,
            **outputs
        }


def get_model(name: str, **kwargs) -> nn.Module:
    """
    Get model by name.
    
    Args:
        name: Model name
            - MLP_SIG: Single output for Vanilla/FNW/FNC/finetune
            - MLP_EXP_DELAY: DFM with exponential delay
            - MLP_tn_dp: ES-DFM/DEFUSE pretrain
            - MLP_dp: DEFER pretrain
            - Bi-DEFUSE: Bi-DEFUSE with attention
    """
    models = {
        "MLP_SIG": MLPSig,
        "MLP_EXP_DELAY": MLPExpDelay,
        "MLP_tn_dp": MLPTnDp,
        "MLP_dp": MLPDp,
        "Bi-DEFUSE": BiDefuseModel,
    }
    
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    
    return models[name](**kwargs)
