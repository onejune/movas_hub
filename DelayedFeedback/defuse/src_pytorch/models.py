#!/usr/bin/env python3
"""
DEFUSE PyTorch Models
"""

import torch
import torch.nn as nn


class DEFUSEModel(nn.Module):
    """
    DEFUSE Model with shared backbone + multiple heads
    - cv_logits: conversion prediction
    - tn_logits: true negative prediction  
    - dp_logits: delayed positive prediction
    """
    def __init__(self, vocab_sizes, embed_dim=8, hidden_dims=[256, 256, 128]):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(size, embed_dim, padding_idx=0)
            for col, size in vocab_sizes.items()
        })
        
        input_dim = len(vocab_sizes) * embed_dim
        
        # Shared backbone
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.01),
            ])
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        
        # Output heads
        self.cv_head = nn.Linear(hidden_dims[-1], 1)  # CVR prediction
        self.tn_head = nn.Linear(hidden_dims[-1], 1)  # True Negative
        self.dp_head = nn.Linear(hidden_dims[-1], 1)  # Delayed Positive
        
    def forward(self, x_dict):
        # Embedding lookup
        embeds = []
        for col, emb in self.embeddings.items():
            embeds.append(emb(x_dict[col]))
        x = torch.cat(embeds, dim=-1)
        
        # Backbone
        h = self.backbone(x)
        
        # Output heads
        cv_logits = self.cv_head(h)
        tn_logits = self.tn_head(h)
        dp_logits = self.dp_head(h)
        
        return {
            'cv_logits': cv_logits,
            'tn_logits': tn_logits,
            'dp_logits': dp_logits,
        }


class BiDEFUSEModel(nn.Module):
    """
    Bi-DEFUSE Model with in-window and out-window heads
    - logits_inw: in-window conversion prediction
    - logits_outw: out-window conversion prediction
    - tn_logits, dp_logits: auxiliary heads
    """
    def __init__(self, vocab_sizes, embed_dim=8, hidden_dims=[256, 256, 128]):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(size, embed_dim, padding_idx=0)
            for col, size in vocab_sizes.items()
        })
        
        input_dim = len(vocab_sizes) * embed_dim
        
        # Shared backbone
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.01),
            ])
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        
        # Output heads
        self.inw_head = nn.Linear(hidden_dims[-1], 1)   # In-window CVR
        self.outw_head = nn.Linear(hidden_dims[-1], 1)  # Out-window CVR
        self.tn_head = nn.Linear(hidden_dims[-1], 1)    # True Negative
        self.dp_head = nn.Linear(hidden_dims[-1], 1)    # Delayed Positive
        
    def forward(self, x_dict):
        # Embedding lookup
        embeds = []
        for col, emb in self.embeddings.items():
            embeds.append(emb(x_dict[col]))
        x = torch.cat(embeds, dim=-1)
        
        # Backbone
        h = self.backbone(x)
        
        # Output heads
        return {
            'logits_inw': self.inw_head(h),
            'logits_outw': self.outw_head(h),
            'tn_logits': self.tn_head(h),
            'dp_logits': self.dp_head(h),
        }
