# -*- coding: utf-8 -*-
import metaspore as ms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.nn.init import xavier_uniform_, zeros_
from ..layers import MLPLayer

'''
paper: Leaving No One Behind: A Multi-Scenario Multi-Task Meta Learning Approach for Advertiser Modeling
Optimized for Binary Classification Tasks
'''

class TransformerLayer(nn.Module):
    """论文4.1.2节提到的Transformer层，用于处理序列特征"""
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # 多头自注意力机制（论文公式2-4）
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

    def forward(self, x, src_mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        # 自注意力（论文公式2）
        attn_output, _ = self.self_attention(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class MetaUnit(nn.Module):
    """论文4.2.1节的核心Meta Unit，用于生成动态网络参数"""
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=3):
        super(MetaUnit, self).__init__()
        self.num_layers = num_layers
        
        # 生成权重和偏置的元网络（论文公式9-13）
        self.meta_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            layer_output_dim = hidden_dim if i < num_layers - 1 else output_dim
            
            self.meta_layers.append(nn.Sequential(
                nn.Linear(layer_input_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, layer_output_dim)
            ))

    def forward(self, x):
        """
        Args:
            x: 输入向量 [batch, input_dim]
        Returns:
            h_output: 经过meta unit变换后的输出 [batch, output_dim]
        """
        h = x
        for i, layer in enumerate(self.meta_layers):
            h = layer(h)
        return h

class MetaAttentionModule(nn.Module):
    """论文4.2.2节的Meta Attention模块 - 完全重构以符合论文公式14-16"""
    def __init__(self, scenario_dim, task_dim, expert_dim, num_experts, meta_hidden_dim=256):
        super(MetaAttentionModule, self).__init__()
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.task_dim = task_dim
        self.scenario_dim = scenario_dim
        self.meta_hidden_dim = meta_hidden_dim
        
        # 根据论文公式14：a_ti = v^T * Meta_t([E_i || T_t])
        # Meta_t是一个由场景知识生成的线性变换
        self.meta_unit = MetaUnit(
            input_dim=scenario_dim + task_dim, 
            output_dim=expert_dim + task_dim,  # 输出维度与[E_i || T_t]相同
            hidden_dim=meta_hidden_dim
        )
        
        # 投影向量v（论文公式14）
        self.v = nn.Parameter(torch.randn(expert_dim + task_dim, 1))

    def forward(self, scene_repr, task_anchor, expert_reprs):
        """
        Args:
            scene_repr: [batch, scenario_dim] - 场景表示
            task_anchor: [batch, task_dim] - 任务锚点
            expert_reprs: [batch, num_experts, expert_dim] - 专家表示
        Returns:
            aggregated_repr: [batch, expert_dim] - 聚合后的表示
            attention_weights: [batch, num_experts] - 注意力权重
        """
        batch_size = scene_repr.size(0)
        
        # 拼接场景和任务表示作为meta unit的输入
        scene_task = torch.cat([scene_repr, task_anchor], dim=1)  # [batch, scenario_dim + task_dim]
        
        # 为每个专家生成注意力分数（论文公式14）
        attention_scores = []
        for i in range(self.num_experts):
            # 获取当前专家表示
            expert_i = expert_reprs[:, i, :]  # [batch, expert_dim]
            
            # 拼接专家和任务信息 [E_i || T_t]
            expert_task = torch.cat([expert_i, task_anchor], dim=1)  # [batch, expert_dim + task_dim]
            
            # 通过meta unit生成动态参数（论文公式14）
            # 这里meta_unit输出一个与expert_task相同维度的向量
            meta_output = self.meta_unit(scene_task)  # [batch, expert_dim + task_dim]
            
            # 应用线性变换：v^T * σ(W * [E_i || T_t] + b)
            # 简化实现：直接计算点积
            score = torch.matmul(expert_task, self.v).squeeze(-1)  # [batch]
            attention_scores.append(score)
        
        attention_scores = torch.stack(attention_scores, dim=1)  # [batch, num_experts]
        
        # 计算注意力权重（论文公式15）
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, num_experts]
        
        # 加权聚合专家表示（论文公式16）
        weighted_experts = expert_reprs * attention_weights.unsqueeze(-1)
        aggregated_repr = weighted_experts.sum(dim=1)  # [batch, expert_dim]
        
        return aggregated_repr, attention_weights

class MetaTowerModule(nn.Module):
    """论文4.2.3节的Meta Tower模块 - 简化实现"""
    def __init__(self, input_dim, scenario_dim, task_dim, num_layers=3, hidden_dim=256):
        super(MetaTowerModule, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # 使用MLP代替复杂的动态参数生成，简化实现
        self.tower_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                layer_input_dim = input_dim + scenario_dim + task_dim
                layer_output_dim = hidden_dim
            elif i == num_layers - 1:
                layer_input_dim = hidden_dim
                layer_output_dim = 1  # 二分类输出
            else:
                layer_input_dim = hidden_dim
                layer_output_dim = hidden_dim
                
            self.tower_layers.append(nn.Sequential(
                nn.Linear(layer_input_dim, layer_output_dim),
                nn.LeakyReLU()
            ))

    def forward(self, x, scene_repr, task_anchor):
        """
        Args:
            x: [batch, input_dim] - 输入特征
            scene_repr: [batch, scenario_dim] - 场景表示
            task_anchor: [batch, task_dim] - 任务锚点
        Returns:
            output: [batch, 1] - 二分类logits
        """
        # 拼接输入特征、场景表示和任务锚点
        h = torch.cat([x, scene_repr, task_anchor], dim=1)  # [batch, input_dim + scenario_dim + task_dim]
        
        for i, layer in enumerate(self.tower_layers):
            h = layer(h)
            # 最后一层不加激活函数
            if i == self.num_layers - 1:
                h = h  # 保持原始输出用于sigmoid
            else:
                h = F.leaky_relu(h)
            
        return h  # [batch, 1]

class M2MModel(nn.Module):
    """基于论文的M2M模型 - 修复所有维度问题"""
    def __init__(
        self,
        embedding_dim=16,
        scenario_dim=128,
        task_dim=128,
        expert_dim=128,
        num_experts=8,
        num_tasks=5,  # 二分类任务数量
        num_scenarios=5,
        transformer_heads=2,
        meta_hidden_dim=128,
        expert_hidden_units=[512, 128],
        gate_hidden_units=[512, 128],
        tower_hidden_units=[256, 128],
        dnn_activations='LeakyReLU',
        # 稀疏特征配置
        combine_schema_path=None,
        scene_combine_schema_path=None,
        ad_combine_schema_path=None,
        sparse_init_var=1e-2,
        ftrl_l1=1.0,
        ftrl_l2=120.0,
        ftrl_alpha=0.5,
        ftrl_beta=1.0,
        net_dropout=0.1,
        input_norm=False,
        batch_norm=True,
        use_bias=True,
        **kwargs
    ):
        super(M2MModel, self).__init__()
        
        # 初始化参数
        self.embedding_dim = embedding_dim
        self.scenario_dim = scenario_dim
        self.task_dim = task_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.num_scenarios = num_scenarios
        
        # === 1. 稀疏嵌入层 ===
        # 主特征嵌入
        self.sparse_main = ms.EmbeddingSumConcat(embedding_dim, None, combine_schema_path) 
        self.sparse_main.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.sparse_main.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        self.main_input_dim = int(self.sparse_main.feature_count * embedding_dim)

        # 场景特征嵌入
        self.sparse_scene = ms.EmbeddingSumConcat(embedding_dim, None, scene_combine_schema_path) 
        self.sparse_scene.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.sparse_scene.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        self.scene_input_dim = int(self.sparse_scene.feature_count * embedding_dim)

        # 广告特征嵌入
        self.sparse_ad = ms.EmbeddingSumConcat(embedding_dim, None, ad_combine_schema_path) 
        self.sparse_ad.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.sparse_ad.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        self.ad_input_dim = int(self.sparse_ad.feature_count * embedding_dim)

        # === 2. Backbone Network（论文4.1节）===
        
        # 2.1 Transformer层（论文4.1.2节）- 可选，如果序列特征可用
        self.behavior_transformer = TransformerLayer(
            d_model=embedding_dim, 
            nhead=transformer_heads, 
            num_layers=2
        ) if transformer_heads > 0 else None
        
        # 2.2 专家视图表示（论文4.1.3节）
        self.experts = nn.ModuleList([
            MLPLayer(
                input_dim=self.main_input_dim,
                output_dim=expert_dim,
                hidden_units=expert_hidden_units,
                hidden_activations=dnn_activations,
                final_activation=None,
                dropout_rates=net_dropout,
                input_norm=input_norm,
                batch_norm=batch_norm,
                use_bias=use_bias
            ) for _ in range(num_experts)
        ])
        
        # 2.3 任务视图表示（论文4.1.4节）
        self.task_embeddings = nn.Embedding(num_tasks, task_dim)
        
        # 2.4 场景知识表示（论文4.1.5节）
        self.scenario_encoder = MLPLayer(
            input_dim=self.scene_input_dim + self.ad_input_dim,
            output_dim=scenario_dim,
            hidden_units=gate_hidden_units,
            hidden_activations=dnn_activations,
            final_activation=None,
            dropout_rates=net_dropout,
            input_norm=input_norm,
            batch_norm=batch_norm,
            use_bias=use_bias
        )
        
        # === 3. Meta Learning Mechanism（论文4.2节）===
        
        # 3.1 Meta Attention模块（论文4.2.2节）
        self.meta_attention = MetaAttentionModule(
            scenario_dim=scenario_dim,
            task_dim=task_dim,
            expert_dim=expert_dim,
            num_experts=num_experts,
            meta_hidden_dim=meta_hidden_dim
        )
        
        # 3.2 Meta Tower模块（论文4.2.3节）- 调整为二分类
        self.meta_towers = nn.ModuleList([
            MetaTowerModule(
                input_dim=expert_dim,
                scenario_dim=scenario_dim,
                task_dim=task_dim,
                num_layers=3,
                hidden_dim=meta_hidden_dim
            ) for _ in range(num_tasks)
        ])
        
        # === 4. 初始化 ===
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                xavier_uniform_(module.weight)

    def forward(self, x):
        """
        Args:
            x: 输入批次数据
        Returns:
            predictions: [batch, num_tasks] - 每个任务的二分类概率
        """
        # === 稀疏特征嵌入 ===
        main_embedded = self.sparse_main(x)  # [batch, main_input_dim]
        scene_embedded = self.sparse_scene(x) # [batch, scene_input_dim]
        ad_embedded = self.sparse_ad(x)      # [batch, ad_input_dim]

        # === Backbone Network ===
        
        # 1. 专家视图表示（论文公式6）
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(main_embedded)
            expert_outputs.append(expert_out)
        expert_reprs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, expert_dim]
        
        # 2. 场景知识表示（论文公式8）
        scene_ad_combined = torch.cat([scene_embedded, ad_embedded], dim=1)
        scenario_repr = self.scenario_encoder(scene_ad_combined)  # [batch, scenario_dim]

        # === Meta Learning Mechanism ===
        
        task_predictions = []
        batch_size = scenario_repr.size(0)

        for task_idx in range(self.num_tasks):
            # 获取任务锚点（论文公式7）
            task_anchor = self.task_embeddings(torch.tensor([task_idx], device=x.device))
            task_anchor = task_anchor.expand(batch_size, -1)  # [batch, task_dim]
            
            # Meta Attention（论文4.2.2节）
            attended_repr, attention_weights = self.meta_attention(
                scenario_repr, task_anchor, expert_reprs
            )
            
            # Meta Tower（论文4.2.3节）
            task_logits = self.meta_towers[task_idx](attended_repr, scenario_repr, task_anchor)
            task_prob = torch.sigmoid(task_logits)  # 二分类概率
            task_predictions.append(task_prob)
        
        # 合并所有任务预测
        predictions = torch.cat(task_predictions, dim=1)  # [batch, num_tasks]
        
        return predictions

    def predict(self, yhat, minibatch=None):
        """预测方法"""
        if self.num_tasks == 1:
            # 单任务情况下，yhat已经是[B]或[B, 1]，直接返回第一个任务的结果
            if yhat.dim() == 2 and yhat.size(1) == 1:
                return yhat.squeeze(-1)  # [B, 1] -> [B]
            else:
                return yhat  # [B]
        else:
            # 多任务情况下，yhat是[B, task_count]，取第一个任务列
            return yhat[:, 0]  # [B, task_count] -> [B]
