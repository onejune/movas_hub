import torch
import torch.nn as nn
import torch.nn.functional as F
import metaspore as ms
from ...layers import MLPLayer

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(βx)"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
        
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class LoRA(nn.Module):
    """Low-Rank Adaptation module for feature-gate"""
    def __init__(self, input_dim, rank=8):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(input_dim, rank, bias=False)
        self.B = nn.Linear(rank, input_dim, bias=False)
        nn.init.zeros_(self.B.weight)
    
    def forward(self, x):
        return self.B(self.A(x))

class FeatureGate(nn.Module):
    """Feature-gate mechanism: 2 * Sigmoid(LORA(v))"""
    def __init__(self, input_dim, rank=8):
        super().__init__()
        self.lora = LoRA(input_dim, rank)
    
    def forward(self, v):
        lora_out = self.lora(v)
        importance_scores = 2.0 * torch.sigmoid(lora_out)
        return importance_scores

class SelfGate(nn.Module):
    """Self-gate mechanism for expert outputs"""
    def __init__(self, input_dim, num_experts=1):
        super().__init__()
        self.num_experts = num_experts
        self.gate_mlp = MLPLayer(
            input_dim=input_dim,
            output_dim=num_experts,
            hidden_units=[],
            hidden_activations='Swish',
            final_activation=None,
            use_bias=True
        )
        self.activation = nn.Sigmoid() if num_experts == 1 else nn.Softmax(dim=-1)
    
    def forward(self, expert_outputs):
        batch_size, num_experts, expert_dim = expert_outputs.shape
        # Compute gate weights for each expert output
        reshaped_experts = expert_outputs.view(-1, expert_dim) # [B*num_experts, expert_dim]
        gate_logits = self.gate_mlp(reshaped_experts) # [B*num_experts, num_experts]
        gate_logits = gate_logits.view(batch_size, num_experts, num_experts) # [B, num_experts, num_experts]
        
        # Use diagonal for self-weighting (each expert gates itself)
        self_gate_weights = torch.diagonal(gate_logits, dim1=1, dim2=2) # [B, num_experts]
        self_gate_weights = self_gate_weights.unsqueeze(-1) # [B, num_experts, 1]
        return expert_outputs * self_gate_weights # [B, num_experts, expert_dim]

class HoMEExpert(nn.Module):
    """HoME Expert with normalization and Swish activation"""
    def __init__(self, input_dim, output_dim, hidden_units=[], activation='Swish', 
                 input_norm=False, batch_norm=False, use_bias=True):
        super().__init__()
        self.mlp = MLPLayer(input_dim, output_dim, hidden_units, activation,
                           final_activation=None, input_norm=input_norm,
                           batch_norm=batch_norm, use_bias=use_bias)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        output = self.mlp(x)
        output = self.layer_norm(output)
        return output

class HoMEMetaLayer(nn.Module):
    """
    First level of HoME: Generate meta representations (z_meta_watch, z_meta_shared, z_meta_inter).
    Creates shared and group-specific experts.
    """
    def __init__(self, input_dim, expert_out_dim, task_groups, 
                 num_shared_experts=2, num_group_experts=2,
                 expert_hidden_units=[64, 32], gate_hidden_units=[32],
                 dnn_activations='Swish', use_bias=True, input_norm=False, batch_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.expert_out_dim = expert_out_dim
        self.task_groups = task_groups
        self.num_shared_experts = num_shared_experts
        self.num_group_experts = num_group_experts
        self.dnn_activations = dnn_activations
        self.use_bias = use_bias
        self.input_norm = input_norm
        self.batch_norm = batch_norm
        
        # Identify unique groups
        self.group_names = [f"group_{i}" for i in range(len(task_groups))]
        self.num_groups = len(self.group_names)
        
        # Create global shared experts (for z_meta_shared)
        self.shared_experts = nn.ModuleList([
            HoMEExpert(
                input_dim=input_dim,
                output_dim=expert_out_dim,
                hidden_units=expert_hidden_units,
                activation=dnn_activations,
                input_norm=input_norm,
                batch_norm=batch_norm,
                use_bias=use_bias
            ) for _ in range(num_shared_experts)
        ])
        self.shared_gate = MLPLayer(
            input_dim=input_dim,
            output_dim=num_shared_experts,
            hidden_units=gate_hidden_units,
            hidden_activations=dnn_activations,
            final_activation=None,
            input_norm=input_norm,
            batch_norm=batch_norm,
            use_bias=use_bias
        )
        
        # Create group-specific experts (for z_meta_watch, z_meta_inter, etc.)
        self.group_experts = nn.ModuleDict()
        self.group_gates = nn.ModuleDict()
        for group_name in self.group_names:
            experts_for_group = nn.ModuleList([
                HoMEExpert(
                    input_dim=input_dim,
                    output_dim=expert_out_dim,
                    hidden_units=expert_hidden_units,
                    activation=dnn_activations,
                    input_norm=input_norm,
                    batch_norm=batch_norm,
                    use_bias=use_bias
                ) for _ in range(num_group_experts)
            ])
            self.group_experts[group_name] = experts_for_group
            
            self.group_gates[group_name] = MLPLayer(
                input_dim=input_dim,
                output_dim=num_group_experts,
                hidden_units=gate_hidden_units,
                hidden_activations=dnn_activations,
                final_activation=None,
                input_norm=input_norm,
                batch_norm=batch_norm,
                use_bias=use_bias
            )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, v):
        # v: [batch_size, input_dim]
        
        # Compute z_meta_shared
        shared_expert_outs = torch.stack([
            expert(v) for expert in self.shared_experts
        ], dim=1) # [B, num_shared, D]
        shared_gate_logits = self.shared_gate(v) # [B, num_shared]
        shared_weights = self.softmax(shared_gate_logits).unsqueeze(-1) # [B, num_shared, 1]
        z_meta_shared = torch.sum(shared_expert_outs * shared_weights, dim=1) # [B, D]
        
        # Compute z_meta_group for each group
        z_meta_dict = {}
        for group_name in self.group_names:
            group_expert_outs = torch.stack([
                expert(v) for expert in self.group_experts[group_name]
            ], dim=1) # [B, num_group_experts, D]
            group_gate_logits = self.group_gates[group_name](v) # [B, num_group_experts]
            group_weights = self.softmax(group_gate_logits).unsqueeze(-1) # [B, num_group_experts, 1]
            z_meta_dict[group_name] = torch.sum(group_expert_outs * group_weights, dim=1) # [B, D]
            
        return z_meta_dict, z_meta_shared

class HoMESecondLayer(nn.Module):
    """
    Second level of HoME: Task-specific prediction using meta representations.
    Creates its own set of experts for the second level.
    """
    def __init__(self, meta_dim, expert_out_dim, task_numb, task_to_group_map,
                 num_shared_experts_second=2, num_group_experts_second=2,
                 expert_hidden_units=[64, 32], gate_hidden_units=[32], tower_hidden_units=[32, 16],
                 dnn_activations='Swish', use_bias=True, input_norm=False, batch_norm=False,
                 net_dropout=None, feature_gate_enabled=True, self_gate_enabled=True, feature_gate_rank=8):
        super().__init__()
        self.task_numb = task_numb
        self.meta_dim = meta_dim
        self.expert_out_dim = expert_out_dim
        self.task_to_group_map = task_to_group_map
        self.feature_gate_enabled = feature_gate_enabled
        self.self_gate_enabled = self_gate_enabled
        
        # Second level experts (input dimension is 2 * meta_dim because of concatenation)
        gate_input_dim = meta_dim * 2  # z_meta_shared + z_meta_group
        
        # Create second level global shared experts (for task-specific combination)
        self.shared_experts_second = nn.ModuleList([
            HoMEExpert(
                input_dim=gate_input_dim,
                output_dim=expert_out_dim,
                hidden_units=expert_hidden_units,
                activation=dnn_activations,
                input_norm=input_norm,
                batch_norm=batch_norm,
                use_bias=use_bias
            ) for _ in range(num_shared_experts_second)
        ])
        
        # Create second level group-specific experts (for task-specific combination)
        self.group_experts_second = nn.ModuleDict()
        self.num_groups = max(task_to_group_map.values()) + 1
        for group_id in range(self.num_groups):
            group_name = f"group_{group_id}"
            experts_for_group = nn.ModuleList([
                HoMEExpert(
                    input_dim=gate_input_dim,
                    output_dim=expert_out_dim,
                    hidden_units=expert_hidden_units,
                    activation=dnn_activations,
                    input_norm=input_norm,
                    batch_norm=batch_norm,
                    use_bias=use_bias
                ) for _ in range(num_group_experts_second)
            ])
            self.group_experts_second[group_name] = experts_for_group

        # Task-specific components
        self.feature_gates = nn.ModuleList() if feature_gate_enabled else None
        self.task_specific_gates = nn.ModuleList()
        self.towers = nn.ModuleList()
        self.self_gates = nn.ModuleList() if self_gate_enabled else None

        for task_id in range(task_numb):
            if feature_gate_enabled:
                self.feature_gates.append(FeatureGate(gate_input_dim, feature_gate_rank))
            
            # Gate to combine shared experts and group experts for this task
            # Input: concatenated meta representations relevant to the task
            group_id = task_to_group_map[task_id]
            group_name = f"group_{group_id}"
            num_experts_for_task = num_shared_experts_second + num_group_experts_second
            self.task_specific_gates.append(
                MLPLayer(
                    input_dim=gate_input_dim,
                    output_dim=num_experts_for_task,
                    hidden_units=gate_hidden_units,
                    hidden_activations=dnn_activations,
                    final_activation=None,
                    input_norm=input_norm,
                    batch_norm=batch_norm,
                    use_bias=use_bias
                )
            )
            
            self.towers.append(
                MLPLayer(
                    input_dim=expert_out_dim, # Output from gated experts
                    output_dim=1, # Single output per task
                    hidden_units=tower_hidden_units,
                    hidden_activations=dnn_activations,
                    final_activation='Sigmoid',
                    dropout_rates=net_dropout,
                    input_norm=input_norm,
                    batch_norm=batch_norm,
                    use_bias=use_bias
                )
            )
            
            if self_gate_enabled:
                # The number of experts for this task is the sum of shared and group experts
                self.self_gates.append(SelfGate(expert_out_dim, num_experts=num_experts_for_task))

    def forward(self, meta_repr_dict, z_meta_shared, v, task_ids):
        # meta_repr_dict: dict of group_name -> [batch_size, meta_dim]
        # z_meta_shared: [batch_size, meta_dim]
        # v: original input for feature gate
        # task_ids: list of task indices to compute
        
        predictions = []
        for task_id in task_ids:
            group_id = self.task_to_group_map[task_id]
            group_name = f"group_{group_id}"
            
            # Get relevant meta representations
            z_meta_group = meta_repr_dict[group_name] # [B, D]
            
            # Combine meta representations (as per paper examples)
            combined_meta = torch.cat([z_meta_shared, z_meta_group], dim=1) # [B, 2*D]
            
            # Apply feature gate if enabled
            if self.feature_gate_enabled:
                feat_gate_scores = self.feature_gates[task_id](combined_meta)
                combined_meta = combined_meta * feat_gate_scores
            
            # Get outputs from second level experts
            # 1. Second level global shared experts
            shared_exp_outs = torch.stack([
                expert(combined_meta) for expert in self.shared_experts_second
            ], dim=1) # [B, num_shared_second, D]
            
            # 2. Second level group-specific experts for this task's group
            group_exp_outs = torch.stack([
                expert(combined_meta) for expert in self.group_experts_second[group_name]
            ], dim=1) # [B, num_group_second, D]
            
            # Concatenate all expert outputs for this task
            all_expert_outs = torch.cat([shared_exp_outs, group_exp_outs], dim=1) # [B, num_total, D]
            
            # Apply Self-Gate if enabled
            if self.self_gate_enabled:
                all_expert_outs = self.self_gates[task_id](all_expert_outs) # [B, num_total, D]
            
            # Compute gate weights for this task
            gate_out = self.task_specific_gates[task_id](combined_meta) # [B, num_total]
            gate_weights = torch.softmax(gate_out, dim=1).unsqueeze(-1) # [B, num_total, 1]
            
            # Weight and sum expert outputs
            weighted_experts = all_expert_outs * gate_weights # [B, num_total, D]
            aggregated_out = torch.sum(weighted_experts, dim=1) # [B, D]
            
            # Final prediction
            pred = self.towers[task_id](aggregated_out) # [B, 1]
            predictions.append(pred)
            
        return torch.cat(predictions, dim=1) # [B, len(task_ids)]

class HoME(nn.Module):
    """
    HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning
    Implements the key innovations from the paper with correct expert separation.
    """
    def __init__(self,
                 embedding_dim=10,
                 column_name_path=None,
                 combine_schema_path=None,
                 task_groups=None,  # List of lists, e.g., [[0,1], [2,3]]
                 task_numb=4,
                 num_shared_experts=2,
                 num_group_experts=2,
                 num_shared_experts_second=2,
                 num_group_experts_second=2,
                 meta_expert_hidden_units=[64, 32],
                 meta_gate_hidden_units=[32],
                 expert_hidden_units=[64, 32],
                 expert_out_dim=16,
                 gate_hidden_units=[32],
                 tower_hidden_units=[32, 16],
                 dnn_activations='Swish',
                 use_bias=True,
                 input_norm=False,
                 batch_norm=False,
                 net_dropout=None,
                 net_regularizer=None,
                 sparse_init_var=1e-2,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 feature_gate_enabled=True,
                 self_gate_enabled=True,
                 feature_gate_rank=8):
        super().__init__()
        
        self.task_numb = task_numb
        self.expert_out_dim = expert_out_dim
        self.feature_gate_enabled = feature_gate_enabled
        self.self_gate_enabled = self_gate_enabled
        self.task_groups = task_groups or [[i] for i in range(task_numb)]
        
        # Build task to group mapping
        self.task_to_group_map = {}
        for group_id, group_tasks in enumerate(self.task_groups):
            for task_id in group_tasks:
                self.task_to_group_map[task_id] = group_id
        
        # Set up embedding layer using metaspore
        self.embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path
        self.sparse = ms.EmbeddingSumConcat(self.embedding_dim, self.column_name_path, self.combine_schema_path)
        self.sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        self.input_dim = int(self.sparse.feature_count * self.embedding_dim)

        # First Level: Meta Representations (creates its own experts)
        self.meta_layer = HoMEMetaLayer(
            input_dim=self.input_dim,
            expert_out_dim=expert_out_dim,
            task_groups=task_groups,
            num_shared_experts=num_shared_experts,
            num_group_experts=num_group_experts,
            expert_hidden_units=meta_expert_hidden_units,
            gate_hidden_units=meta_gate_hidden_units,
            dnn_activations=dnn_activations,
            use_bias=use_bias,
            input_norm=input_norm,
            batch_norm=batch_norm
        )

        # Second Level: Task Predictions (creates its own experts with correct input dim)
        self.second_layer = HoMESecondLayer(
            meta_dim=expert_out_dim,
            expert_out_dim=expert_out_dim,
            task_numb=task_numb,
            task_to_group_map=self.task_to_group_map,
            num_shared_experts_second=num_shared_experts_second,
            num_group_experts_second=num_group_experts_second,
            expert_hidden_units=expert_hidden_units,
            gate_hidden_units=gate_hidden_units,
            tower_hidden_units=tower_hidden_units,
            dnn_activations=dnn_activations,
            use_bias=use_bias,
            input_norm=input_norm,
            batch_norm=batch_norm,
            net_dropout=net_dropout,
            feature_gate_enabled=feature_gate_enabled,
            self_gate_enabled=self_gate_enabled,
            feature_gate_rank=feature_gate_rank
        )
    
    def forward(self, x):
        # Process input through embedding
        v = self.sparse(x)
        
        # First level: Generate meta representations
        meta_repr_dict, z_meta_shared = self.meta_layer(v)
        
        # Second level: Generate task predictions using meta representations and its own experts
        task_ids = list(range(self.task_numb))
        prediction = self.second_layer(meta_repr_dict, z_meta_shared, v, task_ids)
        
        return prediction

    def predict(self, yhat, bid = None):
        #返回第一个任务的预测结果
        return yhat[:, 0]



