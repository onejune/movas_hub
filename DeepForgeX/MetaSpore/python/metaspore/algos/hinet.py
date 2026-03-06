import torch
import metaspore as ms
import torch.nn.functional as F

from .layers import MLPLayer

class SubExpertIntegration(torch.nn.Module):
    """
    Sub Expert Integration (SEI) module using metaspore interface
    """
    def __init__(self, input_dim, hidden_dim, num_sub_experts=4, sparse_init_var=1e-2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_sub_experts = num_sub_experts
        
        # Create sub-experts using MLP layers
        self.sub_experts = torch.nn.ModuleList([
            MLPLayer(
                input_dim=input_dim,
                output_dim=input_dim,
                hidden_units=[hidden_dim],
                hidden_activations='ReLU',
                final_activation=None
            ) for _ in range(num_sub_experts)
        ])
        
        # Initialize sub-experts
        for expert in self.sub_experts:
            if hasattr(expert, 'initializer'):
                expert.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
        # Gate for combining sub-experts
        self.gate = torch.nn.Linear(input_dim, num_sub_experts)
        self.gate.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        gates = torch.softmax(self.gate(x), dim=-1)  # [batch_size, num_sub_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.sub_experts], dim=1)  # [batch_size, num_sub_experts, input_dim]
        output = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)  # [batch_size, input_dim]
        return output

class ScenarioSharedExpert(torch.nn.Module):
    """
    Scenario-shared expert using SEI with metaspore interface
    """
    def __init__(self, input_dim, hidden_dim, num_sub_experts=4, sparse_init_var=1e-2):
        super().__init__()
        self.sei = SubExpertIntegration(input_dim, hidden_dim, num_sub_experts, sparse_init_var)

    def forward(self, x):
        return self.sei(x)

class ScenarioSpecificExpert(torch.nn.Module):
    """
    Scenario-specific expert network with metaspore interface
    """
    def __init__(self, input_dim, hidden_dim, sparse_init_var=1e-2):
        super().__init__()
        self.net = MLPLayer(
            input_dim=input_dim,
            output_dim=input_dim,
            hidden_units=[hidden_dim],
            hidden_activations='ReLU',
            final_activation=None
        )
        # Initialize if possible
        if hasattr(self.net, 'initializer'):
            self.net.initializer = ms.NormalTensorInitializer(var=sparse_init_var)

    def forward(self, x):
        return self.net(x)

class ScenarioAwareAttention(torch.nn.Module):
    """
    Scenario-Aware Attention Network (SAN) with metaspore interface
    """
    def __init__(self, input_dim, num_scenarios, sparse_init_var=1e-2):
        super().__init__()
        self.num_scenarios = num_scenarios
        self.scenario_embeddings = ms.EmbeddingSumConcat(
            input_dim, 
            column_name_path=None, 
            combine_schema_path=None
        )
        self.scenario_embeddings.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
        self.attention_net = MLPLayer(
            input_dim=input_dim * 2,
            output_dim=1,
            hidden_units=[input_dim // 2],
            hidden_activations='ReLU',
            final_activation=None
        )
        if hasattr(self.attention_net, 'initializer'):
            self.attention_net.initializer = ms.NormalTensorInitializer(var=sparse_init_var)

    def forward(self, shared_reprs, specific_reprs, current_scenario_id):
        # shared_reprs: list of [batch_size, input_dim] for each scenario
        # specific_reprs: list of [batch_size, input_dim] for each scenario
        # current_scenario_id: int
        batch_size = shared_reprs[0].size(0)
        current_shared = shared_reprs[current_scenario_id]  # [batch_size, input_dim]
        current_specific = specific_reprs[current_scenario_id]  # [batch_size, input_dim]

        # Get scenario embeddings
        scenario_embed = self.scenario_embeddings.weight if hasattr(self.scenario_embeddings, 'weight') else \
                         torch.randn(self.num_scenarios, current_shared.size(-1)).to(current_shared.device)
        
        expanded_current = current_shared.unsqueeze(0).expand(self.num_scenarios, -1, -1)  # [num_scenarios, batch_size, input_dim]
        expanded_scenario_embed = scenario_embed.unsqueeze(1).expand(-1, batch_size, -1)  # [num_scenarios, batch_size, input_dim]

        combined = torch.cat([expanded_current, expanded_scenario_embed], dim=-1)  # [num_scenarios, batch_size, input_dim*2]
        attention_logits = self.attention_net(combined).squeeze(-1)  # [num_scenarios, batch_size]
        attention_weights = torch.softmax(attention_logits.transpose(0, 1), dim=-1)  # [batch_size, num_scenarios]

        weighted_sum_shared = torch.zeros_like(current_shared)
        weighted_sum_specific = torch.zeros_like(current_specific)
        for i in range(self.num_scenarios):
            weight = attention_weights[:, i].unsqueeze(-1)  # [batch_size, 1]
            weighted_sum_shared += weight * shared_reprs[i]
            weighted_sum_specific += weight * specific_reprs[i]

        # Concatenate and project back
        enhanced_repr = torch.cat([current_shared, weighted_sum_shared, current_specific, weighted_sum_specific], dim=-1)
        enhanced_repr = torch.nn.Linear(enhanced_repr.size(-1), current_shared.size(-1)).to(enhanced_repr.device)(enhanced_repr)
        return enhanced_repr

class ScenarioExtractionLayer(torch.nn.Module):
    """
    Scenario Extraction Layer: Combines shared and specific experts with SAN using metaspore interface
    """
    def __init__(self, input_dim, hidden_dim, num_scenarios, num_sub_experts=4, sparse_init_var=1e-2):
        super().__init__()
        self.num_scenarios = num_scenarios
        self.shared_expert = ScenarioSharedExpert(input_dim, hidden_dim, num_sub_experts, sparse_init_var)
        self.specific_experts = torch.nn.ModuleList([
            ScenarioSpecificExpert(input_dim, hidden_dim, sparse_init_var) for _ in range(num_scenarios)
        ])
        self.san = ScenarioAwareAttention(input_dim, num_scenarios, sparse_init_var)

    def forward(self, x, scenario_ids):
        """
        Process batch with potentially mixed scenarios.
        x: [batch_size, input_dim]
        scenario_ids: [batch_size] tensor of integers indicating scenario for each sample
        """
        # Scenario extraction - process each sample according to its scenario
        scenario_outputs = []
        unique_scenarios = torch.unique(scenario_ids)
        for s_id in unique_scenarios:
            mask = (scenario_ids == s_id)
            masked_input = x[mask]
            # Process using the specific scenario's logic
            shared_out = self.shared_expert(masked_input)
            specific_out = self.specific_experts[s_id.item()](masked_input)
            
            # Prepare inputs for SAN
            all_shared = [shared_out if i == s_id.item() else torch.zeros_like(shared_out) for i in range(self.num_scenarios)]
            all_specific = [specific_out if i == s_id.item() else torch.zeros_like(specific_out) for i in range(self.num_scenarios)]
            
            enhanced = self.san(all_shared, all_specific, s_id.item())
            scenario_outputs.append(enhanced)
        
        # Reconstruct scenario-processed features in original batch order
        scenario_processed = torch.zeros_like(x)
        idx = 0
        for s_id in unique_scenarios:
            mask = (scenario_ids == s_id)
            count = mask.sum().item()
            scenario_processed[mask] = scenario_outputs[idx][:count]
            idx += 1
        
        return scenario_processed

class CustomizedGateControl(torch.nn.Module):
    """
    Customized Gate Control (CGC) for Task Extraction Layer with metaspore interface
    Supports multi-task processing where each sample can belong to multiple tasks.
    """
    def __init__(self, input_dim, task_expert_dims, num_tasks, shared_expert_dim=None, sparse_init_var=1e-2):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_expert_dims = task_expert_dims
        self.shared_expert_dim = shared_expert_dim or input_dim

        # Shared experts
        self.shared_experts = torch.nn.ModuleList([
            MLPLayer(
                input_dim=input_dim,
                output_dim=input_dim,
                hidden_units=[self.shared_expert_dim],
                hidden_activations='ReLU',
                final_activation=None
            )
        ])
        
        # Initialize shared experts
        for expert in self.shared_experts:
            if hasattr(expert, 'initializer'):
                expert.initializer = ms.NormalTensorInitializer(var=sparse_init_var)

        # Task-specific experts
        self.task_experts = torch.nn.ModuleList([
            torch.nn.ModuleList([
                MLPLayer(
                    input_dim=input_dim,
                    output_dim=input_dim,
                    hidden_units=[task_dim],
                    hidden_activations='ReLU',
                    final_activation=None
                ) for _ in range(2)  # 2 experts per task
            ]) for task_dim in task_expert_dims
        ])
        
        # Initialize task experts
        for task_exps in self.task_experts:
            for exp in task_exps:
                if hasattr(exp, 'initializer'):
                    exp.initializer = ms.NormalTensorInitializer(var=sparse_init_var)

        # Gates for each task - now computes gates for ALL tasks for each sample
        self.task_gates = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1 + 2)  # 1 shared + 2 task-specific
            for _ in range(num_tasks)
        ])
        
        # Initialize gates
        for gate in self.task_gates:
            gate.initializer = ms.NormalTensorInitializer(var=sparse_init_var)

    def forward(self, x, task_multihot):
        """
        Process batch with potentially mixed tasks per sample using multi-hot encoding.
        x: [batch_size, input_dim]
        task_multihot: [batch_size, num_tasks] binary tensor indicating which tasks each sample belongs to
        """
        # Compute outputs from shared and task-specific experts for the entire batch
        shared_outputs = [expert(x) for expert in self.shared_experts] # List of [batch_size, input_dim]
        task_specific_outputs = [
            [exp(x) for exp in self.task_experts[tid]] # For each task, compute both experts
            for tid in range(self.num_tasks)
        ] # List of lists: [[exp0_task0, exp1_task0], [exp0_task1, exp1_task1], ...]

        # Compute gates for each task for each sample
        all_task_outputs = []
        for t_id in range(self.num_tasks):
            gate_input = x # Use same input for all tasks
            gate_weights = torch.softmax(self.task_gates[t_id](gate_input), dim=-1) # [batch_size, 3]
            
            # Combine shared and task-specific contributions for this task
            shared_contrib = gate_weights[:, 0:1] * shared_outputs[0] # [batch_size, input_dim]
            task_contrib1 = gate_weights[:, 1:2] * task_specific_outputs[t_id][0] # [batch_size, input_dim]
            task_contrib2 = gate_weights[:, 2:3] * task_specific_outputs[t_id][1] # [batch_size, input_dim]
            
            task_output = shared_contrib + task_contrib1 + task_contrib2 # [batch_size, input_dim]
            all_task_outputs.append(task_output) # List of [batch_size, input_dim] for each task
        
        # Stack outputs: [num_tasks, batch_size, input_dim]
        stacked_task_outputs = torch.stack(all_task_outputs, dim=0)
        
        # Transpose to [batch_size, num_tasks, input_dim]
        transposed_outputs = stacked_task_outputs.transpose(0, 1)
        
        # Mask outputs based on task_multihot: [batch_size, num_tasks, 1]
        mask = task_multihot.unsqueeze(-1).float()
        
        # Apply mask: [batch_size, num_tasks, input_dim]
        masked_outputs = transposed_outputs * mask
        
        return masked_outputs # Shape: [batch_size, num_tasks, input_dim]

class TaskExtractionLayer(torch.nn.Module):
    """
    Task Extraction Layer using CGC with multi-task support
    """
    def __init__(self, input_dim, task_expert_dims, num_tasks, sparse_init_var=1e-2):
        super().__init__()
        self.cgc = CustomizedGateControl(input_dim, task_expert_dims, num_tasks, sparse_init_var=sparse_init_var)

    def forward(self, x, task_multihot):
        return self.cgc(x, task_multihot)

class HiNet(torch.nn.Module):
    """
    Hierarchical information extraction Network (HiNet) with metaspore interface
    Supports multi-task multi-scenario learning.
    """
    def __init__(self, 
                 scenario_count=6,           # Number of scenarios
                 task_count=2,               # Number of tasks
                 embedding_dim=16,           # Embedding dimension
                 base_dnn_hidden_units=[512, 256],  # Base DNN hidden units
                 task_towers_hidden_units=[128],    # Task towers hidden units
                 scenario_expert_hidden_dim=64,     # Hidden dim for scenario experts
                 task_expert_dims=None,             # Hidden dims for task experts
                 num_sub_experts=4,                 # Number of sub-experts in SEI
                 sparse_init_var=1e-2,              # Sparse initialization variance
                 scale_factor=1.0,                  # Scale factor for gates
                 batch_norm=False,                  # Whether to use batch norm
                 net_dropout=None,                  # Network dropout rate
                 ftrl_l1=1.0,                       # FTRL L1 regularization
                 ftrl_l2=120.0,                     # FTRL L2 regularization
                 ftrl_alpha=0.5,                    # FTRL alpha parameter
                 ftrl_beta=1.0,                     # FTRL beta parameter
                 column_name_path=None,             # Feature column name path
                 combine_schema_path=None):         # Feature combination schema path
        super().__init__()
        
        self.scenario_count = scenario_count
        self.task_count = task_count
        self.embedding_dim = embedding_dim
        self.scenario_expert_hidden_dim = scenario_expert_hidden_dim
        self.task_expert_dims = task_expert_dims or [64, 64]  # Default to 64 for each task
        self.scale_factor = scale_factor
        
        # Shared embedding layer
        self.shared_embedding = ms.EmbeddingSumConcat(
            embedding_dim, 
            column_name_path, 
            combine_schema_path
        )
        self.shared_embedding.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1, 
            l2=ftrl_l2, 
            alpha=ftrl_alpha, 
            beta=ftrl_beta
        )
        self.shared_embedding.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
        # Input dimension after embeddings
        input_dim = int(self.shared_embedding.feature_count * embedding_dim) if hasattr(self.shared_embedding, 'feature_count') else embedding_dim
        
        # Shared normalization layer
        self.shared_bn = ms.nn.Normalization(input_dim)
        
        # Scenario Extraction Layer
        self.scenario_extraction = ScenarioExtractionLayer(
            input_dim=input_dim,
            hidden_dim=scenario_expert_hidden_dim,
            num_scenarios=scenario_count,
            num_sub_experts=num_sub_experts,
            sparse_init_var=sparse_init_var
        )
        
        # Task Extraction Layer - now supports multi-task
        self.task_extraction = TaskExtractionLayer(
            input_dim=input_dim,
            task_expert_dims=self.task_expert_dims,
            num_tasks=task_count,
            sparse_init_var=sparse_init_var
        )
        
        # Task-specific towers - each processes its own feature representation
        self.task_towers = torch.nn.ModuleList()
        for _ in range(task_count):
            tower = MLPLayer(
                input_dim=input_dim,  # Each task gets its own processed feature vector
                output_dim=1,
                hidden_units=task_towers_hidden_units,
                hidden_activations='ReLU',
                final_activation=None,
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            )
            if hasattr(tower, 'initializer'):
                tower.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
            self.task_towers.append(tower)
        
        # Final activation function
        self.final_activation = torch.nn.Sigmoid()
    
    def forward(self, x):
        # Get current scenario and task info from internal state
        scenario_ids = getattr(self, '_current_domain_ids', None)
        task_multihot = getattr(self, '_current_task_ids', None)
        
        if scenario_ids is None or task_multihot is None:
            raise ValueError("Scenario IDs and Task multihot must be set via do_extra_work before forward pass")
        
        # Shared embedding
        shared_emb = self.shared_embedding(x)
        shared_emb_bn = self.shared_bn(shared_emb)
        
        # Scenario extraction - process each sample according to its scenario
        scenario_processed = self.scenario_extraction(shared_emb_bn, scenario_ids)
        
        # Task extraction - process samples considering all their relevant tasks using multi-hot mask
        # Output shape: [batch_size, num_tasks, input_dim]
        task_processed_per_task = self.task_extraction(scenario_processed, task_multihot)
        
        # Task-specific predictions
        task_predictions = []
        for t_id in range(self.task_count):
            # Select the features processed for task t_id: [batch_size, input_dim]
            task_features = task_processed_per_task[:, t_id, :]
            # Pass through the corresponding task tower
            task_output = self.task_towers[t_id](task_features)
            task_prediction = self.final_activation(task_output)
            task_predictions.append(task_prediction)
        
        prediction = torch.cat(task_predictions, dim=1)
        
        # 根据是否存在 _current_task_ids 决定返回的内容
        if hasattr(self, '_current_task_ids') and self._current_task_ids is not None:
            # 如果存在 _current_task_ids，则返回对应 task 的预测结果
            batch_size = prediction.size(0)
            selected_predictions = torch.zeros(batch_size, 1, device=prediction.device)
            
            for idx, task_id in enumerate(self._current_task_ids):
                selected_predictions[idx, 0] = prediction[idx, task_id]
            self._current_task_ids = None
            return selected_predictions
        else:
            # 如果不存在 _current_task_ids，则返回所有 task 的预测结果
            return prediction

    def do_extra_work(self, minibatch):
        if hasattr(minibatch, 'domain_id'):
            domain_id_data = minibatch.domain_id
        elif isinstance(minibatch, dict) and 'domain_id' in minibatch:
            domain_id_data = minibatch['domain_id']
        else:
             # Assume it's a DataFrame-like object
            domain_id_data = minibatch['domain_id'].values

        if not isinstance(domain_id_data, torch.Tensor):
            self._current_domain_ids = torch.tensor(domain_id_data, dtype=torch.long)
        else:
            self._current_domain_ids = domain_id_data.long()

        # --- Extract task_id ---
        if hasattr(minibatch, 'task_id'): # Changed from task_ids to task_id for singular
            task_id_data = minibatch.task_id
        elif isinstance(minibatch, dict) and 'task_id' in minibatch:
            task_id_data = minibatch['task_id']
        else:
            task_id_data = minibatch['task_id'].values # Changed from task_ids

        if not isinstance(task_id_data, torch.Tensor):
            self._current_task_ids = torch.tensor(task_id_data, dtype=torch.long)
        else:
            self._current_task_ids = task_id_data.long()
