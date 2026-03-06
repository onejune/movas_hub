import torch
import metaspore as ms
import torch.nn.functional as F

from ..layers import MLPLayer

class SubExpertIntegration(torch.nn.Module):
    """
    Sub Expert Integration (SEI) module using metaspore interface.
    Combines multiple sub-experts using learned gating.
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
        
        # Initialize sub-experts weights if initializer attribute exists
        for expert in self.sub_experts:
            if hasattr(expert, 'initializer'):
                expert.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
        # Gate network to determine importance of each sub-expert
        self.gate = torch.nn.Linear(input_dim, num_sub_experts)
        # Initialize gate weights
        self.gate.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        # Compute gating coefficients for each sub-expert
        gates = torch.softmax(self.gate(x), dim=-1)  # [batch_size, num_sub_experts]
        
        # Compute outputs from all sub-experts
        expert_outputs = torch.stack([expert(x) for expert in self.sub_experts], dim=1)  
        # expert_outputs shape: [batch_size, num_sub_experts, input_dim]
        
        # Weighted sum of expert outputs based on gates
        # gates.unsqueeze(-1) shape: [batch_size, num_sub_experts, 1]
        # Element-wise multiplication broadcasts correctly
        output = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)  
        # output shape: [batch_size, input_dim]
        return output

class ScenarioSharedExpert(torch.nn.Module):
    """
    Scenario-shared expert using SEI with metaspore interface.
    Represents common knowledge across scenarios.
    """
    def __init__(self, input_dim, hidden_dim, num_sub_experts=4, sparse_init_var=1e-2):
        super().__init__()
        self.sei = SubExpertIntegration(input_dim, hidden_dim, num_sub_experts, sparse_init_var)

    def forward(self, x):
        return self.sei(x)

class ScenarioSpecificExpert(torch.nn.Module):
    """
    Scenario-specific expert network with metaspore interface.
    Learns knowledge unique to a particular scenario.
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
    Scenario-Aware Attention Network (SAN) with metaspore interface.
    Enhances representations by attending over scenario-related information.
    Note: Simplified implementation assuming fixed number of scenarios known at init.
    """
    def __init__(self, input_dim, num_scenarios, sparse_init_var=1e-2):
        super().__init__()
        self.num_scenarios = num_scenarios
        # Embeddings for each scenario
        self.scenario_embeddings = torch.nn.Embedding(num_scenarios, input_dim)
        self.scenario_embeddings.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
        # Attention mechanism: takes concatenation of current repr and scenario emb
        self.attention_net = MLPLayer(
            input_dim=input_dim * 2, # current repr + scenario emb
            output_dim=1,
            hidden_units=[input_dim // 2],
            hidden_activations='ReLU',
            final_activation=None
        )
        if hasattr(self.attention_net, 'initializer'):
            self.attention_net.initializer = ms.NormalTensorInitializer(var=sparse_init_var)

    def forward(self, shared_reprs, specific_reprs, current_scenario_id):
        """
        shared_reprs: list of [batch_size, input_dim] tensors, one for each scenario (or zero-padded).
                      Only the one at current_scenario_id index contains actual data for this batch segment.
        specific_reprs: list of [batch_size, input_dim] tensors, one for each scenario (or zero-padded).
                        Only the one at current_scenario_id index contains actual data for this batch segment.
        current_scenario_id: int, the ID of the scenario being processed.
        """
        # Get the actual representations for the current scenario
        batch_size = shared_reprs[0].size(0) # Assuming all have same batch size
        current_shared_repr = shared_reprs[current_scenario_id]  # [batch_size, input_dim]
        current_specific_repr = specific_reprs[current_scenario_id]  # [batch_size, input_dim]

        # Get embeddings for all scenarios
        all_scenario_indices = torch.arange(self.num_scenarios, device=current_shared_repr.device)
        all_scenario_embeddings = self.scenario_embeddings(all_scenario_indices) 
        # all_scenario_embeddings: [num_scenarios, input_dim]
        
        # Expand current shared repr to compare against all scenario embeddings
        expanded_current_shared = current_shared_repr.unsqueeze(0).expand(self.num_scenarios, -1, -1) 
        # expanded_current_shared: [num_scenarios, batch_size, input_dim]
        
        # Expand scenario embeddings to match batch size
        expanded_scenario_embs = all_scenario_embeddings.unsqueeze(1).expand(-1, batch_size, -1) 
        # expanded_scenario_embs: [num_scenarios, batch_size, input_dim]

        # Concatenate for attention scoring
        concat_for_attention = torch.cat([expanded_current_shared, expanded_scenario_embs], dim=-1) 
        # concat_for_attention: [num_scenarios, batch_size, input_dim * 2]
        
        # Compute attention scores/logits
        attention_logits = self.attention_net(concat_for_attention).squeeze(-1) 
        # attention_logits: [num_scenarios, batch_size]
        
        # Softmax over scenarios for each sample
        attention_weights = torch.softmax(attention_logits.transpose(0, 1), dim=-1) 
        # attention_weights: [batch_size, num_scenarios]

        # Weighted sum of representations from all "potential" scenarios
        # In practice, shared_reprs and specific_reprs will mostly be zero except for current_scenario_id,
        # so this simplifies. But we follow the general structure.
        # We need to aggregate across all scenarios based on attention weights.
        # We'll aggregate the provided representations (which might be padded).
        weighted_sum_shared = torch.zeros_like(current_shared_repr)
        weighted_sum_specific = torch.zeros_like(current_specific_repr)
        
        for i in range(self.num_scenarios):
             # Get weight for scenario i for each sample in the batch
             weight = attention_weights[:, i].unsqueeze(-1)  # [batch_size, 1]
             # Add weighted contribution
             weighted_sum_shared += weight * shared_reprs[i]
             weighted_sum_specific += weight * specific_reprs[i]

        # Combine the original and enhanced representations
        # Common pattern: concatenate and then project back to original dim
        enhanced_repr = torch.cat([
            current_shared_repr, weighted_sum_shared, 
            current_specific_repr, weighted_sum_specific
        ], dim=-1)
        # enhanced_repr: [batch_size, input_dim * 4]
        
        # Project back to original dimension
        projector = torch.nn.Linear(enhanced_repr.size(-1), current_shared_repr.size(-1))
        projector = projector.to(enhanced_repr.device)
        # Initialize projector if needed? Not done here for brevity, but could be.
        final_enhanced_repr = projector(enhanced_repr)
        # final_enhanced_repr: [batch_size, input_dim]
        
        return final_enhanced_repr

class ScenarioExtractionLayer(torch.nn.Module):
    """
    Scenario Extraction Layer: Combines shared and specific experts with SAN using metaspore interface.
    Processes batches that may contain samples from different scenarios efficiently.
    """
    def __init__(self, input_dim, hidden_dim, num_scenarios, num_sub_experts=4, sparse_init_var=1e-2):
        super().__init__()
        self.num_scenarios = num_scenarios
        self.shared_expert = ScenarioSharedExpert(input_dim, hidden_dim, num_sub_experts, sparse_init_var)
        # One specific expert per scenario
        self.specific_experts = torch.nn.ModuleList([
            ScenarioSpecificExpert(input_dim, hidden_dim, sparse_init_var) for _ in range(num_scenarios)
        ])
        self.san = ScenarioAwareAttention(input_dim, num_scenarios, sparse_init_var)

    def forward(self, x, scenario_ids):
        """
        Process batch with potentially mixed scenarios.
        x: [batch_size, input_dim] tensor of shared embeddings.
        scenario_ids: [batch_size] tensor of long integers indicating scenario for each sample.
        Returns:
            scenario_processed: [batch_size, input_dim] tensor of scenario-enhanced features.
        """
        # Efficiently handle mixed scenarios in a batch
        scenario_outputs = []
        unique_scenarios = torch.unique(scenario_ids)
        
        # Process samples for each unique scenario separately
        for s_id in unique_scenarios:
            mask = (scenario_ids == s_id)
            masked_input = x[mask] # Extract samples belonging to scenario s_id
            
            # Apply shared and specific experts to these samples
            shared_out = self.shared_expert(masked_input)
            specific_out = self.specific_experts[s_id.item()](masked_input)
            
            # Prepare dummy representations for all scenarios (zero-padded)
            # This is required for the SAN which expects lists of reps for all scenarios
            # Even though only the current one matters, others are zero.
            all_shared_reprs = [
                shared_out if i == s_id.item() else torch.zeros_like(shared_out) 
                for i in range(self.num_scenarios)
            ]
            all_specific_reprs = [
                specific_out if i == s_id.item() else torch.zeros_like(specific_out) 
                for i in range(self.num_scenarios)
            ]
            
            # Apply SAN to enhance the representation for this group of samples
            enhanced = self.san(all_shared_reprs, all_specific_reprs, s_id.item())
            
            # Store the result for this scenario group
            scenario_outputs.append(enhanced)
        
        # Reconstruct the full batch output tensor in the original order
        # We know how many samples were in each group from the masks
        scenario_processed = torch.zeros_like(x) # Pre-allocate output tensor
        idx = 0 # Index into scenario_outputs list
        for s_id in unique_scenarios:
            mask = (scenario_ids == s_id)
            num_samples_in_group = mask.sum().item()
            # Place the processed samples back into their correct positions
            scenario_processed[mask] = scenario_outputs[idx][:num_samples_in_group]
            idx += 1
        
        return scenario_processed

class CustomizedGateControl(torch.nn.Module):
    """
    Customized Gate Control (CGC) for Task Extraction Layer with metaspore interface.
    Assumes each sample belongs to exactly one scenario but can belong to multiple tasks.
    Calculates features for ALL tasks for each sample, regardless of task membership.
    """
    def __init__(self, input_dim, task_expert_dims, num_tasks, shared_expert_dim=None, sparse_init_var=1e-2):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_expert_dims = task_expert_dims
        self.shared_expert_dim = shared_expert_dim or input_dim

        # Shared experts (usually just one or a few)
        self.shared_experts = torch.nn.ModuleList([
            MLPLayer(
                input_dim=input_dim,
                output_dim=input_dim,
                hidden_units=[self.shared_expert_dim],
                hidden_activations='ReLU',
                final_activation=None
            )
        ])
        
        # Initialize shared experts weights if initializer attribute exists
        for expert in self.shared_experts:
            if hasattr(expert, 'initializer'):
                expert.initializer = ms.NormalTensorInitializer(var=sparse_init_var)

        # Task-specific experts: one list of experts per task
        self.task_experts = torch.nn.ModuleList([
            torch.nn.ModuleList([
                MLPLayer(
                    input_dim=input_dim,
                    output_dim=input_dim,
                    hidden_units=[task_dim],
                    hidden_activations='ReLU',
                    final_activation=None
                ) for _ in range(2) # Using 2 task-specific experts per task
            ]) for task_dim in task_expert_dims
        ])
        
        # Initialize task experts weights if initializer attribute exists
        for task_exps in self.task_experts:
            for exp in task_exps:
                if hasattr(exp, 'initializer'):
                    exp.initializer = ms.NormalTensorInitializer(var=sparse_init_var)

        # Gates for each task: determines how much shared vs task-specific experts contribute
        # Output size is 1 (shared) + 2 (task-specific) = 3 weights per task gate
        self.task_gates = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1 + 2) # 1 shared + 2 task-specific experts
            for _ in range(num_tasks)
        ])
        
        # Initialize gate weights
        for gate in self.task_gates:
            gate.initializer = ms.NormalTensorInitializer(var=sparse_init_var)

    def forward(self, x):
        """
        Process batch where each sample belongs to exactly one scenario but can belong to multiple tasks.
        Calculates features for ALL tasks for each sample.
        x: [batch_size, input_dim] tensor of scenario-processed features.
        Returns:
            task_processed: [batch_size, num_tasks, input_dim] tensor of task-enhanced features for ALL tasks.
        """
        # Compute outputs from shared experts once for the whole batch
        shared_outputs = [expert(x) for expert in self.shared_experts] # List of [batch_size, input_dim]
        
        # Compute outputs from task-specific experts once for the whole batch (per task/expert)
        task_specific_outputs = [
            [exp(x) for exp in self.task_experts[tid]] # For each task tid, compute both experts
            for tid in range(self.num_tasks)
        ] # Result: List[List[Tensor]]

        all_task_outputs = []
        for t_id in range(self.num_tasks):
            # Compute gating weights for task t_id using the shared input x
            gate_input = x # Use the full batch input for gate calculation
            gate_weights = torch.softmax(self.task_gates[t_id](gate_input), dim=-1) # [batch_size, 3]
            
            # Combine shared and task-specific contributions for task t_id
            shared_contrib = gate_weights[:, 0:1] * shared_outputs[0] # [batch_size, input_dim]
            task_contrib1 = gate_weights[:, 1:2] * task_specific_outputs[t_id][0] # [batch_size, input_dim]
            task_contrib2 = gate_weights[:, 2:3] * task_specific_outputs[t_id][1] # [batch_size, input_dim]
            
            task_output = shared_contrib + task_contrib1 + task_contrib2 # [batch_size, input_dim]
            all_task_outputs.append(task_output)

        # Stack outputs along a new dimension to get [batch_size, num_tasks, input_dim]
        stacked_outputs = torch.stack(all_task_outputs, dim=1) # [batch_size, num_tasks, input_dim]

        return stacked_outputs # [batch_size, num_tasks, input_dim]

class TaskExtractionLayer(torch.nn.Module):
    """
    Task Extraction Layer using CGC.
    Calculates features for ALL tasks for each sample.
    """
    def __init__(self, input_dim, task_expert_dims, num_tasks, sparse_init_var=1e-2):
        super().__init__()
        self.cgc = CustomizedGateControl(input_dim, task_expert_dims, num_tasks, sparse_init_var=sparse_init_var)

    def forward(self, x):
        """
        x: [batch_size, input_dim] tensor of scenario-processed features.
        Returns:
            task_processed: [batch_size, num_tasks, input_dim] tensor of task-enhanced features for ALL tasks.
        """
        return self.cgc(x)

class HiNet(torch.nn.Module):
    """
    Hierarchical information extraction Network (HiNet) with metaspore interface.
    Assumes each sample belongs to exactly one domain_id (scenario) but can belong to N task_ids.
    Forward calculates outputs for ALL tasks. Loss masking happens externally.
    """
    def __init__(self, 
                 domain_count=6,           # Number of scenarios (domains)
                 task_count=2,               # Number of tasks
                 embedding_dim=8,           # Dimension of feature embeddings
                 task_towers_hidden_units=[128, 64],    # Hidden units for task tower networks
                 scenario_expert_hidden_dim=32,     # Hidden dim for scenario experts (shared & specific)
                 task_expert_dims=None,             # List of hidden dims for task experts (one per task)
                 num_sub_experts=2,                 # Number of sub-experts in SEI
                 sparse_init_var=1e-2,              # Variance for sparse initializers
                 scale_factor=1.0,                  # Scale factor (might be used in gates, kept for compatibility)
                 batch_norm=False,                  # Whether to use batch norm (in towers)
                 net_dropout=None,                  # Dropout rate (in towers)
                 ftrl_l1=1.0,                       # FTRL L1 reg parameter
                 ftrl_l2=120.0,                     # FTRL L2 reg parameter
                 ftrl_alpha=0.5,                    # FTRL alpha parameter
                 ftrl_beta=1.0,                     # FTRL beta parameter
                 column_name_path=None,             # Path for feature column names
                 combine_schema_path=None):         # Path for feature combination schema
        super().__init__()
        
        self.domain_count = domain_count
        self.task_count = task_count
        self.embedding_dim = embedding_dim
        self.scenario_expert_hidden_dim = scenario_expert_hidden_dim
        self.task_expert_dims = task_expert_dims or [32] * task_count # Default to 64 for each task
        self.scale_factor = scale_factor
        
        # Shared embedding layer: aggregates raw sparse features into dense vectors
        self.shared_embedding = ms.EmbeddingSumConcat(
            embedding_dim, 
            column_name_path, 
            combine_schema_path
        )
        # Configure embedding updater and initializer
        self.shared_embedding.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1, 
            l2=ftrl_l2, 
            alpha=ftrl_alpha, 
            beta=ftrl_beta
        )
        self.shared_embedding.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
        # Calculate input dimension after embeddings
        if hasattr(self.shared_embedding, 'feature_count'):
             input_dim = int(self.shared_embedding.feature_count * embedding_dim)
        else:
             raise AttributeError("shared_embedding.feature_count is missing. Please check schema definition.")
        
        # Shared normalization layer (BatchNorm typically)
        self.shared_bn = ms.nn.Normalization(input_dim)
        
        # Scenario Extraction Layer: learns scenario-specific representations
        self.scenario_extraction = ScenarioExtractionLayer(
            input_dim=input_dim,
            hidden_dim=scenario_expert_hidden_dim,
            num_scenarios=domain_count,
            num_sub_experts=num_sub_experts,
            sparse_init_var=sparse_init_var
        )
        
        # Task Extraction Layer: learns task-specific representations for ALL tasks
        self.task_extraction = TaskExtractionLayer(
            input_dim=input_dim,
            task_expert_dims=self.task_expert_dims,
            num_tasks=task_count,
            sparse_init_var=sparse_init_var
        )
        
        # Task-specific towers: map final task representations to prediction scores
        self.task_towers = torch.nn.ModuleList()
        for _ in range(task_count):
            tower = MLPLayer(
                input_dim=input_dim,  # Input dim matches processed feature dim
                output_dim=1,         # Single output (logit) per sample per task
                hidden_units=task_towers_hidden_units,
                hidden_activations='ReLU',
                final_activation=None, # Sigmoid applied later
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            )
            if hasattr(tower, 'initializer'):
                tower.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
            self.task_towers.append(tower)
        
        # Final activation function applied to tower outputs
        self.final_activation = torch.nn.Sigmoid()
    
    def forward(self, x):
        """
        Defines the forward pass of the HiNet model.
        Calculates predictions for ALL tasks for each sample.
        Assumes do_extra_work was called beforehand to store scenario_ids.
        x: Input tensor (likely sparse indices passed to embedding layer).
        Returns:
            task_predictions: List of [batch_size, 1] tensors, one for each task.
        """
        # Retrieve scenario IDs set by do_extra_work
        scenario_ids = getattr(self, '_current_scenario_ids', None)
        # Note: task_multihot is NOT needed in forward for this implementation
        
        if scenario_ids is None:
            raise ValueError("Scenario IDs must be set via do_extra_work before forward pass")
        
        # Step 1: Shared Embedding and Normalization
        shared_emb = self.shared_embedding(x) # Dense representation [batch_size, input_dim]
        shared_emb_bn = self.shared_bn(shared_emb) # Normalized [batch_size, input_dim]
        
        # Step 2: Scenario-level Feature Enhancement
        scenario_processed = self.scenario_extraction(shared_emb_bn, scenario_ids) 
        # Enhanced features [batch_size, input_dim]
        
        # Step 3: Task-level Feature Enhancement (Calculates features for ALL tasks)
        task_processed = self.task_extraction(scenario_processed) 
        # Enhanced features [batch_size, num_tasks, input_dim]
        
        # Step 4: Task-specific Predictions (Calculates predictions for ALL tasks)
        task_outputs_list = []
        for i in range(self.task_count):
            # Select the features specific to task i for the whole batch
            # task_processed[:, i, :] -> [batch_size, input_dim]
            task_specific_features = task_processed[:, i, :] # [batch_size, input_dim]
            
            # Pass these features through the corresponding task tower
            task_output = self.task_towers[i](task_specific_features) # [batch_size, 1]
            
            # Apply final activation (e.g., Sigmoid for probability)
            task_prediction = self.final_activation(task_output) # [batch_size, 1]
            
            # Append the prediction for task i to a list
            task_outputs_list.append(task_prediction) # List of [batch_size, 1] tensors
        
        # Concatenate the list of [batch_size, 1] tensors along the task dimension
        task_predictions_tensor = torch.cat(task_outputs_list, dim=1) # [batch_size, num_tasks]
        
        # Return a single tensor containing predictions for all tasks
        return task_predictions_tensor # [batch_size, num_tasks]

    def do_extra_work(self, minibatch):
        """
        Extracts scenario_id and task_multihot matrix from the minibatch dictionary/dataframe.
        Stores them internally for potential use in loss calculation or other external logic.
        minibatch: Dictionary or dataframe-like object containing batch data.
        """
        # Extract scenario_id (domain_id alias supported)
        if isinstance(minibatch, dict):
            scenario_id_col = minibatch.get('scenario_id', minibatch.get('domain_id'))
        else:
            scenario_id_col = minibatch.get('scenario_id', minibatch.get('domain_id')).values
        
        if not isinstance(scenario_id_col, torch.Tensor):
            scenario_id_tensor = torch.tensor(scenario_id_col, dtype=torch.long)
        else:
            scenario_id_tensor = scenario_id_col.long()
        
        # Store extracted IDs/matrix in model state
        self._current_scenario_ids = scenario_id_tensor

    def predict(self, y_hat, minibatch):
        """
        y_hat: tensor of shape [batch_size, num_tasks]
        Returns predictions for the first task: [batch_size, 1] tensor.
        """
        return y_hat[:, :1]  # keep dimension

    def compute_loss(self, predictions, labels, minibatch):
        output = predictions
        # Main prediction loss
        main_loss = F.binary_cross_entropy(output, labels, reduction='mean')

        return main_loss, main_loss
