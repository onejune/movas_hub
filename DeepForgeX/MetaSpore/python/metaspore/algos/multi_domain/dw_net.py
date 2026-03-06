import torch
import torch.nn as nn
import metaspore as ms
from ..layers import MLPLayer 

class SACNModel(nn.Module):
    """
    SACN (Scenario-Aware Cross Network) Model Implementation
    """
    def __init__(
        self,
        domain_count=1,
        embedding_dim=8,
        # --- Feature Columns ---
        column_name_path=None,
        combine_schema_path=None,
        # --- Cross Network ---
        cross_layers=3,
        # --- Final MLP ---
        final_hidden_units=[64],
        final_net_activation="sigmoid",
        # --- Network Config ---
        hidden_activations="ReLU",
        net_dropout=0,
        batch_norm=False,
        use_bias=True,
        # --- Embedding Config ---
        embedding_initializer=None,
        embedding_init_var=0.01,
        embedding_regularizer=None,
        net_regularizer=None,
        ftrl_l1=1.0,
        ftrl_l2=120.0,
        ftrl_alpha=0.5,
        ftrl_beta=1.0,
        # --- Domain Embedding ---
        domain_embedding_initializer=None,
        domain_embedding_init_var=0.01,
        domain_embedding_regularizer=None,
        **kwargs
    ):
        super(SACNModel, self).__init__()

        self.domain_count = domain_count
        self.sparse_embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path

        # Shared Sparse Embedding
        self.shared_sparse_embedding = ms.EmbeddingSumConcat(
            self.sparse_embedding_dim,
            self.column_name_path,
            self.combine_schema_path
        )
        if ftrl_l1 > 0 or ftrl_l2 > 0:
            self.shared_sparse_embedding.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        if embedding_initializer is None or embedding_initializer.lower() == "normal":
            self.shared_sparse_embedding.initializer = ms.NormalTensorInitializer(var=embedding_init_var)

        # Domain Embedding
        self.domain_embedding = nn.Embedding(num_embeddings=self.domain_count, embedding_dim=self.sparse_embedding_dim)

        # Calculate input dimension for networks
        input_dim_for_nets = self.shared_sparse_embedding.feature_count * self.sparse_embedding_dim
        # Total input dimension for cross network includes domain embedding
        total_input_dim = input_dim_for_nets + self.sparse_embedding_dim

        # Cross Network Layers
        self.cross_layers = nn.ModuleList()
        self.cross_biases = nn.ParameterList()
        # Note: Removed unused cross_outputs and output_layer definition
        for i in range(cross_layers):
            # Each cross layer learns a 1-dim projection for the current output to weight the original input x0
            cross_layer = nn.Linear(total_input_dim, 1, bias=False)
            bias = nn.Parameter(torch.zeros(total_input_dim))
            # output_layer = nn.Linear(total_input_dim, total_input_dim, bias=False) # <--- REMOVED
            self.cross_layers.append(cross_layer)
            self.cross_biases.append(bias)
            # self.cross_outputs.append(output_layer) # <--- REMOVED

        # Final MLP after Cross Network
        final_mlp_input_dim = total_input_dim # Output of cross network is same as its input
        self.final_mlp = MLPLayer(
            input_dim=final_mlp_input_dim,
            output_dim=1,
            hidden_units=final_hidden_units,
            hidden_activations=hidden_activations,
            final_activation=None, # No final activation before sigmoid
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
            input_norm=False,
        )

        # Final Activation Layer
        self.final_activation_layer = nn.Sigmoid() if final_net_activation == "sigmoid" else None

        # Store domain IDs during forward pass
        self._current_domain_ids = None

    def do_extra_work(self, minibatch):
        if isinstance(minibatch, dict):
            domain_id_col = minibatch.get('domain_id')
        elif hasattr(minibatch, 'domain_id'):
            domain_id_col = minibatch.domain_id
        else:
            raise ValueError("Input batch must contain 'domain_id' column.")
        
        if not isinstance(domain_id_col, torch.Tensor):
            domain_id_tensor = torch.tensor(domain_id_col, dtype=torch.long)
        else:
            domain_id_tensor = domain_id_col.long()
        self._current_domain_ids = domain_id_tensor

    def forward(self, x):
        domain_ids = self._current_domain_ids

        # Shared Embedding Features
        embedded_features = self.shared_sparse_embedding(x) # Shape: (B, F * E)

        # Domain Embedding Features
        domain_embedded = self.domain_embedding(domain_ids) # Shape: (B, E)

        # Concatenate shared features and domain features
        x_total = torch.cat([embedded_features, domain_embedded], dim=1) # Shape: (B, (F * E) + E)
        x0 = x_total # Original input for cross connections

        # Cross Network
        x_cross = x_total
        # Note: Removed output_layer from zip and loop body
        for i, (cross_layer, bias) in enumerate(zip(self.cross_layers, self.cross_biases)):
            # Calculate cross term: (W_l^T * x_l) * x_0 + b_l
            cross_proj = cross_layer(x_cross).unsqueeze(-1) # Shape: (B, 1, 1)
            cross_term = (x0.unsqueeze(1) * cross_proj).squeeze(1) # Broadcast x0 * (proj reshaped to (B, 1, 1)) -> (B, (F*E)+E, 1) -> (B, (F*E)+E)
            x_cross = cross_term + bias + x_cross # Residual connection

        # Final MLP
        logits = self.final_mlp(x_cross)

        # Apply final activation
        output = self.final_activation_layer(logits) if self.final_activation_layer else logits
        return output


class SAINetModel(nn.Module):
    """
    SAINet (Scenario-Aware Interest Network) Model Implementation
    """
    def __init__(
        self,
        domain_count=1,
        embedding_dim=8,
        # --- Feature Columns ---
        column_name_path=None,
        combine_schema_path=None,
        # --- Interest Network (Simulated with Attention) ---
        interest_attention_dim=64,
        # --- Final MLP ---
        final_hidden_units=[64],
        final_net_activation="sigmoid",
        # --- Network Config ---
        hidden_activations="ReLU",
        net_dropout=0,
        batch_norm=False,
        use_bias=True,
        # --- Embedding Config ---
        embedding_initializer=None,
        embedding_init_var=0.01,
        embedding_regularizer=None,
        net_regularizer=None,
        ftrl_l1=1.0,
        ftrl_l2=120.0,
        ftrl_alpha=0.5,
        ftrl_beta=1.0,
        domain_embedding_initializer=None,
        domain_embedding_init_var=0.01,
        domain_embedding_regularizer=None,
        **kwargs
    ):
        super(SAINetModel, self).__init__()

        self.domain_count = domain_count
        self.sparse_embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path

        # Shared Sparse Embedding
        self.shared_sparse_embedding = ms.EmbeddingSumConcat(
            self.sparse_embedding_dim,
            self.column_name_path,
            self.combine_schema_path
        )
        if ftrl_l1 > 0 or ftrl_l2 > 0:
            self.shared_sparse_embedding.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        if embedding_initializer is None or embedding_initializer.lower() == "normal":
            self.shared_sparse_embedding.initializer = ms.NormalTensorInitializer(var=embedding_init_var)

        # Domain Embedding
        self.domain_embedding = nn.Embedding(num_embeddings=self.domain_count, embedding_dim=self.sparse_embedding_dim)

        # Calculate dimensions
        self.total_feature_dim = self.shared_sparse_embedding.feature_count * self.sparse_embedding_dim
        
        # CORRECTED: Calculate the attention input dimension correctly.
        # It should be [embedded_features (F*E), repeated_domain_embedded_flattened (F*E)] -> concatenated -> (2 * F * E)
        # Original incorrect calculation was: attention_input_dim = self.total_feature_dim + self.sparse_embedding_dim # (B, F*E + E)
        attention_input_dim = self.total_feature_dim + self.total_feature_dim # (B, F*E + F*E)

        # Attention Network for Domain-Aware Interest
        self.attention_mlp = MLPLayer(
            input_dim=attention_input_dim, # Now correctly set to 2 * F * E
            output_dim=interest_attention_dim,
            hidden_units=[128], # Can make this configurable if needed
            hidden_activations=hidden_activations,
            final_activation=hidden_activations,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
            input_norm=False,
        )
        self.attention_output = nn.Linear(interest_attention_dim, self.total_feature_dim, bias=True) # Outputs weights for each feature dim

        # Final MLP after applying domain-aware attention
        final_mlp_input_dim = self.total_feature_dim + self.sparse_embedding_dim # Include domain emb in final concat
        self.final_mlp = MLPLayer(
            input_dim=final_mlp_input_dim,
            output_dim=1,
            hidden_units=final_hidden_units,
            hidden_activations=hidden_activations,
            final_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
            input_norm=False,
        )

        # Final Activation Layer
        self.final_activation_layer = nn.Sigmoid() if final_net_activation == "sigmoid" else None

        # Store domain IDs during forward pass
        self._current_domain_ids = None

    def do_extra_work(self, minibatch):
        if isinstance(minibatch, dict):
            domain_id_col = minibatch.get('domain_id')
        elif hasattr(minibatch, 'domain_id'):
            domain_id_col = minibatch.domain_id
        else:
            raise ValueError("Input batch must contain 'domain_id' column.")
        
        if not isinstance(domain_id_col, torch.Tensor):
            domain_id_tensor = torch.tensor(domain_id_col, dtype=torch.long)
        else:
            domain_id_tensor = domain_id_col.long()
        self._current_domain_ids = domain_id_tensor

    def forward(self, x):
        domain_ids = self._current_domain_ids

        # Shared Embedding Features
        embedded_features = self.shared_sparse_embedding(x) # Shape: (B, F * E)
        batch_size = embedded_features.shape[0] # Get batch size from the actual embedded features

        # Domain Embedding Features
        domain_embedded = self.domain_embedding(domain_ids) # Shape: (B, E)

        # Expand domain embedding to match feature dims for attention calculation
        domain_embedded_expanded = domain_embedded.unsqueeze(1).expand(-1, self.total_feature_dim // self.sparse_embedding_dim, -1) # Shape: (B, F, E)
        domain_embedded_flat = domain_embedded_expanded.reshape(batch_size, -1) # Shape: (B, F * E)

        # Concatenate for attention input - This creates the (B, 2 * F * E) tensor
        attention_input = torch.cat([embedded_features, domain_embedded_flat], dim=1) # Shape: (B, F*E + F*E)

        # Calculate domain-aware attention weights
        # Now attention_input (shape B, 2*F*E) matches the expected input_dim of attention_mlp
        attention_hidden = self.attention_mlp(attention_input) # Shape: (B, interest_dim)
        attention_weights_raw = self.attention_output(attention_hidden) # Shape: (B, F * E)
        attention_weights = torch.softmax(attention_weights_raw, dim=1) # Softmax over feature dimensions

        # Apply attention weights to embedded features (weighted sum simulation for interest)
        weighted_interest_features = embedded_features * attention_weights # Shape: (B, F * E)

        # Concatenate weighted interests and domain embedding
        final_input = torch.cat([weighted_interest_features, domain_embedded], dim=1) # Shape: (B, F*E + E)

        # Final MLP
        logits = self.final_mlp(final_input)

        # Apply final activation
        output = self.final_activation_layer(logits) if self.final_activation_layer else logits
        return output


class DSWINModel(nn.Module):
    """
    DSWIN (Dynamic Scenario Weighted Interest Network) Model Implementation
    """
    def __init__(
        self,
        domain_count=1,
        embedding_dim=8,
        column_name_path=None,
        combine_schema_path=None,
        domain_branch_hidden_units=[512, 128],
        dynamic_weight_hidden_units=[64],
        final_net_activation="sigmoid",
        hidden_activations="ReLU",
        net_dropout=0,
        batch_norm=False,
        use_bias=True,
        embedding_initializer=None,
        embedding_init_var=0.01,
        embedding_regularizer=None,
        net_regularizer=None,
        ftrl_l1=1.0,
        ftrl_l2=120.0,
        ftrl_alpha=0.5,
        ftrl_beta=1.0,
        domain_embedding_initializer=None,
        domain_embedding_init_var=0.01,
        domain_embedding_regularizer=None,
        **kwargs
    ):
        super(DSWINModel, self).__init__()

        self.domain_count = domain_count
        self.sparse_embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path

        # Shared Sparse Embedding
        self.shared_sparse_embedding = ms.EmbeddingSumConcat(
            self.sparse_embedding_dim,
            self.column_name_path,
            self.combine_schema_path
        )
        if ftrl_l1 > 0 or ftrl_l2 > 0:
            self.shared_sparse_embedding.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        if embedding_initializer is None or embedding_initializer.lower() == "normal":
            self.shared_sparse_embedding.initializer = ms.NormalTensorInitializer(var=embedding_init_var)

        # Domain Embedding
        self.domain_embedding = nn.Embedding(num_embeddings=self.domain_count, embedding_dim=self.sparse_embedding_dim)
        
        # Calculate input dimension
        input_dim_for_branches = self.shared_sparse_embedding.feature_count * self.sparse_embedding_dim + self.sparse_embedding_dim # Includes domain emb

        # Per-Domain Branches (MLP for each domain)
        self.domain_branches = nn.ModuleDict()
        for d in range(self.domain_count):
            branch = MLPLayer(
                input_dim=input_dim_for_branches,
                output_dim=1, # Each branch outputs a single logit
                hidden_units=domain_branch_hidden_units,
                hidden_activations=hidden_activations,
                final_activation=None, # No final activation, will apply sigmoid later
                dropout_rates=net_dropout,
                batch_norm=batch_norm,
                use_bias=use_bias,
                input_norm=False,
            )
            self.domain_branches[f'domain_{d}'] = branch

        # Dynamic Weight Network: calculates weights for each domain branch based on domain embedding
        # Input: domain embedding
        self.dynamic_weight_mlp = MLPLayer(
            input_dim=self.sparse_embedding_dim,
            output_dim=self.domain_count, # Outputs logits for domain_count
            hidden_units=dynamic_weight_hidden_units,
            hidden_activations=hidden_activations,
            final_activation=hidden_activations,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
            input_norm=False,
        )
        self.dynamic_weight_output = nn.Linear(self.domain_count, self.domain_count, bias=True) # Final projection for weights
        self.weight_softmax = nn.Softmax(dim=1) # Softmax over domain dimension

        # Final Activation Layer
        self.final_activation_layer = nn.Sigmoid() if final_net_activation == "sigmoid" else None

        # Store domain IDs during forward pass
        self._current_domain_ids = None

    def do_extra_work(self, minibatch):
        if isinstance(minibatch, dict):
            domain_id_col = minibatch.get('domain_id')
        elif hasattr(minibatch, 'domain_id'):
            domain_id_col = minibatch.domain_id
        else:
            raise ValueError("Input batch must contain 'domain_id' column.")
        
        if not isinstance(domain_id_col, torch.Tensor):
            domain_id_tensor = torch.tensor(domain_id_col, dtype=torch.long)
        else:
            domain_id_tensor = domain_id_col.long()
        self._current_domain_ids = domain_id_tensor

    def forward(self, x):
        domain_ids = self._current_domain_ids

        # Shared Embedding Features
        embedded_features = self.shared_sparse_embedding(x) # Shape: (B, F * E)
        batch_size = embedded_features.shape[0] # Get batch size from the actual embedded features

        # Domain Embedding Features
        domain_embedded = self.domain_embedding(domain_ids) # Shape: (B, E)

        # Concatenate for branch input
        branch_input = torch.cat([embedded_features, domain_embedded], dim=1) # Shape: (B, F*E + E)

        # Calculate outputs for all domain branches
        branch_outputs = []
        for d in range(self.domain_count):
            branch_output = self.domain_branches[f'domain_{d}'](branch_input) # Shape: (B, 1)
            branch_outputs.append(branch_output)
        all_branch_outputs = torch.cat(branch_outputs, dim=1) # Shape: (B, domain_count)

        # Calculate dynamic weights using domain embedding
        weight_hidden = self.dynamic_weight_mlp(domain_embedded) # Shape: (B, dyn_hid_dim)
        weight_logits = self.dynamic_weight_output(weight_hidden) # Shape: (B, domain_count)
        dynamic_weights = self.weight_softmax(weight_logits) # Shape: (B, domain_count)

        # Calculate weighted sum of branch outputs
        # Multiply outputs by weights and sum along domain dimension
        weighted_sum_logits = torch.sum(all_branch_outputs * dynamic_weights, dim=1, keepdim=True) # Shape: (B, 1)

        # Apply final activation
        output = self.final_activation_layer(weighted_sum_logits) if self.final_activation_layer else weighted_sum_logits
        return output.squeeze(-1) # Return shape (B,) if needed, otherwise keep (B, 1)
