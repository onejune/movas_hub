import torch
import torch.nn as nn
import metaspore as ms
from ..layers import MLPLayer

class GateNULayer(torch.nn.Module):
    """GateNU层，带缩放因子的门控机制"""
    def __init__(self, input_dim, scale_factor=1.0):
        super(GateNULayer, self).__init__()
        self.scale_factor = scale_factor
        self.gate = torch.nn.Linear(input_dim, input_dim)

    def forward(self, x):
        gate_logits = self.gate(x)
        scaled_logits = gate_logits * self.scale_factor
        gate_weights = torch.sigmoid(scaled_logits)
        return x * gate_weights

class StarGateModel(torch.nn.Module):
    def __init__(self,
                 num_domains=1,
                 embedding_dim=8,
                 center_hidden_units=[1024, 512, 256],
                 domain_specific_hidden_units=[256,128],
                 final_hidden_units=[64],
                 gate_scale_factor=1.0, # 新增：GateNULayer 的缩放因子
                 column_name_path=None,
                 combine_schema_path=None,
                 hidden_activations="ReLU",
                 net_dropout=0.1,
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
                 final_net_activation="sigmoid",
                 **kwargs):
        super(StarGateModel, self).__init__()
        self._current_domain_ids = None  # 用于暂存当前 batch 的 domain_id
        self.num_domains = num_domains
        self.sparse_embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path
        self.gate_scale_factor = gate_scale_factor # Store the scale factor

        self.enable_hidden_gate = False

        self.shared_sparse_embedding = ms.EmbeddingSumConcat(
            self.sparse_embedding_dim,
            column_name_path,
            combine_schema_path
        )
        if ftrl_l1 > 0 or ftrl_l2 > 0:
             self.shared_sparse_embedding.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        if embedding_initializer is None or embedding_initializer.lower() == "normal":
            self.shared_sparse_embedding.initializer = ms.NormalTensorInitializer(var=embedding_init_var)

        self.domain_embedding = nn.Embedding(num_embeddings=self.num_domains, embedding_dim=self.sparse_embedding_dim)

        # Get input dimension directly from embedding layer
        input_dim_for_nets = self.shared_sparse_embedding.feature_count * self.sparse_embedding_dim
        
        # Build layers immediately
        final_activation = None
        center_output_dim = center_hidden_units[-1] if center_hidden_units else input_dim_for_nets
        
        # --- GateNULayer for initial embedded features ---
        # Add a GateNULayer that operates on the raw embedded features
        self.initial_gate_layer = GateNULayer(input_dim_for_nets, scale_factor=self.gate_scale_factor)
        
        if self.enable_hidden_gate:
            # --- Modified Center Net Construction ---
            # Instead of using MLPLayer directly, we manually build it with GateNULayer
            center_layers = []
            current_dim = input_dim_for_nets # This remains the *input* dim for center_net
            for i, hidden_dim in enumerate(center_hidden_units):
                # 1. Linear
                center_layers.append(nn.Linear(current_dim, hidden_dim))
                # 2. Activation
                center_layers.append(nn.ReLU())
                # 3. GateNU (applied after activation, acting on the activated features)
                # center_layers.append(GateNULayer(hidden_dim, scale_factor=self.gate_scale_factor))
                current_dim = hidden_dim # Update for the next iteration
            self.center_net = nn.Sequential(*center_layers)
        else:
            center_output_dim = center_hidden_units[-1] if center_hidden_units else input_dim_for_nets
            self.center_net = MLPLayer(
                input_dim=input_dim_for_nets,
                output_dim=center_output_dim,
                hidden_units=center_hidden_units[:-1] if center_hidden_units else [],
                hidden_activations=hidden_activations,
                final_activation=final_activation,
                dropout_rates=net_dropout,
                batch_norm=batch_norm,
                use_bias=use_bias,
                input_norm=False,
            )

        domain_net_output_dim = domain_specific_hidden_units[-1] if domain_specific_hidden_units else input_dim_for_nets
        self.domain_specific_nets = nn.ModuleList([
            MLPLayer(
                input_dim=input_dim_for_nets,
                output_dim=domain_net_output_dim,
                hidden_units=domain_specific_hidden_units[:-1] if domain_specific_hidden_units else [],
                hidden_activations=hidden_activations,
                final_activation=final_activation,
                dropout_rates=net_dropout,
                batch_norm=False, # Keep domain nets lightweight
                use_bias=use_bias,
                input_norm=False,
            ) for _ in range(self.num_domains)
        ])

        self.gamma_transforms = nn.ModuleList([
            nn.Linear(domain_net_output_dim, center_output_dim) for _ in range(self.num_domains)
        ])
        self.beta_transforms = nn.ModuleList([
            nn.Linear(domain_net_output_dim, center_output_dim) for _ in range(self.num_domains)
        ])

        final_input_dim = center_output_dim
        self.final_mlp = MLPLayer(
            input_dim=final_input_dim,
            output_dim=1,
            hidden_units=final_hidden_units,
            hidden_activations=hidden_activations,
            final_activation=final_activation,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
            input_norm=False,
        )

        if final_net_activation and final_net_activation == 'sigmoid':
            self.final_activation_layer = nn.Sigmoid()
        else:
            self.final_activation_layer = None

    # do_extra_work 保持不变
    def do_extra_work(self, minibatch):
        if isinstance(minibatch, dict):
            domain_id_col = minibatch['domain_id']
        elif hasattr(minibatch, 'domain_id'):
            domain_id_col = minibatch.domain_id
        else:
            domain_id_col = minibatch['domain_id'].values  # 或 .to_numpy()
        if not isinstance(domain_id_col, torch.Tensor):
            domain_id_tensor = torch.tensor(domain_id_col, dtype=torch.long)
        else:
            domain_id_tensor = domain_id_col.long()
        self._current_domain_ids = domain_id_tensor

    def forward(self, x):
        embedded_features = self.shared_sparse_embedding(x)
        
        # --- Apply GateNULayer to the initial embedded features ---
        # This gates the raw concatenated embeddings before they go into center_net or domain_nets
        gated_embedded_features = self.initial_gate_layer(embedded_features)
        
        # Pass the gated features through the center net
        h_center = self.center_net(gated_embedded_features) # Now receives gated features
        batch_size = embedded_features.shape[0] # Original shape for domain net processing

        domain_ids = self._current_domain_ids
        assert domain_ids is not None and domain_ids.shape[0] == batch_size

        # Pass the gated features through the domain-specific nets as well
        domain_net_outputs = []
        for i in range(batch_size):
            d_id = domain_ids[i].item()
            domain_net = self.domain_specific_nets[d_id]
            # Use the gated features here too
            d_out = domain_net(gated_embedded_features[i:i+1, :]) 
            domain_net_outputs.append(d_out)
        h_domain_batch = torch.cat(domain_net_outputs, dim=0)

        gammas = []
        betas = []
        for i in range(batch_size):
            d_id = domain_ids[i].item()
            gamma_transform = self.gamma_transforms[d_id]
            beta_transform = self.beta_transforms[d_id]
            gamma_i = gamma_transform(h_domain_batch[i:i+1, :])
            beta_i = beta_transform(h_domain_batch[i:i+1, :])
            gammas.append(gamma_i)
            betas.append(beta_i)
        gamma_batch = torch.cat(gammas, dim=0)
        beta_batch = torch.cat(betas, dim=0)

        h_fused = gamma_batch * h_center + beta_batch
        final_logits = self.final_mlp(h_fused)
        output = self.final_activation_layer(final_logits) if self.final_activation_layer else final_logits
        return output