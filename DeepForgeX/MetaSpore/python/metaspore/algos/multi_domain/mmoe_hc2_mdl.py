import torch
import torch.nn as nn
import torch.nn.functional as F
import metaspore as ms
from ..layers import MLPLayer
from typing import List, Optional

class HC2MMoEModel(torch.nn.Module):
    """
    HC² Enhanced MMoE for Multi-Domain Modeling.
    Combines MMoE architecture with Hybrid Contrastive Constraints (Generalized + Individual).
    """
    def __init__(self,
                 embedding_dim=10,
                 column_name_path=None,
                 combine_schema_path=None,
                 expert_numb=6,
                 domain_numb=20,
                 expert_hidden_units=[256, 128],
                 expert_out_dim=10,
                 gate_hidden_units=[64],
                 tower_hidden_units=[64],
                 dnn_activations='ReLU',
                 use_bias=True,
                 input_norm=False,
                 batch_norm=False,
                 net_dropout=None,
                 sparse_init_var=1e-2,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 tower_output_dim=1,
                 contrastive_weight_general: float = 0.1,
                 contrastive_weight_individual: float = 0.05,
                 temperature: float = 0.1,
                 diffusion_noise_scale: float = 0.1,
                 inverse_weight_temperature: float = 0.05,
                 dropout_augmentation_prob: float = 0.1,
                 **kwargs):
        super().__init__()
        self.expert_numb = expert_numb
        self.domain_numb = domain_numb
        self.expert_out_dim = expert_out_dim
        self.tower_output_dim = tower_output_dim
        
        # HC² parameters
        self.contrastive_weight_general = contrastive_weight_general
        self.contrastive_weight_individual = contrastive_weight_individual
        self.temperature = temperature
        self.diffusion_noise_scale = diffusion_noise_scale
        self.inverse_weight_temperature = inverse_weight_temperature
        self.dropout_augmentation_prob = dropout_augmentation_prob

        self.embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path
        self.sparse = ms.EmbeddingSumConcat(self.embedding_dim, self.column_name_path, self.combine_schema_path)
        self.sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        self.input_dim = int(self.sparse.feature_count * self.embedding_dim)

        # --- Experts: Shared across all domains ---
        self.experts = torch.nn.ModuleList()
        for i in range(self.expert_numb):
            mlp = MLPLayer(input_dim=self.input_dim,
                           output_dim=self.expert_out_dim,
                           hidden_units=expert_hidden_units,
                           hidden_activations=dnn_activations,
                           final_activation=None,
                           dropout_rates=net_dropout,
                           input_norm=input_norm,
                           batch_norm=batch_norm,
                           use_bias=use_bias)
            self.experts.append(mlp)

        # --- Gates: One per domain ---
        self.gates = torch.nn.ModuleList()
        for i in range(self.domain_numb):
            mlp = MLPLayer(input_dim=self.input_dim,
                           output_dim=self.expert_numb,
                           hidden_units=gate_hidden_units,
                           hidden_activations=dnn_activations,
                           final_activation=None,
                           dropout_rates=net_dropout,
                           input_norm=input_norm,
                           batch_norm=batch_norm,
                           use_bias=use_bias)
            self.gates.append(mlp)
        self.gate_softmax = torch.nn.Softmax(dim=1)

        # --- Towers: One per domain ---
        self.towers = torch.nn.ModuleList()
        for i in range(self.domain_numb):
            mlp = MLPLayer(input_dim=self.expert_out_dim,
                           output_dim=self.tower_output_dim,
                           hidden_units=tower_hidden_units,
                           hidden_activations=dnn_activations,
                           final_activation='Sigmoid',
                           dropout_rates=net_dropout,
                           input_norm=input_norm,
                           batch_norm=batch_norm,
                           use_bias=use_bias)
            self.towers.append(mlp)

        # --- Initialize internal attribute for domain IDs ---
        self._current_domain_ids = None
        
        # Dropout for augmentation in individual contrastive loss
        self.dropout_for_aug = nn.Dropout(p=self.dropout_augmentation_prob)

    def do_extra_work(self, minibatch):
        """Extract and cache domain_id from current batch."""
        if isinstance(minibatch, dict):
            domain_id_col = minibatch['domain_id']
        elif hasattr(minibatch, 'domain_id'):
            domain_id_col = minibatch.domain_id
        else:
            domain_id_col = minibatch['domain_id'].values

        if not isinstance(domain_id_col, torch.Tensor):
            domain_id_tensor = torch.tensor(domain_id_col, dtype=torch.long)
        else:
            domain_id_tensor = domain_id_col.long()

        self._current_domain_ids = domain_id_tensor

    def forward(self, x):
        """
        Forward pass with optional embedding return for contrastive loss.
        Returns: (predictions, expert_outputs, mmoe_output) for contrastive learning
        """
        sparse_features = self.sparse(x)
        batch_size = sparse_features.size(0)

        # Get expert outputs
        expert_outputs = [expert(sparse_features) for expert in self.experts]
        expert_cat = torch.stack(expert_outputs, dim=1)  # [batch, expert_numb, expert_out_dim]

        # Compute all tower outputs
        all_tower_outputs = []
        all_mmoe_outputs = []  # Store the MMoE intermediate representations
        
        for i in range(self.domain_numb):
            gate_out = self.gates[i](sparse_features)
            gate_out = self.gate_softmax(gate_out).unsqueeze(-1)  # [batch, expert_numb, 1]
            mmoe_output = torch.sum(expert_cat * gate_out, dim=1)  # [batch, expert_out_dim]
            tower_out = self.towers[i](mmoe_output)
            all_tower_outputs.append(tower_out)
            all_mmoe_outputs.append(mmoe_output)

        # Select outputs based on current domain IDs
        domain_ids = self._current_domain_ids
        assert domain_ids.size(0) == batch_size
        
        stacked_outputs = torch.stack(all_tower_outputs, dim=1)  # [batch, domain_numb, tower_output_dim]
        expanded_ids = domain_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.tower_output_dim)
        selected_outputs = torch.gather(stacked_outputs, 1, expanded_ids).squeeze(1)
        
        # Also select the corresponding MMoE representations for contrastive learning
        stacked_mmoe = torch.stack(all_mmoe_outputs, dim=1)  # [batch, domain_numb, expert_out_dim]
        expanded_mmoe_ids = domain_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.expert_out_dim)
        selected_mmoe = torch.gather(stacked_mmoe, 1, expanded_mmoe_ids).squeeze(1)
        
        # Stack all expert outputs for generalized contrastive loss
        stacked_experts = torch.stack(expert_outputs, dim=1)  # [batch, expert_numb, expert_out_dim]
        # Take mean across experts to get a single expert representation per sample
        avg_expert_repr = torch.mean(stacked_experts, dim=1)  # [batch, expert_out_dim]
        
        return selected_outputs, avg_expert_repr, selected_mmoe

    def compute_contrastive_losses(self, expert_repr, mmoe_repr, labels, domain_ids):
        """Compute both generalized and individual contrastive losses with enhanced features."""
        # Normalize embeddings for cosine similarity
        expert_repr_norm = F.normalize(expert_repr, p=2, dim=1)
        mmoe_repr_norm = F.normalize(mmoe_repr, p=2, dim=1)
        
        # Generalized Contrastive Loss (based on shared expert representations and labels)
        gen_loss = self._compute_generalized_contrastive_loss_enhanced(expert_repr_norm, labels, domain_ids)
        
        # Individual Contrastive Loss (based on domain-specific MMoE representations and domains)
        ind_loss = self._compute_individual_contrastive_loss_enhanced(mmoe_repr_norm, domain_ids)
        
        return gen_loss, ind_loss

    def _compute_generalized_contrastive_loss_enhanced(self, h_norm, labels, domain_ids):
        """
        Enhanced Generalized Contrastive Loss with:
        1. Diffusion noise for negative samples
        2. Inverse similarity weighting
        """
        batch_size = h_norm.size(0)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(h_norm, h_norm.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = label_mask.float()
        
        # Exclude self-similarity
        pos_mask = pos_mask - torch.eye(batch_size, device=pos_mask.device)
        
        # Negative mask (different labels)
        neg_mask = 1 - pos_mask
        
        # --- ENHANCEMENT 1: Diffusion Noise for Negative Samples ---
        h_with_noise = h_norm + torch.randn_like(h_norm) * self.diffusion_noise_scale
        h_with_noise_norm = F.normalize(h_with_noise, p=2, dim=1)
        sim_matrix_with_noise = torch.matmul(h_norm, h_with_noise_norm.T) / self.temperature
        
        # Use original similarity for positive pairs, noisy similarity for negative pairs
        sim_matrix_enhanced = sim_matrix * pos_mask + sim_matrix_with_noise * neg_mask
        
        # --- ENHANCEMENT 2: Inverse Similarity Weighting ---
        with torch.no_grad():
            # Compute inverse similarity weights based on original similarity
            inv_sim_weights = torch.exp(-sim_matrix / self.inverse_weight_temperature)
            # Normalize weights for each sample
            inv_sim_weights = inv_sim_weights / (inv_sim_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute log probabilities with enhanced similarities
        exp_sim = torch.exp(sim_matrix_enhanced)
        exp_sim_sum = torch.sum(exp_sim * neg_mask, dim=1, keepdim=True) + torch.sum(exp_sim * pos_mask, dim=1, keepdim=True)
        log_prob = sim_matrix_enhanced - torch.log(exp_sim_sum + 1e-8)
        
        # Apply inverse similarity weighting to positive samples
        weighted_pos_log_prob = (log_prob * pos_mask * inv_sim_weights).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        
        # Generalized contrastive loss with enhancements
        gen_loss = -weighted_pos_log_prob.mean()
        
        return gen_loss

    def _compute_individual_contrastive_loss_enhanced(self, h_norm, domain_ids):
        """
        Enhanced Individual Contrastive Loss based on:
        1. Scene-aware contrastive samples: augmented samples with dropout
        2. Cross-scene encoding strategy: using shared repr with other domain-specific nets
        """
        batch_size = h_norm.size(0)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(h_norm, h_norm.T) / self.temperature
        
        # Create mask for same domain (for positive samples in domain-specific context)
        domain_mask = domain_ids.unsqueeze(0) == domain_ids.unsqueeze(1)
        pos_mask = domain_mask.float() - torch.eye(batch_size, device=domain_mask.device)
        
        # Negative mask (different domains)
        neg_mask = 1 - domain_mask.float()
        
        # --- ENHANCEMENT: Scene-aware contrastive samples using dropout augmentation ---
        # Create augmented representations by applying dropout to domain-specific representations
        h_aug = self.dropout_for_aug(h_norm)
        h_aug_norm = F.normalize(h_aug, p=2, dim=1)
        sim_matrix_aug = torch.matmul(h_norm, h_aug_norm.T) / self.temperature
        
        # For positive pairs, use both original and augmented similarities
        # For negative pairs, use original similarities
        sim_matrix_enhanced = sim_matrix * neg_mask + sim_matrix_aug * pos_mask
        
        # Compute log probabilities with enhanced similarities
        exp_sim = torch.exp(sim_matrix_enhanced)
        exp_sim_neg = torch.sum(exp_sim * neg_mask, dim=1, keepdim=True)
        exp_sim_pos = torch.sum(exp_sim * pos_mask, dim=1, keepdim=True)
        
        # Log probability calculation
        denominator = exp_sim_pos + exp_sim_neg
        log_prob = sim_matrix_enhanced - torch.log(denominator + 1e-8)
        
        # Average log probabilities for same-domain (positive) samples
        pos_log_prob = (log_prob * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        
        # Individual contrastive loss (without diffusion noise and inverse similarity weighting)
        ind_loss = -pos_log_prob.mean()
        
        return ind_loss

    def compute_loss(self, predictions, labels, minibatch, **kwargs):
        """Compute total loss including contrastive losses."""
        if isinstance(predictions, tuple):
            output, expert_repr, mmoe_repr = predictions
        else:
            # This shouldn't happen in normal training flow
            raise ValueError("Predictions should be a tuple (output, expert_repr, mmoe_repr) when computing loss")
            
        gen_loss, ind_loss = self.compute_contrastive_losses(expert_repr, mmoe_repr, labels, self._current_domain_ids)
        labels = labels.float().view(-1)
        output = output.view(-1)
        #ms.MovasLogger.random_log(f'compute_loss: output= {output.shape} {output}, labels= {labels.shape} {labels}')
        output = torch.clamp(output, min=1e-7, max=1 - 1e-7)
        main_loss = F.binary_cross_entropy(output, labels)
        # Total loss
        total_loss = main_loss + \
                     self.contrastive_weight_general * gen_loss + \
                     self.contrastive_weight_individual * ind_loss
        
        return total_loss, main_loss
    
    def predict(self, yhat, minibatch=None):
        """Prediction function for inference."""
        if isinstance(yhat, tuple):
            pvr, _, _ = yhat
        else:
            pvr = yhat
        return pvr



