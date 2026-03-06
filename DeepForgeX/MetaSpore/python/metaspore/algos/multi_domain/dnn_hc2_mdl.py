import torch
import torch.nn as nn
import torch.nn.functional as F
import metaspore as ms
from typing import List, Optional

class HC2WideDeepModel(torch.nn.Module):
    """
    HC² (Hybrid Contrastive Constraints) Model based on Wide & Deep architecture.
    Implements both Generalized Contrastive Loss and Individual Contrastive Loss.
    Uses different combine schemas for wide and deep parts, with FTRL optimizer for sparse features.
    Supports shared and domain-specific networks with enhanced contrastive learning features.
    """
    def __init__(self,
                 num_domains: int = 1,
                 embedding_dim: int = 8,
                 shared_hidden_units: List[int] = [256, 128, 64],
                 domain_specific_hidden_units: List[int] = [64, 32],
                 wide_column_name_path: Optional[str] = None,
                 wide_combine_schema_path: Optional[str] = None,
                 deep_column_name_path: Optional[str] = None,
                 deep_combine_schema_path: Optional[str] = None,
                 hidden_activations: str = "ReLU",
                 net_dropout: float = 0.0,
                 batch_norm: bool = False,
                 use_bias: bool = True,
                 embedding_init_var: float = 0.01,
                 contrastive_weight_general: float = 0.1,
                 contrastive_weight_individual: float = 0.05,
                 temperature: float = 0.1,
                 ftrl_l1: float = 1.0,
                 ftrl_l2: float = 120.0,
                 ftrl_alpha: float = 0.5,
                 ftrl_beta: float = 1.0,
                 diffusion_noise_scale: float = 0.1,
                 inverse_weight_temperature: float = 0.05,
                 dropout_augmentation_prob: float = 0.1,
                 **kwargs):
        super(HC2WideDeepModel, self).__init__()
        
        self._current_domain_ids = None
        self.num_domains = num_domains
        self.sparse_embedding_dim = embedding_dim
        self.contrastive_weight_general = contrastive_weight_general
        self.contrastive_weight_individual = contrastive_weight_individual
        self.temperature = temperature
        self.diffusion_noise_scale = diffusion_noise_scale
        self.inverse_weight_temperature = inverse_weight_temperature
        self.dropout_augmentation_prob = dropout_augmentation_prob
        
        # Wide part: linear model with sparse features
        self.wide_sparse_embedding = ms.EmbeddingSumConcat(
            8,
            wide_column_name_path,
            wide_combine_schema_path
        )
        # Apply FTRL optimizer to wide sparse embedding
        if ftrl_l1 > 0 or ftrl_l2 > 0:
            self.wide_sparse_embedding.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.wide_sparse_embedding.initializer = ms.NormalTensorInitializer(var=embedding_init_var)
        
        # Deep part: shared + domain-specific networks
        self.deep_sparse_embedding = ms.EmbeddingSumConcat(
            self.sparse_embedding_dim,
            deep_column_name_path,
            deep_combine_schema_path
        )
        # Apply FTRL optimizer to deep sparse embedding
        if ftrl_l1 > 0 or ftrl_l2 > 0:
            self.deep_sparse_embedding.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.deep_sparse_embedding.initializer = ms.NormalTensorInitializer(var=embedding_init_var)
        
        # Calculate input dimension for deep network
        deep_input_dim = self.deep_sparse_embedding.feature_count * self.sparse_embedding_dim
        
        # Shared deep layers (scene-shared network)
        hidden_activations_fn = getattr(nn, hidden_activations)()
        shared_layers = []
        prev_dim = deep_input_dim
        for unit in shared_hidden_units:
            shared_layers.append(nn.Linear(prev_dim, unit))
            if batch_norm:
                shared_layers.append(nn.BatchNorm1d(unit))
            shared_layers.append(hidden_activations_fn)
            if net_dropout > 0:
                shared_layers.append(nn.Dropout(net_dropout))
            prev_dim = unit
        
        self.shared_deep_layers = nn.Sequential(*shared_layers)
        self.shared_output_dim = prev_dim
        
        # Domain-specific layers (scene-specific network)
        self.domain_specific_layers = nn.ModuleList()
        domain_prev_dim = self.shared_output_dim
        for unit in domain_specific_hidden_units:
            layer = nn.Linear(domain_prev_dim, unit)
            if batch_norm:
                self.domain_specific_layers.append(nn.BatchNorm1d(domain_prev_dim))
            self.domain_specific_layers.append(layer)
            self.domain_specific_layers.append(getattr(nn, hidden_activations)())
            if net_dropout > 0:
                self.domain_specific_layers.append(nn.Dropout(net_dropout))
            domain_prev_dim = unit
        
        # Final output layer for each domain
        self.domain_output_layers = nn.ModuleList([
            nn.Linear(domain_prev_dim, 1) for _ in range(num_domains)
        ])
        
        # Domain embedding for contrastive learning
        self.domain_embedding = nn.Embedding(num_embeddings=num_domains, embedding_dim=self.sparse_embedding_dim)
        
        # Final sigmoid activation
        self.sigmoid = nn.Sigmoid()
        
        # Dropout for augmentation in individual contrastive loss
        self.dropout_for_aug = nn.Dropout(p=self.dropout_augmentation_prob)

    def do_extra_work(self, minibatch):
        """Process domain_id from batch for internal use."""
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
        """Forward pass with optional embedding return for contrastive loss."""
        # Wide part
        wide_output = self.wide_sparse_embedding(x)
        wide_output = torch.sum(wide_output, dim=1, keepdim=True)
        
        # Deep part - shared network
        deep_features = self.deep_sparse_embedding(x)
        shared_deep_output = self.shared_deep_layers(deep_features)
        
        # Deep part - domain-specific network
        domain_specific_output = shared_deep_output
        for layer in self.domain_specific_layers:
            domain_specific_output = layer(domain_specific_output)
        
        # Apply domain-specific output layers
        batch_size = domain_specific_output.shape[0]
        domain_ids = self._current_domain_ids
        assert domain_ids is not None and domain_ids.shape[0] == batch_size
        
        domain_outputs = []
        for i in range(batch_size):
            d_id = domain_ids[i].item()
            domain_layer = self.domain_output_layers[d_id]
            d_out = domain_layer(domain_specific_output[i:i+1, :])
            domain_outputs.append(d_out)
        domain_specific_output_final = torch.cat(domain_outputs, dim=0)
        
        # Combine wide and deep outputs
        combined_output = wide_output + domain_specific_output_final
        output = self.sigmoid(combined_output)
        
        # Return both shared and domain-specific representations for contrastive learning
        return output, shared_deep_output, domain_specific_output

    def compute_contrastive_losses(self, shared_repr, domain_specific_repr, labels, domain_ids):
        """
        Compute both generalized and individual contrastive losses with enhanced features.
        """
        # Normalize embeddings for cosine similarity
        shared_repr_norm = F.normalize(shared_repr, p=2, dim=1)
        domain_specific_repr_norm = F.normalize(domain_specific_repr, p=2, dim=1)
        
        # Generalized Contrastive Loss (based on shared representations and labels)
        gen_loss = self._compute_generalized_contrastive_loss_enhanced(shared_repr_norm, labels, domain_ids)
        
        # Individual Contrastive Loss (based on domain-specific representations and domains)
        ind_loss = self._compute_individual_contrastive_loss_enhanced(domain_specific_repr_norm, domain_ids)
        
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

    def compute_loss(self, predictions, labels, minibatch):
        output, shared_repr, domain_specific_repr = predictions
        gen_loss, ind_loss = self.compute_contrastive_losses(shared_repr, domain_specific_repr, labels, self._current_domain_ids)
        
        # Main prediction loss
        main_loss = F.binary_cross_entropy(output, labels.float())
        
        # Total loss
        total_loss = main_loss + \
                     self.contrastive_weight_general * gen_loss + \
                     self.contrastive_weight_individual * ind_loss
        
        return total_loss, main_loss
    
    def predict(self, yhat, minibatch = None):
        pvr, _, _ = yhat
        return pvr



