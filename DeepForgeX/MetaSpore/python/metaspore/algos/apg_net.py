import torch
import metaspore as ms
import torch.nn.functional as F

from .layers import LRLayer, MLPLayer, CrossNet

class APGLinear(torch.nn.Module):
    """
    APG Linear Layer implementing adaptive parameter generation.
    This replaces traditional linear layers with dynamic weight generation based on input conditions.
    """
    def __init__(self, input_dim, output_dim, generator_hidden_dim, low_rank_k=64, condition_dim=None):
        super(APGLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_k = low_rank_k
        
        # Shared parameters (U and V in the paper's decomposition)
        # U is (output_dim, low_rank_k), V is (low_rank_k, input_dim)
        self.U_shared = torch.nn.Parameter(torch.randn(output_dim, low_rank_k))
        self.V_shared = torch.nn.Parameter(torch.randn(low_rank_k, input_dim))
        
        # For generating S (the specific part), we need a small network that takes condition
        if condition_dim is None:
            condition_dim = input_dim  # Default to using input as condition
        self.condition_dim = condition_dim
        
        # --- Fix: Add a projection layer if condition_dim != input_dim ---
        if condition_dim != input_dim:
            self.condition_projector = torch.nn.Linear(condition_dim, input_dim)
        else:
            self.condition_projector = None
        # ---

        # Small network to generate the core matrix S from condition
        # S will have shape (low_rank_k, low_rank_k) per instance
        # It now expects input_dim features after potential projection
        self.s_generator = torch.nn.Sequential(
            torch.nn.Linear(input_dim, generator_hidden_dim), # Use a new hyperparameter
            torch.nn.ReLU(),
            torch.nn.Linear(generator_hidden_dim, low_rank_k), # Intermediate layer before output
            torch.nn.ReLU(), # Non-linearity before final layer
            torch.nn.Linear(low_rank_k, low_rank_k * low_rank_k),
            torch.nn.Tanh()
        )
        
        # Initialize shared parameters
        torch.nn.init.xavier_uniform_(self.U_shared)
        torch.nn.init.xavier_uniform_(self.V_shared)
        # Initialize projector if exists
        if self.condition_projector is not None:
             torch.nn.init.xavier_uniform_(self.condition_projector.weight)
             if self.condition_projector.bias is not None:
                 torch.nn.init.zeros_(self.condition_projector.bias)


    def forward(self, x, condition=None):
        batch_size = x.size(0)

        # If no condition provided, use the input itself
        if condition is None:
            # Use mean pooling of input features as condition (or just the input if already flat)
            # For APGNet, we typically pass the original flat_x, so it should match the first layer's condition_dim
            # But subsequent layers might receive flat_x which has different dim than their input_dim
            # So we project condition to match what s_generator expects (input_dim of THIS layer)
            condition = x.mean(dim=1) if len(x.shape) > 2 else x # Default fallback if condition is None
        
        # --- Fix: Project condition to match s_generator's expected input dimension ---
        projected_condition = self.condition_projector(condition) if self.condition_projector is not None else condition
        # Ensure projected_condition matches the expected input dimension for s_generator
        if projected_condition.size(-1) != self.input_dim:
             raise RuntimeError(f"Projected condition dimension {projected_condition.size(-1)} does not match "
                                f"s_generator's expected input dimension {self.input_dim}.")

        # Generate S matrix for each instance in the batch
        s_flat = self.s_generator(projected_condition)  # Shape: (batch_size, low_rank_k * low_rank_k)
        S_specific = s_flat.view(batch_size, self.low_rank_k, self.low_rank_k)  # Shape: (batch_size, low_rank_k, low_rank_k)
        
        v_x = torch.matmul(x, self.V_shared.t())  # Shape: (batch_size, low_rank_k)
        s_v_x_unsqueezed = torch.bmm(S_specific, v_x.unsqueeze(-1)) # (batch_size, low_rank_k, 1)
        s_v_x = s_v_x_unsqueezed.squeeze(-1) # (batch_size, low_rank_k)
        output = torch.matmul(s_v_x, self.U_shared.t()) # Shape: (batch_size, output_dim)
        
        return output


class APGNet(torch.nn.Module):
    """
    APG Network implementation based on the paper 'APG: Adaptive Parameter Generation Network for Click-Through Rate Prediction'.
    This network uses adaptive parameter generation to improve CTR prediction performance.
    """
    def __init__(self,
                 use_wide=True,
                 wide_embedding_dim=10,
                 deep_embedding_dim=10,
                 wide_combine_schema_path=None,
                 deep_combine_schema_path=None,
                 sparse_init_var=1e-2,
                 adam_learning_rate=1e-5,
                 apg_hidden_units=[512, 256],
                 generator_hidden_dim=512,
                 apg_activations='ReLU',
                 use_bias=True,
                 batch_norm=False,
                 net_dropout=None,
                 net_regularizer=None,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 apg_low_rank_k=64,
                 sparse_optimizer_type='adam'):  # New parameter
        super(APGNet, self).__init__()
        self.use_wide = use_wide
        self.apg_low_rank_k = apg_low_rank_k

        # Wide layer (LR layer) - similar to DCN but using APG principles could also be adapted
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(wide_embedding_dim, None, wide_combine_schema_path)
            # Set updater based on config using the external function
            ms.set_updater(self.lr_sparse, sparse_optimizer_type, 
                           learning_rate=adam_learning_rate, 
                           l1=ftrl_l1, 
                           l2=ftrl_l2, 
                           alpha=ftrl_alpha, 
                           beta=ftrl_beta)
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
            self.lr = LRLayer(wide_embedding_dim, self.lr_sparse.feature_count)
        
        self.dnn_sparse = ms.EmbeddingSumConcat(deep_embedding_dim, None, deep_combine_schema_path)
        
        # Set updater based on config using the external function
        ms.set_updater(self.dnn_sparse, sparse_optimizer_type,
                       learning_rate=adam_learning_rate,
                       l1=ftrl_l1,
                       l2=ftrl_l2,
                       alpha=ftrl_alpha,
                       beta=ftrl_beta)
        
        self.dnn_sparse.initializer = ms.NormalTensorInitializer(var=0.01)

        self.dnn_input_dim = int(self.dnn_sparse.feature_count * deep_embedding_dim)
        self.dnn_sparse_bn = ms.nn.Normalization(self.dnn_input_dim)

        # APG layers - replacing traditional DNN layers
        self.apg_layers = torch.nn.ModuleList()
        prev_dim = self.dnn_input_dim
        
        for i, unit in enumerate(apg_hidden_units):
            # Pass the input_dim of this layer as the condition_dim by default
            # This means condition will be projected to match the layer's input_dim
            apg_layer = APGLinear(prev_dim, unit, generator_hidden_dim, low_rank_k=self.apg_low_rank_k, condition_dim=self.dnn_input_dim)
            self.apg_layers.append(apg_layer)
            prev_dim = unit

        # Final output layer
        self.final_apg_layer = APGLinear(prev_dim, 1, generator_hidden_dim, low_rank_k=self.apg_low_rank_k, condition_dim=self.dnn_input_dim)

        # Batch normalization option for APG layers
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn_layers = torch.nn.ModuleList([
                torch.nn.BatchNorm1d(unit) for unit in apg_hidden_units
            ])
        
        # Activation function
        if apg_activations == 'ReLU':
            self.activation = torch.nn.ReLU()
        elif apg_activations == 'Tanh':
            self.activation = torch.nn.Tanh()
        elif apg_activations == 'Sigmoid':
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = torch.nn.ReLU()  # Default
            
        # Dropout
        self.dropout = torch.nn.Dropout(net_dropout) if net_dropout else None

        # Final activation
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, x):
        # Get embeddings
        x = self.dnn_sparse(x)
        x_bn = self.dnn_sparse_bn(x)
        
        # Flatten embeddings for dense processing
        flat_x = x_bn.contiguous().view(-1, self.dnn_input_dim)
        
        # Process through APG layers
        current_input = flat_x
        condition = flat_x
        
        for i, apg_layer in enumerate(self.apg_layers):
            current_input = apg_layer(current_input, condition)
            if self.batch_norm and i < len(self.bn_layers):
                current_input = self.bn_layers[i](current_input)
            current_input = self.activation(current_input)
            if self.dropout:
                current_input = self.dropout(current_input)
        
        # Final APG layer to get logits
        logit = self.final_apg_layer(current_input, condition)
        # Add wide component if enabled
        if self.use_wide:
            lr_feature_map = self.lr_sparse(x) # Use x from sparse layer
            lr_logit = self.lr(lr_feature_map)
            logit += lr_logit

        # Apply final activation
        prediction = self.final_activation(logit)
        return prediction

    def compute_loss(self, predictions, labels, minibatch, **kwargs):
        labels = labels.float().view(-1)
        output = predictions.view(-1)
        if torch.isnan(output).any():
            print("WARNING: NaN detected in model predictions during loss calculation.")
            output = torch.nan_to_num(output, nan=0.5, posinf=1.0 - 1e-7, neginf=1e-7)
        output = torch.clamp(output, min=1e-7, max=1 - 1e-7)
        total_loss = F.binary_cross_entropy(output, labels, reduction='mean')

        return total_loss, total_loss