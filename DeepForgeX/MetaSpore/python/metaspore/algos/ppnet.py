import torch
import metaspore as ms
from .layers import MLPLayer # 假设MLPLayer在同级layers模块中

class PPNetGateLayer(torch.nn.Module):
    """
    PPNet中的Gate层，用于生成个性化缩放权重。
    其输入包含左侧特征（已detach）和ID特征（未detach）。
    """
    def __init__(self, gate_input_dim, target_output_dim, hidden_dim=64):
        """
        Args:
            gate_input_dim (int): Gate网络的输入维度（通常是所有特征Embedding的总维度）。
            target_output_dim (int): Gate网络输出的维度，应与目标DNN层的输出维度一致。
            hidden_dim (int): Gate网络中间层的维度。
        """
        super(PPNetGateLayer, self).__init__()
        self.linear1 = torch.nn.Linear(gate_input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_dim, target_output_dim)
        self.sigmoid = torch.nn.Sigmoid()
        self.scale_factor = 2.0

    def forward(self, gate_input):
        """
        Args:
            gate_input (Tensor): Gate网络的输入张量，形状为 [batch_size, gate_input_dim]。
                                 其中一部分来自detach()后的左侧特征embedding，另一部分来自原始的ID特征embedding。
        Returns:
            Tensor: 个性化缩放权重，形状为 [batch_size, target_output_dim]。
        """
        x = self.linear1(gate_input)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x * self.scale_factor


class PPNet(torch.nn.Module):
    """
    快手参数个性化CTR模型 - PPNet (Parameter Personalized Net)
    使用MetaSpore风格实现，通过分离的Embedding层区分DNN和Gate的输入。
    Gate部分被抽象为独立的PPNetGateLayer。
    GateNN的输入中，左侧特征不接收反向传播梯度，ID特征接收梯度。
    """
    def __init__(self,
                 embedding_dim=16,
                 column_name_path=None,        # Path for non-ID sparse features
                 combine_schema_path=None,     # Schema for non-ID sparse features
                 id_column_name_path=None,     # Path for ID features (e.g., uid, pid, aid)
                 gate_combine_schema_path=None,  # Schema for ID features
                 dnn_hidden_units=[1024, 512, 256],
                 dnn_hidden_activations="ReLU",
                 net_dropout=0.0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 use_bias=True,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 final_net_activation="sigmoid",
                 gate_hidden_dim=64, # 用于 PPNetGateLayer 的 hidden_dim
                 **kwargs):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._gate_hidden_dim = gate_hidden_dim

        # --- Embedding Layer for Non-ID Features (DNN Input & Left part of Gate Input) ---
        self._dnn_sparse_embedding = ms.EmbeddingSumConcat(
            self._embedding_dim,
            column_name_path,
            combine_schema_path
        )
        self._dnn_sparse_embedding.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1,
            l2=ftrl_l2,
            alpha=ftrl_alpha,
            beta=ftrl_beta
        )
        self._dnn_sparse_embedding.initializer = ms.NormalTensorInitializer(var=0.001)

        # --- Embedding Layer for ID Features (Right part of Gate Input) ---
        self._gate_id_embedding = ms.EmbeddingSumConcat(
            self._embedding_dim,
            id_column_name_path,
            gate_combine_schema_path
        )
        self._gate_id_embedding.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1,
            l2=ftrl_l2,
            alpha=ftrl_alpha,
            beta=ftrl_beta
        )
        self._gate_id_embedding.initializer = ms.NormalTensorInitializer(var=0.001)

        # --- Calculate dimensions ---
        dnn_sparse_feature_count = self._dnn_sparse_embedding.feature_count
        dnn_input_dim = dnn_sparse_feature_count * self._embedding_dim

        id_feature_count = self._gate_id_embedding.feature_count
        id_emb_dim_total = id_feature_count * self._embedding_dim

        # Total embedding dim for Gate NN input (DNN sparse emb + ID emb)
        gate_input_dim = dnn_input_dim + id_emb_dim_total


        # --- Batch Normalization (applies to DNN sparse embeddings) ---
        if batch_norm:
            self._bn = ms.nn.Normalization(dnn_input_dim, momentum=0.01, eps=1e-5, affine=True)
        else:
            self._bn = None

        # --- DNN Layers (Main Network) and Gate Layers ---
        self._dnn_layers = torch.nn.ModuleList()
        self._gate_layers = torch.nn.ModuleList()

        input_dim = dnn_input_dim
        for hidden_size in dnn_hidden_units:
            # Add Linear layer for DNN main path
            linear_layer = torch.nn.Linear(input_dim, hidden_size, bias=use_bias)
            self._dnn_layers.append(linear_layer)

            # Add corresponding PPNet Gate layer
            gate_layer = PPNetGateLayer(
                gate_input_dim=gate_input_dim,
                target_output_dim=hidden_size,
                hidden_dim=self._gate_hidden_dim
            )
            self._gate_layers.append(gate_layer)

            # Add Activation layer
            if isinstance(dnn_hidden_activations, str):
                activation_cls = getattr(torch.nn, dnn_hidden_activations, None)
                if activation_cls:
                    self._dnn_layers.append(activation_cls())
                else:
                    raise ValueError(f"Activation function {dnn_hidden_activations} not found in torch.nn")
            else:
                self._dnn_layers.append(dnn_hidden_activations)

            # Add Dropout layer
            if net_dropout > 0:
                self._dnn_layers.append(torch.nn.Dropout(net_dropout))

            input_dim = hidden_size # Update input dimension for next layer

        # --- Final Output Layer ---
        self._output_layer = torch.nn.Linear(input_dim, 1, bias=use_bias)
        if final_net_activation == 'sigmoid':
            self._final_activation = torch.nn.Sigmoid()
        elif final_net_activation == 'relu':
            self._final_activation = torch.nn.ReLU()
        else:
            self._final_activation = None # No final activation


    def forward(self, x):
        # 1. Get embeddings for Non-ID features (for DNN main path & LEFT part of Gate NN input)
        dnn_sparse_emb = self._dnn_sparse_embedding(x) # Shape: [B, F_non_id * D]

        # 2. Get embeddings for ID features (for RIGHT part of Gate NN input)
        id_emb = self._gate_id_embedding(x) # Shape: [B, F_id * D]

        # 3. Apply Batch Normalization (optional) to DNN sparse embeddings (used in DNN main path)
        if self._bn is not None:
            dnn_sparse_emb_bn = self._bn(dnn_sparse_emb)
        else:
            dnn_sparse_emb_bn = dnn_sparse_emb

        # 4. Prepare input for Gate layers (concatenate DNN sparse emb (DETACHED) and ID emb (ORIGINAL))
        # CRITICAL: Only detach the left part (non-ID features) for Gate NN input.
        # The right part (ID features) should remain connected to allow gradients.
        gate_nn_input = torch.cat([dnn_sparse_emb_bn.detach(), id_emb], dim=1) # Shape: [B, (F_non_id + F_id) * D]
        # detach() creates a new tensor for dnn_sparse_emb_bn that does not track gradients,
        # ensuring gradients from Gate NN do not flow back to the left-side embeddings.

        # 5. Forward pass through DNN with Gates
        # Use the original, non-detached embeddings for the main DNN path
        current_input = dnn_sparse_emb_bn 
        layer_idx = 0
        for i in range(len(self._dnn_layers)):
            layer = self._dnn_layers[i]
            if isinstance(layer, torch.nn.Linear):
                # Apply Linear layer of DNN main path
                output_before_activation = layer(current_input)
                # Apply Gate layer using the mixed input: detached left part, original right part
                gate_weights = self._gate_layers[layer_idx](gate_nn_input) # Input: [B, (F*D)_total] with left part detached
                # Apply gate: element-wise multiplication
                current_input = output_before_activation * gate_weights
                layer_idx += 1 # Move to next gate for next DNN layer
            else:
                # Apply Activation or Dropout
                current_input = layer(current_input)

        # 6. Final Output
        output = self._output_layer(current_input)
        if self._final_activation is not None:
            output = self._final_activation(output)

        return output