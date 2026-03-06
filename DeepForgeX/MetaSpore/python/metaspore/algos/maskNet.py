import torch
import metaspore as ms
from .layers import MLPLayer

class MaskBlock(torch.nn.Module):
    """
    Mask Block: 包含 Instance-Guided Mask 和 LN-EMB 分支
    """
    def __init__(self, embedding_dim, total_feature_dim, dropout_rate=None, use_bias=True, input_norm=False, batch_norm=False):
        super(MaskBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.total_feature_dim = total_feature_dim

        # Aggregation Layer: 维度扩展
        self.aggregation_layer = MLPLayer(
            input_dim=total_feature_dim,
            output_dim=total_feature_dim * 2,  # 扩展维度
            hidden_units=[],  # 仅一个线性层
            hidden_activations='ReLU',
            final_activation=None,
            dropout_rates=dropout_rate,
            input_norm=input_norm,
            batch_norm=batch_norm,
            use_bias=use_bias
        )

        # Projection Layer: 维度压缩回 total_feature_dim
        self.projection_layer = MLPLayer(
            input_dim=total_feature_dim * 2,
            output_dim=total_feature_dim,
            hidden_units=[],  # 仅一个线性层
            hidden_activations=None,
            final_activation=None,
            dropout_rates=dropout_rate,
            input_norm=input_norm,
            batch_norm=batch_norm,
            use_bias=use_bias
        )

        # Dropout for regularization
        self.dropout = torch.nn.Dropout(dropout_rate) if dropout_rate else None

    def forward(self, inputs):
        # inputs shape: [batch_size, total_feature_dim]
        batch_size = inputs.size(0)
        num_features = self.total_feature_dim // self.embedding_dim

        # Reshape to [batch_size, num_features, embedding_dim]
        reshaped_inputs = inputs.view(batch_size, num_features, self.embedding_dim)

        # Left Branch: Layer Normalization per feature
        ln_emb_branch = torch.nn.functional.layer_norm(reshaped_inputs, (self.embedding_dim,))
        ln_emb_branch_flat = ln_emb_branch.view(batch_size, self.total_feature_dim)

        # Right Branch: Instance-Guided Mask
        agg_out = self.aggregation_layer(inputs)  # [batch_size, expanded_dim]
        proj_out = self.projection_layer(agg_out)  # [batch_size, total_feature_dim]
        mask_vector = torch.sigmoid(proj_out)  # Sigmoid for soft mask
        if self.dropout:
            mask_vector = self.dropout(mask_vector)

        # Hadamard Product of both branches
        output = ln_emb_branch_flat * mask_vector

        return output


class MaskNet(torch.nn.Module):
    """
    MaskNet: 支持 Wide 部分的改造版本
    """
    def __init__(self,
                 use_wide=True,
                 wide_embedding_dim=10,
                 deep_embedding_dim=10,
                 wide_column_name_path=None,
                 wide_combine_schema_path=None,
                 deep_column_name_path=None,
                 deep_combine_schema_path=None,
                 n_blocks=2,
                 block_type='serial',  # 'serial' or 'parallel'
                 mask_hidden_units=[],
                 dnn_hidden_units=[128, 64],
                 dnn_hidden_activations='ReLU',
                 use_bias=True,
                 input_norm=False,
                 batch_norm=False,
                 net_dropout=None,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 sparse_init_var=1e-2,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 final_net_activation="sigmoid"):
        super().__init__()
        self.use_wide = use_wide
        self.n_blocks = n_blocks
        self.block_type = block_type

        # Wide 部分
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(
                wide_embedding_dim,
                wide_column_name_path,
                wide_combine_schema_path
            )
            self.lr_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=0.01)

        # Deep 部分（MaskNet）
        self.dnn_sparse = ms.EmbeddingSumConcat(
            deep_embedding_dim,
            deep_column_name_path,
            deep_combine_schema_path
        )
        self.dnn_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.dnn_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        
        # Mask Blocks
        self.mask_blocks = torch.nn.ModuleList()
        total_feature_dim = int(self.dnn_sparse.feature_count * deep_embedding_dim)
        for i in range(n_blocks):
            block = MaskBlock(
                embedding_dim=deep_embedding_dim,
                total_feature_dim=total_feature_dim,
                dropout_rate=net_dropout,
                use_bias=use_bias,
                input_norm=input_norm,
                batch_norm=batch_norm
            )
            self.mask_blocks.append(block)

        # For Parallel: Concatenation layer
        if self.block_type == 'parallel':
            self.concat_layer = torch.nn.Linear(total_feature_dim * n_blocks, total_feature_dim)

        # DNN layers after Mask Blocks
        self.dnn = MLPLayer(
            input_dim=total_feature_dim,
            output_dim=1,
            hidden_units=dnn_hidden_units,
            hidden_activations=dnn_hidden_activations,
            final_activation=None,  # Final activation handled separately
            dropout_rates=net_dropout,
            input_norm=input_norm,
            batch_norm=batch_norm,
            use_bias=use_bias
        )

        # Final activation
        if final_net_activation == 'sigmoid':
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = None

    def forward(self, x):
        # Wide 部分
        if self.use_wide:
            wide_out = self.lr_sparse(x)
            wide_out = torch.sum(wide_out, dim=1, keepdim=True)
        
        # Deep 部分 (MaskNet)
        deep_out = self.dnn_sparse(x)  # [batch_size, total_feature_dim]

        if self.block_type == 'serial':
            for block in self.mask_blocks:
                deep_out = block(deep_out)
        elif self.block_type == 'parallel':
            block_outputs = []
            for block in self.mask_blocks:
                block_out = block(deep_out)
                block_outputs.append(block_out)
            deep_out = torch.cat(block_outputs, dim=1)  # [batch_size, total_feature_dim * n_blocks]
            deep_out = self.concat_layer(deep_out)  # [batch_size, total_feature_dim]
        else:
            raise ValueError(f"block_type must be 'serial' or 'parallel', got {self.block_type}")

        deep_out = self.dnn(deep_out)

        # Combine Wide and Deep
        final_out = torch.add(wide_out, deep_out) if self.use_wide else deep_out
        
        return self.final_activation(final_out) if self.final_activation else final_out