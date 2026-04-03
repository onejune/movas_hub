import torch
import metaspore as ms

from .layers import MLPLayer
from .dense_feature import create_dense_encoder


class WideDeep(torch.nn.Module):
    def __init__(self,
                use_wide=True,
                wide_embedding_dim=8,
                deep_embedding_dim=8,
                wide_column_name_path=None,
                wide_combine_schema_path=None,
                deep_column_name_path=None,
                deep_combine_schema_path=None,
                dnn_hidden_units=[512,256,128],
                dnn_hidden_activations="ReLU",
                net_dropout=0,
                batch_norm=False,
                embedding_regularizer=None,
                net_regularizer=None,
                use_bias=True,
                ftrl_l1=1.0,
                ftrl_l2=120.0,
                ftrl_alpha=0.5,
                ftrl_beta=1.0,
                final_net_activation="sigmoid",
                **kwargs):
        super().__init__()
        self.use_wide = use_wide
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(wide_embedding_dim, wide_column_name_path, wide_combine_schema_path)
            self.lr_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        self.dnn_sparse = ms.EmbeddingSumConcat(deep_embedding_dim, deep_column_name_path, deep_combine_schema_path)
        self.dnn_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.dnn_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        self.dnn = MLPLayer(input_dim = self.dnn_sparse.feature_count * deep_embedding_dim,
                                output_dim = 1,
                                hidden_units = dnn_hidden_units,
                                hidden_activations = dnn_hidden_activations,
                                final_activation = None,
                                dropout_rates = net_dropout,
                                batch_norm = batch_norm,
                                use_bias = use_bias,
                                input_norm = True)
        if final_net_activation == 'sigmoid':
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = None

    def forward(self, x):
        if self.use_wide:
            wide_out = self.lr_sparse(x)
            wide_out = torch.sum(wide_out, dim=1, keepdim=True)
        dnn_out = self.dnn_sparse(x)
        dnn_out = self.dnn(dnn_out)
        final_out = torch.add(wide_out, dnn_out) if self.use_wide else dnn_out
        return self.final_activation(final_out) if self.final_activation else final_out


class WideDeep2(torch.nn.Module):
    def __init__(self,
                use_wide=True,
                wide_embedding_dim=10,
                deep_embedding_dim=10,
                wide_column_name_path=None,
                wide_combine_schema_path=None,
                deep_column_name_path=None,
                deep_combine_schema_path=None,
                dnn_hidden_units=[512,256,128],
                dnn_hidden_activations="ReLU",
                net_dropout=0,
                batch_norm=False,
                embedding_regularizer=None,
                net_regularizer=None,
                use_bias=True,
                adam_learning_rate=0.01,
                ftrl_l1=1.0,
                ftrl_l2=120.0,
                ftrl_alpha=0.5,
                ftrl_beta=1.0,
                final_net_activation="sigmoid",
                **kwargs):
        super().__init__()
        self.use_wide = use_wide
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(wide_embedding_dim,
                                                wide_column_name_path,
                                                wide_combine_schema_path)
            self.lr_sparse.updater = ms.AdamTensorUpdater(adam_learning_rate)
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        self.dnn_sparse = ms.EmbeddingSumConcat(deep_embedding_dim,
                                           deep_column_name_path,
                                           deep_combine_schema_path)
        self.dnn_sparse.updater = ms.AdamTensorUpdater(adam_learning_rate)
        self.dnn_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        self.dnn = MLPLayer(input_dim = self.dnn_sparse.feature_count * deep_embedding_dim,
                                output_dim = 1,
                                hidden_units = dnn_hidden_units,
                                hidden_activations = dnn_hidden_activations,
                                final_activation = None,
                                dropout_rates = net_dropout,
                                batch_norm = batch_norm,
                                use_bias = use_bias,
                                input_norm = True)
        if final_net_activation == 'sigmoid':
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = None

    def forward(self, x):
        if self.use_wide:
            wide_out = self.lr_sparse(x)
            wide_out = torch.sum(wide_out, dim=1, keepdim=True)
        dnn_out = self.dnn_sparse(x)
        dnn_out = self.dnn(dnn_out)
        final_out = torch.add(wide_out, dnn_out) if self.use_wide else dnn_out
        return self.final_activation(final_out) if self.final_activation else final_out


class WideDeepDense(torch.nn.Module):
    """
    WideDeep + Dense 特征
    
    - Sparse: EmbeddingSumConcat
    - Dense: 编码器处理 (linear/minmax/standard/log/numeric)
    - 两者 concat 后送入 MLP
    
    dense_fea_list 由 trainFlow 传入，不在模型内加载
    """
    
    def __init__(self,
                 use_wide=True,
                 wide_embedding_dim=8,
                 deep_embedding_dim=8,
                 wide_column_name_path=None,
                 wide_combine_schema_path=None,
                 deep_column_name_path=None,
                 deep_combine_schema_path=None,
                 # Dense 配置
                 dense_fea_list=None,           # 由 trainFlow 传入
                 dense_encoder_type='linear',
                 dense_output_dim=None,
                 dense_embedding_dim=16,
                 dense_hidden_dim=64,
                 dense_batch_norm=True,
                 dense_dropout=0.0,
                 # DNN 配置
                 dnn_hidden_units=[512, 256, 128],
                 dnn_hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 use_bias=True,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 final_net_activation="sigmoid",
                 **kwargs):
        super().__init__()
        
        self.use_wide = use_wide
        self.dense_fea_list = dense_fea_list or []
        
        # Wide
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(
                wide_embedding_dim, wide_column_name_path, wide_combine_schema_path
            )
            self.lr_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        
        # Deep Sparse
        self.dnn_sparse = ms.EmbeddingSumConcat(
            deep_embedding_dim, deep_column_name_path, deep_combine_schema_path
        )
        self.dnn_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.dnn_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        sparse_dim = self.dnn_sparse.feature_count * deep_embedding_dim
        
        # Dense
        num_dense = len(self.dense_fea_list)
        if num_dense > 0:
            encoder_kwargs = {}
            if dense_encoder_type == 'linear':
                encoder_kwargs = {'output_dim': dense_output_dim, 'batch_norm': dense_batch_norm, 'dropout': dense_dropout}
            elif dense_encoder_type == 'numeric':
                encoder_kwargs = {'embedding_dim': dense_embedding_dim, 'hidden_dim': dense_hidden_dim}
            
            self.dense_encoder = create_dense_encoder(dense_encoder_type, num_dense, **encoder_kwargs)
            dense_dim = self.dense_encoder.output_dim
            
            # 如果需要映射到指定维度
            if dense_output_dim and dense_output_dim != dense_dim:
                self.dense_proj = torch.nn.Linear(dense_dim, dense_output_dim)
                dense_dim = dense_output_dim
            else:
                self.dense_proj = None
            
            print(f"[WideDeepDense] Sparse: {sparse_dim}, Dense: {dense_dim} ({dense_encoder_type})")
        else:
            self.dense_encoder = None
            self.dense_proj = None
            dense_dim = 0
            print(f"[WideDeepDense] Sparse only: {sparse_dim}")
        
        # MLP
        self.dnn = MLPLayer(
            input_dim=sparse_dim + dense_dim,
            output_dim=1,
            hidden_units=dnn_hidden_units,
            hidden_activations=dnn_hidden_activations,
            final_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
            input_norm=True
        )
        
        self.final_activation = torch.nn.Sigmoid() if final_net_activation == 'sigmoid' else None
        
        # 用于存储 dense 数据
        self._dense_data = None
    
    def do_extra_work(self, minibatch):
        """从 minibatch (DataFrame) 提取 dense 特征"""
        if self.dense_encoder is None or not self.dense_fea_list:
            return
        
        # minibatch 是 DataFrame，直接按列名提取
        # .astype(float) 确保是 float64，然后 torch 转 float32
        import numpy as np
        dense_values = minibatch[self.dense_fea_list].values.astype(np.float32)
        self._dense_data = torch.from_numpy(dense_values)
    
    def forward(self, x):
        # Wide
        if self.use_wide:
            wide_out = self.lr_sparse(x)
            wide_out = torch.sum(wide_out, dim=1, keepdim=True)
        
        # Sparse
        sparse_out = self.dnn_sparse(x)
        
        # Dense
        if self.dense_encoder is not None and self._dense_data is not None:
            dense_out = self.dense_encoder(self._dense_data)
            if self.dense_proj:
                dense_out = self.dense_proj(dense_out)
            dnn_input = torch.cat([sparse_out, dense_out], dim=1)
        else:
            dnn_input = sparse_out
        
        # MLP
        dnn_out = self.dnn(dnn_input)
        
        # Combine
        final_out = torch.add(wide_out, dnn_out) if self.use_wide else dnn_out
        
        return self.final_activation(final_out) if self.final_activation else final_out
