
import torch
import metaspore as ms

from .layers import MLPLayer
from .dense_feature import DenseFeatureLayer


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
    WideDeep 模型 - 支持 Dense 特征
    
    在原有 WideDeep 基础上增加 Dense 特征支持:
    - Sparse 特征: 通过 EmbeddingSumConcat 获取 Embedding
    - Dense 特征: 通过 DenseFeatureLayer 处理连续值
    - 两者 concat 后送入 MLP
    
    配置示例:
        dense_features_path: ./conf/dense_features  # 每行一个特征名
        dense_output_dim: 16  # 可选，映射到固定维度
    
    Args:
        dense_features_path: dense 特征配置文件路径 (每行一个特征名)
        dense_output_dim: dense 特征输出维度，None 保持原始维度
        其他参数同 WideDeep
    """
    
    def __init__(self,
                 use_wide=True,
                 wide_embedding_dim=8,
                 deep_embedding_dim=8,
                 wide_column_name_path=None,
                 wide_combine_schema_path=None,
                 deep_column_name_path=None,
                 deep_combine_schema_path=None,
                 # Dense 特征配置
                 dense_features_path=None,
                 dense_output_dim=None,
                 dense_batch_norm=True,
                 dense_dropout=0.0,
                 # DNN 配置
                 dnn_hidden_units=[512, 256, 128],
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
        self.dense_features_path = dense_features_path
        
        # Wide 部分 (线性模型)
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(
                wide_embedding_dim,
                wide_column_name_path,
                wide_combine_schema_path
            )
            self.lr_sparse.updater = ms.FTRLTensorUpdater(
                l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta
            )
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        
        # Deep Sparse 部分
        self.dnn_sparse = ms.EmbeddingSumConcat(
            deep_embedding_dim,
            deep_column_name_path,
            deep_combine_schema_path
        )
        self.dnn_sparse.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta
        )
        self.dnn_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        sparse_dim = self.dnn_sparse.feature_count * deep_embedding_dim
        
        # Dense 部分 (新增)
        if dense_features_path:
            self.dense_layer = DenseFeatureLayer(
                dense_features_path=dense_features_path,
                output_dim=dense_output_dim,
                batch_norm=dense_batch_norm,
                dropout=dense_dropout
            )
            dense_dim = self.dense_layer.output_dim
            print(f"[WideDeepDense] Sparse dim: {sparse_dim}, Dense dim: {dense_dim}")
        else:
            self.dense_layer = None
            dense_dim = 0
            print(f"[WideDeepDense] Sparse only, dim: {sparse_dim}")
        
        # MLP 输入维度 = sparse + dense
        mlp_input_dim = sparse_dim + dense_dim
        
        self.dnn = MLPLayer(
            input_dim=mlp_input_dim,
            output_dim=1,
            hidden_units=dnn_hidden_units,
            hidden_activations=dnn_hidden_activations,
            final_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=use_bias,
            input_norm=True
        )
        
        if final_net_activation == 'sigmoid':
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = None
        
        # 存储列名信息 (运行时从 minibatch 获取)
        self._column_names = None
    
    def do_extra_work(self, minibatch):
        """
        钩子函数: 从 minibatch 提取 dense 特征
        
        在 SparseModel.__call__() 中，forward() 之前被调用
        """
        if self.dense_layer is not None:
            # 首次调用时获取列名
            if self._column_names is None:
                if hasattr(minibatch, 'column_names'):
                    self._column_names = minibatch.column_names
                elif hasattr(minibatch, 'schema'):
                    self._column_names = minibatch.schema
                else:
                    # 尝试从 conf/column_name 加载
                    try:
                        with open('./conf/column_name', 'r') as f:
                            self._column_names = [line.strip() for line in f if line.strip()]
                    except:
                        raise ValueError("Cannot get column names from minibatch or conf/column_name")
            
            self.dense_layer.extract(minibatch, self._column_names)
    
    def forward(self, x):
        """前向传播"""
        # Wide 部分
        if self.use_wide:
            wide_out = self.lr_sparse(x)
            wide_out = torch.sum(wide_out, dim=1, keepdim=True)
        
        # Deep Sparse 部分
        sparse_out = self.dnn_sparse(x)
        
        # Dense 部分 (新增)
        if self.dense_layer is not None:
            dense_out = self.dense_layer(x)
            dnn_input = torch.cat([sparse_out, dense_out], dim=1)
        else:
            dnn_input = sparse_out
        
        # MLP
        dnn_out = self.dnn(dnn_input)
        
        # Wide + Deep 融合
        final_out = torch.add(wide_out, dnn_out) if self.use_wide else dnn_out
        
        return self.final_activation(final_out) if self.final_activation else final_out
