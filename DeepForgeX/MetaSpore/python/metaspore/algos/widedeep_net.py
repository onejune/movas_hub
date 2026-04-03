
import torch
import metaspore as ms

from .layers import MLPLayer
from .dense_feature import (
    DenseFeatureLayer, 
    DenseFeatureMinMaxScaler, 
    DenseFeatureStandardScaler, 
    DenseFeatureNumericEmbedding, 
    DenseFeatureLogTransform
)


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
    - Dense 特征: 通过多种编码器处理连续值
    - 两者 concat 后送入 MLP
    
    支持的编码器类型:
    - 'linear': DenseFeatureLayer (基础线性变换 + BN + Dropout)
    - 'minmax': DenseFeatureMinMaxScaler (归一化到 [0,1])
    - 'standard': DenseFeatureStandardScaler (Z-score 标准化)
    - 'numeric': DenseFeatureNumericEmbedding (每特征独立 MLP)
    - 'log': DenseFeatureLogTransform (对数变换)
    
    配置示例:
        dense_features_path: ./conf/dense_features  # 每行一个特征名
        dense_encoder_type: 'linear'  # 编码器类型
        dense_output_dim: 16  # 可选，映射到固定维度
    
    Args:
        dense_features_path: dense 特征配置文件路径 (每行一个特征名)
        dense_encoder_type: 编码器类型 ('linear'|'minmax'|'standard'|'numeric'|'log')
        dense_output_dim: dense 特征输出维度，None 保持原始维度
        dense_embedding_dim: numeric encoder 的 embedding 维度
        dense_hidden_dim: numeric encoder 的隐藏层维度
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
                 dense_encoder_type='linear',  # 新增：编码器类型
                 dense_output_dim=None,
                 dense_embedding_dim=16,  # numeric encoder 专用
                 dense_hidden_dim=64,     # numeric encoder 专用
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
        self.dense_encoder_type = dense_encoder_type
        
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
        
        # Dense 部分 (新增) - 支持多种编码器
        if dense_features_path:
            # 根据类型创建对应的编码器
            if dense_encoder_type == 'linear':
                self.dense_layer = DenseFeatureLayer(
                    dense_features_path=dense_features_path,
                    output_dim=dense_output_dim,
                    batch_norm=dense_batch_norm,
                    dropout=dense_dropout
                )
            elif dense_encoder_type == 'minmax':
                self.dense_layer = DenseFeatureMinMaxScaler(
                    dense_features=self.dense_fea_list
                )
                
                # 对于 MinMaxScaler，输出维度通常等于特征数
                # 如果指定了 dense_output_dim，额外添加线性变换
                if dense_output_dim and dense_output_dim != self.dense_layer.output_dim:
                    self.dense_post_linear = torch.nn.Linear(
                        self.dense_layer.output_dim, dense_output_dim
                    )
                else:
                    self.dense_post_linear = None
            elif dense_encoder_type == 'standard':
                self.dense_layer = DenseFeatureStandardScaler(
                    dense_features=self.dense_fea_list
                )
                
                if dense_output_dim and dense_output_dim != self.dense_layer.output_dim:
                    self.dense_post_linear = torch.nn.Linear(
                        self.dense_layer.output_dim, dense_output_dim
                    )
                else:
                    self.dense_post_linear = None
            elif dense_encoder_type == 'numeric':
                self.dense_layer = DenseFeatureNumericEmbedding(
                    dense_features=self.dense_fea_list,
                    embedding_dim=dense_embedding_dim,
                    hidden_dim=dense_hidden_dim
                )
                
                # Numeric embedding 的输出维度是特征数 * embedding_dim
                # 如果指定了 dense_output_dim，添加线性变换
                if dense_output_dim and dense_output_dim != self.dense_layer.output_dim:
                    self.dense_post_linear = torch.nn.Linear(
                        self.dense_layer.output_dim, dense_output_dim
                    )
                else:
                    self.dense_post_linear = None
            elif dense_encoder_type == 'log':
                self.dense_layer = DenseFeatureLogTransform(
                    dense_features=self.dense_fea_list
                )
                
                if dense_output_dim and dense_output_dim != self.dense_layer.output_dim:
                    self.dense_post_linear = torch.nn.Linear(
                        self.dense_layer.output_dim, dense_output_dim
                    )
                else:
                    self.dense_post_linear = None
            else:
                raise ValueError(f"Unknown dense encoder type: {dense_encoder_type}. "
                               f"Supported: linear, minmax, standard, numeric, log")
            
            # 计算最终 dense 维度
            if hasattr(self, 'dense_post_linear') and self.dense_post_linear is not None:
                dense_dim = dense_output_dim
            else:
                dense_dim = self.dense_layer.output_dim
            
            print(f"[WideDeepDense] Sparse dim: {sparse_dim}, Dense dim: {dense_dim}, "
                  f"Encoder: {dense_encoder_type}")
        else:
            self.dense_layer = None
            self.dense_post_linear = None
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
        
        # 存储列名信息 (由 trainFlow 设置)
        self._column_names = None
    
    def set_column_names(self, column_names: list):
        """
        设置列名列表，用于 dense 特征提取
        
        在 trainFlow 中构建模型后调用:
            self.model_module.set_column_names(self.used_fea_list)
        """
        self._column_names = column_names
        if self.dense_layer is not None:
            if hasattr(self.dense_layer, 'set_column_names'):
                self.dense_layer.set_column_names(column_names)
            print(f"[WideDeepDense] Column names set, total {len(column_names)} columns")
    
    def do_extra_work(self, minibatch):
        """
        钩子函数: 从 minibatch 提取 dense 特征
        
        在 SparseModel.__call__() 中，forward() 之前被调用
        """
        if self.dense_layer is not None:
            if hasattr(self.dense_layer, 'extract'):
                # 对于 DenseFeatureLayer 类型，使用其特有的 extract 方法
                self.dense_layer.extract(minibatch, self._column_names)
            else:
                # 对于其他编码器类型，实现通用的特征提取逻辑
                # 首先确保编码器有特征名列表
                if not hasattr(self.dense_layer, 'feature_names'):
                    # 如果编码器没有特征名，从配置文件加载
                    if self.dense_features_path:
                        feature_names = self._load_feature_names_from_path(self.dense_features_path)
                        self.dense_layer.feature_names = feature_names
                    else:
                        print("[WideDeepDense] Error: No feature names available for dense features")
                        return
                
                # 提取 dense 特征数据
                if hasattr(minibatch, 'tensor'):
                    # MiniBatch 对象
                    data = minibatch.tensor
                    # 建立特征索引映射（如果还没有的话）
                    if not hasattr(self.dense_layer, '_feature_indices'):
                        self._build_feature_indices_for_encoder(data)
                    
                    dense_cols = [data[:, idx:idx+1] for idx in self.dense_layer._feature_indices]
                    dense_data = torch.cat(dense_cols, dim=1).float()
                
                elif hasattr(minibatch, '__class__') and minibatch.__class__.__name__ == 'DataFrame':
                    # Pandas DataFrame - 直接按特征名提取
                    dense_values = minibatch[self.dense_layer.feature_names].values
                    dense_data = torch.tensor(dense_values, dtype=torch.float32)
                
                elif isinstance(minibatch, torch.Tensor):
                    # 纯 tensor - 需要通过列名索引
                    if hasattr(self, '_column_names') and self._column_names:
                        # 从全局列名中找到 dense 特征的位置
                        dense_indices = []
                        for name in self.dense_layer.feature_names:
                            if name in self._column_names:
                                idx = self._column_names.index(name)
                                dense_indices.append(idx)
                        
                        dense_data = minibatch[:, dense_indices].float()
                    else:
                        print("[WideDeepDense] Error: No column names available for tensor indexing")
                        return
                
                else:
                    print(f"[WideDeepDense] Warning: Unsupported minibatch type {type(minibatch)}, skipping")
                    return
                
                # 存储 dense 数据供 forward 使用
                self.dense_layer._dense_output = dense_data
    
    def _load_feature_names_from_path(self, path: str) -> list:
        """从配置文件加载特征名列表"""
        feature_names = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name and not name.startswith('#'):
                    feature_names.append(name)
        return feature_names
    
    def _build_feature_indices_for_encoder(self, tensor_data):
        """为编码器构建特征索引映射"""
        # 这个方法只在特殊情况下使用，通常在 set_column_names 时已建立
        if hasattr(self, '_column_names') and self._column_names:
            self.dense_layer._feature_indices = []
            for name in self.dense_layer.feature_names:
                if name in self._column_names:
                    idx = self._column_names.index(name)
                    self.dense_layer._feature_indices.append(idx)
                else:
                    raise ValueError(f"Dense feature '{name}' not found in columns")
        else:
            raise ValueError("Column names not set, cannot build feature indices")
    
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
            # 获取 dense 特征数据
            if hasattr(self.dense_layer, '_dense_output') and self.dense_layer._dense_output is not None:
                # 使用从 minibatch 提取的数据
                dense_raw = self.dense_layer._dense_output
            else:
                # 如果没有提取数据，尝试直接处理 x（这种情况较少见）
                # 对于非 DenseFeatureLayer 的编码器，需要特殊处理
                if hasattr(self.dense_layer, 'feature_names'):
                    # 从 x 中提取对应的 dense 特征（假设 x 包含所有特征）
                    # 这里需要根据具体实现调整
                    dense_raw = x  # 临时实现
                else:
                    dense_raw = x  # 临时实现
            
            # 通过编码器处理
            dense_processed = self.dense_layer(dense_raw)
            
            # 如果有后处理线性层，应用它
            if hasattr(self, 'dense_post_linear') and self.dense_post_linear is not None:
                dense_processed = self.dense_post_linear(dense_processed)
            
            dnn_input = torch.cat([sparse_out, dense_processed], dim=1)
        else:
            dnn_input = sparse_out
        
        # MLP
        dnn_out = self.dnn(dnn_input)
        
        # Wide + Deep 融合
        final_out = torch.add(wide_out, dnn_out) if self.use_wide else dnn_out
        
        return self.final_activation(final_out) if self.final_activation else final_out
