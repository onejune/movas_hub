import torch
import torch.nn as nn
import metaspore as ms
from typing import List, Optional

from ..layers import MLPLayer

class STEMNet(nn.Module):
    """
    STEM模型：Shared and Task Embeddings Model
    通过解耦共享嵌入和任务特定嵌入来解决多任务学习中的特征冲突
    """
    
    def __init__(self,
                 embedding_dim: int = 64,
                 column_name_path: Optional[str] = None,
                 combine_schema_path: Optional[str] = None,
                 task_count: int = 2,  
                 shared_hidden_units: List[int] = None,
                 tower_hidden_units: List[int] = None,
                 dnn_activation: str = 'ReLU',
                 use_bias: bool = True,
                 input_norm: bool = False,
                 batch_norm: bool = False,
                 dropout_rate: Optional[float] = None,
                 sparse_init_var: float = 1e-2,
                 ftrl_l1: float = 1.0,
                 ftrl_l2: float = 120.0,
                 ftrl_alpha: float = 0.5,
                 ftrl_beta: float = 1.0,
                 use_uncertainty_weighting: bool = False):
        super().__init__()
        
        # 初始化参数
        self.embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path
        self.task_count = task_count  # 修改：使用task_count
        self.task_names = [f'task_{i}' for i in range(task_count)]  # 自动生成任务名称
        self.use_uncertainty_weighting = use_uncertainty_weighting  # 添加自适应任务权重开关

        # 稀疏特征嵌入层 - 共享嵌入
        self.shared_sparse = ms.EmbeddingSumConcat(embedding_dim, column_name_path, combine_schema_path)
        
        # 配置共享稀疏层参数
        self.shared_sparse.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1,
            l2=ftrl_l2, 
            alpha=ftrl_alpha,
            beta=ftrl_beta
        )
        self.shared_sparse.initializer = ms.NormalTensorInitializer(
            var=sparse_init_var
        )
        
        # 任务特定稀疏特征嵌入层
        self.task_sparse_layers = nn.ModuleDict()
        for i in range(self.task_count):  # 修改：使用task_count
            task_sparse = ms.EmbeddingSumConcat(embedding_dim, column_name_path, combine_schema_path)
            
            # 配置任务特定稀疏层参数
            task_sparse.updater = ms.FTRLTensorUpdater(
                l1=ftrl_l1,
                l2=ftrl_l2, 
                alpha=ftrl_alpha,
                beta=ftrl_beta
            )
            task_sparse.initializer = ms.NormalTensorInitializer(
                var=sparse_init_var
            )
            
            self.task_sparse_layers[f'task_{i}'] = task_sparse  # 修改：使用索引命名
        
        # 计算输入维度 (共享嵌入 + 任务特定嵌入)
        self.input_dim = int(self.shared_sparse.feature_count * self.embedding_dim * 2)
        
        # 共享MLP层
        self.shared_mlp = MLPLayer(
            input_dim=self.input_dim,
            output_dim=shared_hidden_units[-1] if shared_hidden_units else 64,
            hidden_units=shared_hidden_units or [256, 128],
            hidden_activations=dnn_activation,
            final_activation=dnn_activation,
            dropout_rates=dropout_rate,
            input_norm=input_norm,
            batch_norm=batch_norm,
            use_bias=use_bias
        )
        
        # 任务特定塔网络
        self.tower_layers = nn.ModuleList()
        for i in range(self.task_count):  # 修改：使用task_count
            tower_mlp = MLPLayer(
                input_dim=shared_hidden_units[-1] if shared_hidden_units else 64,
                output_dim=1,
                hidden_units=tower_hidden_units or [64],
                hidden_activations=dnn_activation,
                final_activation=None,  # 移除激活函数，后续统一处理
                dropout_rates=dropout_rate,
                input_norm=input_norm,
                batch_norm=batch_norm,
                use_bias=use_bias
            )
            self.tower_layers.append(tower_mlp)

        # --- 自适应任务权重 ---
        initial_log_var_value = 0.0
        if self.use_uncertainty_weighting:
            # 不确定性权重参数 - 作为可训练参数
            self.register_parameter('log_vars', torch.nn.Parameter(torch.full((self.task_count,), initial_log_var_value)))
        
        # 最终激活函数
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, x):
        # 获取共享嵌入
        shared_embeddings = self.shared_sparse(x)
        
        # 多任务预测
        task_predictions = []
        for i in range(self.task_count):  # 修改：使用task_count
            task_key = f'task_{i}'
            # 获取任务特定嵌入
            task_embeddings = self.task_sparse_layers[task_key](x)
            
            # 拼接共享嵌入和任务特定嵌入
            combined_embeddings = torch.cat([shared_embeddings, task_embeddings], dim=1)
            
            # 通过共享MLP
            shared_output = self.shared_mlp(combined_embeddings)
            
            # 通过任务特定塔
            task_pred = self.tower_layers[i](shared_output)
            task_predictions.append(task_pred)
        
        # 合并所有任务预测 [batch_size, task_num, 1] -> [batch_size, task_num]
        logits = torch.cat(task_predictions, dim=1)
        
        # 最终激活
        final_prediction = self.final_activation(logits)
        return final_prediction

    def get_task_weights(self):
        """获取当前任务权重（用于监控训练过程）"""
        if self.use_uncertainty_weighting:
            weights = []
            for i in range(self.task_count):
                weight = torch.exp(-self.log_vars[i])
                weights.append(weight)
            return [w.item() for w in weights]
        else:
            return [1.0 for _ in range(self.task_count)]

    def get_task_uncertainties(self):
        """获取当前任务不确定性（log_var值）"""
        if self.use_uncertainty_weighting:
            uncertainties = []
            for i in range(self.task_count):
                uncertainties.append(self.log_vars[i])
            return [u.item() for u in uncertainties]
        else:
            return [0.0 for _ in range(self.task_count)]

    def predict(self, yhat, minibatch=None):
        # Return the prediction for the first task (index 0) as an example
        return yhat[:, 0] # [B]

