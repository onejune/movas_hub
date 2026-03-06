# models/AuxEventFusedModel.py
import torch
import torch.nn as nn
import metaspore as ms
from typing import Dict, List, Optional, Any
from .layers import MLPLayer

class AuxEventFusedModel(torch.nn.Module):
    def __init__(self, 
                 # 统一特征处理
                 column_name_path: str,
                 combine_schema_path: str,
                 embedding_dim: int = 16,
                 # 模型结构参数
                 num_intermediate_events: int = 5,  # 辅助任务数量
                 dnn_hidden_units: List[int] = [256, 128],
                 hidden_activations: str = 'ReLU',
                 net_dropout: float = 0.2,
                 # 预测头参数
                 main_task_hidden_units: List[int] = [64, 32],
                 aux_task_hidden_units: List[int] = [64, 32],
                 time_task_hidden_units: List[int] = [64, 32],
                 uncertainty_hidden_units: List[int] = [32],
                 task_weight_hidden_units: List[int] = [64],
                 # FTRL参数
                 ftrl_l1: float = 1.0,
                 ftrl_l2: float = 120.0,
                 ftrl_alpha: float = 0.5,
                 ftrl_beta: float = 1.0,
                 sparse_init_var: float = 1e-2):
        super().__init__()

        # --- 统一特征嵌入 ---
        self.all_sparse = ms.EmbeddingSumConcat(
            embedding_dim,
            column_name_path,
            combine_schema_path
        )
        self.all_sparse.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta
        )
        self.all_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        self.all_input_dim = int(self.all_sparse.feature_count * embedding_dim)

        # --- 路径融合层 (即主干网络) ---
        self.dnn = MLPLayer(
            input_dim=self.all_input_dim,
            output_dim=dnn_hidden_units[-1],
            hidden_units=dnn_hidden_units[:-1],
            hidden_activations=hidden_activations,
            final_activation=hidden_activations,
            dropout_rates=net_dropout,
            input_norm=True,
            batch_norm=True,
            use_bias=True
        )

        # --- 预测头 ---
        # 主任务预测头：impression -> purchase 预测
        self.main_task_head = MLPLayer(
            input_dim=dnn_hidden_units[-1],
            output_dim=1,
            hidden_units=main_task_hidden_units,
            hidden_activations=hidden_activations,
            final_activation='Sigmoid',  # 输出概率
            dropout_rates=net_dropout,
            input_norm=True,
            batch_norm=True,
            use_bias=True
        )

        # 辅助任务预测头：中间事件预测
        self.num_intermediate_events = num_intermediate_events
        self.auxiliary_heads = nn.ModuleList([
            MLPLayer(
                input_dim=dnn_hidden_units[-1],
                output_dim=1,
                hidden_units=aux_task_hidden_units,
                hidden_activations=hidden_activations,
                final_activation='Sigmoid',  # 输出概率
                dropout_rates=net_dropout,
                input_norm=True,
                batch_norm=True,
                use_bias=True
            ) for _ in range(self.num_intermediate_events)
        ])

        # 转化时间预测头（辅助任务）
        self.time_prediction_head = MLPLayer(
            input_dim=dnn_hidden_units[-1],
            output_dim=1,
            hidden_units=time_task_hidden_units,
            hidden_activations=hidden_activations,
            final_activation=None,  # 回归任务，线性输出
            dropout_rates=net_dropout,
            input_norm=True,
            batch_norm=True,
            use_bias=True
        )

        # 不确定性估计模块 (可选使用)
        self.uncertainty_estimator = MLPLayer(
            input_dim=dnn_hidden_units[-1],
            output_dim=1,
            hidden_units=uncertainty_hidden_units,
            hidden_activations=hidden_activations,
            final_activation='Sigmoid',  # 输出[0,1]范围的不确定性
            dropout_rates=net_dropout,
            input_norm=True,
            batch_norm=True,
            use_bias=True
        )

        # 动态任务权重学习 (可选使用)
        self.task_weight_net = MLPLayer(
            input_dim=dnn_hidden_units[-1],
            output_dim=2 + self.num_intermediate_events,  # 主任务+时间预测+中间事件
            hidden_units=task_weight_hidden_units,
            hidden_activations=hidden_activations,
            final_activation=None,  # 输出原始logits，后续用softmax
            dropout_rates=net_dropout,
            input_norm=True,
            batch_norm=True,
            use_bias=True
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        # 获取统一处理后的特征嵌入
        all_features = self.all_sparse(x)
        all_out = all_features.view(-1, self.all_input_dim)

        # 通过路径融合网络 (主干网络)
        fused_output = self.dnn(all_out)

        # 主任务预测
        main_pred = self.main_task_head(fused_output)

        # 辅助任务预测
        aux_preds = []
        for head in self.auxiliary_heads:
            aux_preds.append(head(fused_output))

        # 时间预测
        time_pred = self.time_prediction_head(fused_output)

        # 不确定性估计
        uncertainty = self.uncertainty_estimator(fused_output)

        # 任务权重
        task_weights_logits = self.task_weight_net(fused_output)
        task_weights = torch.softmax(task_weights_logits, dim=-1)

        return {
            'predictions': {
                'main_task': main_pred,
                'auxiliary_tasks': aux_preds, # List of tensors
                'time_prediction': time_pred
            },
            'uncertainty': uncertainty,
            'task_weights': task_weights
        }

    def predict(self, yhat, bid = None):
        return yhat['main_task']
