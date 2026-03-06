import torch
import metaspore as ms
import sys

from ..layers import MLPLayer

class MtlSharedBottomModel(torch.nn.Module):
    def __init__(self,
                 embedding_dim=8,
                 column_name_path=None,
                 combine_schema_path=None,
                 task_numb=2,
                 bottom_hidden_units=[512, 256],
                 tower_hidden_units=[32],
                 dnn_activations='ReLU',
                 use_bias=True,
                 input_norm=False,
                 batch_norm=True,
                 net_dropout=None,
                 net_regularizer=None,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0):
        super().__init__()
        self.task_numb = task_numb

        self.sparse = ms.EmbeddingSumConcat(embedding_dim, column_name_path, combine_schema_path)
        self.sparse.updater = ms.FTRLTensorUpdater(l1 = ftrl_l1, l2 = ftrl_l2, alpha = ftrl_alpha, beta = ftrl_beta)
        self.sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        self.input_dim = int(self.sparse.feature_count * embedding_dim)

        # Shared Bottom Layer
        self.bottom_mlp = MLPLayer(input_dim=self.input_dim,
                                   output_dim=bottom_hidden_units[-1] if bottom_hidden_units else self.input_dim,
                                   hidden_units=bottom_hidden_units[:-1] if bottom_hidden_units else [],
                                   hidden_activations=dnn_activations,
                                   final_activation=None,
                                   dropout_rates=net_dropout,
                                   input_norm=input_norm,
                                   batch_norm=batch_norm,
                                   use_bias=use_bias)

        # Task-specific Tower Layers
        self.towers = []
        for i in range(0, self.task_numb):
            mlp = MLPLayer(input_dim=bottom_hidden_units[-1] if bottom_hidden_units else self.input_dim,
                           output_dim=1,
                           hidden_units=tower_hidden_units,
                           hidden_activations=dnn_activations,
                           final_activation='Sigmoid',
                           dropout_rates=net_dropout,
                           input_norm=input_norm,
                           batch_norm=batch_norm,
                           use_bias=use_bias)
            self.towers.append(mlp)

        # 初始化 _current_task_ids 属性
        self._current_task_ids = None

    def forward(self, x):
        x = self.sparse(x)
        
        # Shared Bottom Layer
        bottom_out = self.bottom_mlp(x)
        
        # Task-specific Tower Layers
        predictions = []
        for i in range(0, self.task_numb):
            tower_out = self.towers[i](bottom_out)
            predictions.append(tower_out)
        # 拼接所有任务的预测结果 [batch_size, task_numb]
        prediction = torch.cat(predictions, dim=1)
        
        # 根据是否存在 _current_task_ids 决定返回的内容
        if hasattr(self, '_current_task_ids') and self._current_task_ids is not None:
            # 如果存在 _current_task_ids，则返回对应 task 的预测结果
            batch_size = prediction.size(0)
            selected_predictions = torch.zeros(batch_size, 1, device=prediction.device)
            
            for idx, task_id in enumerate(self._current_task_ids):
                selected_predictions[idx, 0] = prediction[idx, task_id]
            self._current_task_ids = None
            return selected_predictions
        else:
            # 如果不存在 _current_task_ids，则返回所有 task 的预测结果
            return prediction

    def predict(self, yhat, minibatch = None):
        #返回第一个任务的预测结果
        return yhat[:, 0]