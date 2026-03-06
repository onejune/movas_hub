import torch
import metaspore as ms
import sys

from ...layers import MLPLayer

class SceneAwareMMoE(torch.nn.Module):
    """
    Scene-Aware MMoE (SceneAwareMMoE) 网络
    
    基于原生MMoE的改进版本，引入了可配置的场景特征作为门控网络的输入，
    使得门控权重的计算不仅依赖于输入特征，还依赖于当前的场景信息。
    场景特征通过独立的combine_schema配置，可以灵活配置不同的场景特征。
    """
    def __init__(self,
                 embedding_dim=10,
                 column_name_path=None,
                 combine_schema_path=None,
                 scene_column_name_path=None,
                 scene_combine_schema_path=None,
                 expert_numb=2,
                 task_numb=2,
                 expert_hidden_units=[],
                 expert_out_dim=10,
                 gate_hidden_units=[],
                 tower_hidden_units=[],
                 dnn_activations='ReLU',
                 use_bias=True,
                 input_norm=False,
                 batch_norm=False,
                 net_dropout=None,
                 net_regularizer=None,
                 sparse_init_var=1e-2,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0):
        """
        初始化SceneAwareMMoE网络
        
        Args:
            scene_column_name_path: 场景特征列名文件路径
            scene_combine_schema_path: 场景特征组合模式文件路径
        """
        super().__init__()
        self.expert_numb = expert_numb
        self.task_numb = task_numb
        self.expert_out_dim = expert_out_dim

        # 原始特征嵌入部分
        self.embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path
        self.sparse = ms.EmbeddingSumConcat(self.embedding_dim, self.column_name_path, self.combine_schema_path)
        self.sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, 
                                                    l2=ftrl_l2, 
                                                    alpha=ftrl_alpha, 
                                                    beta=ftrl_beta)
        self.sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        self.input_dim = int(self.sparse.feature_count * self.embedding_dim)

        # 场景特征嵌入部分
        self.scene_column_name_path = scene_column_name_path
        self.scene_combine_schema_path = scene_combine_schema_path
        if self.scene_combine_schema_path is not None:
            self.scene_sparse = ms.EmbeddingSumConcat(self.embedding_dim, 
                                                    self.scene_column_name_path, 
                                                    self.scene_combine_schema_path)
            self.scene_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, 
                                                              l2=ftrl_l2, 
                                                              alpha=ftrl_alpha, 
                                                              beta=ftrl_beta)
            self.scene_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
            self.scene_input_dim = int(self.scene_sparse.feature_count * self.embedding_dim)
        else:
            self.scene_sparse = None
            self.scene_input_dim = self.embedding_dim  # 默认场景嵌入维度

        # 专家网络
        self.experts = nn.ModuleList()
        for i in range(0, self.expert_numb):
            mlp = MLPLayer(input_dim=self.input_dim,
                           output_dim=self.expert_out_dim,
                           hidden_units=expert_hidden_units,
                           hidden_activations=dnn_activations,
                           final_activation=None,
                           dropout_rates=net_dropout,
                           input_norm=input_norm,
                           batch_norm=batch_norm,
                           use_bias=use_bias)
            self.experts.append(mlp)

        # 门控网络（使用场景特征作为输入）
        self.gates = nn.ModuleList()
        for i in range(0, self.task_numb):
            # 门控网络的输入是场景特征
            mlp = MLPLayer(input_dim=self.scene_input_dim if self.scene_sparse else self.embedding_dim,
                           output_dim=self.expert_numb,
                           hidden_units=gate_hidden_units,
                           hidden_activations=dnn_activations,
                           final_activation=None,
                           dropout_rates=net_dropout,
                           input_norm=input_norm,
                           batch_norm=batch_norm,
                           use_bias=use_bias)
            self.gates.append(mlp)
        self.gate_softmax = torch.nn.Softmax(dim=1)

        # 塔网络
        self.towers = nn.ModuleList()
        for i in range(0, self.task_numb):
            mlp = MLPLayer(input_dim=self.expert_out_dim,
                           output_dim=1,
                           hidden_units=tower_hidden_units,
                           hidden_activations=dnn_activations,
                           final_activation='Sigmoid',
                           dropout_rates=net_dropout,
                           input_norm=input_norm,
                           batch_norm=batch_norm,
                           use_bias=use_bias)
            self.towers.append(mlp)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 原始输入特征
            scene_x: 场景特征输入（可选，如果未提供则使用默认处理）
        """
        # 原始特征处理
        x_sparse = self.sparse(x)
        
        # 场景特征处理
        scene_emb = self.scene_sparse(x)
        
        # 专家网络处理
        expert_outputs = []
        for i in range(0, self.expert_numb):
            expert_out = self.experts[i](x_sparse)
            expert_outputs.append(expert_out)
        expert_cat = torch.cat(expert_outputs, dim=1)
        expert_cat = expert_cat.reshape(-1, self.expert_numb, self.expert_out_dim)

        # 门控网络处理（使用场景特征）
        predictions = []
        for i in range(0, self.task_numb):
            gate_out = self.gates[i](scene_emb)  # 使用场景特征作为门控输入
            gate_out = self.gate_softmax(gate_out)
            gate_out = gate_out.reshape(-1, self.expert_numb, 1)
            tower_input = torch.mul(expert_cat, gate_out)
            tower_input = torch.sum(tower_input, 1)
            tower_out = self.towers[i](tower_input)
            predictions.append(tower_out)
        
        prediction = torch.cat(predictions, dim=1)
        return prediction

    def predict(self, yhat, bid = None):
        #返回第一个任务的预测结果
        return yhat[:, 0]