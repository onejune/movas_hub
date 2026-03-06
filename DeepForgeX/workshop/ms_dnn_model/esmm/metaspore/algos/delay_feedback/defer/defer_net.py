#
# Copyright 2022 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import metaspore as ms

from ...layers import MLPLayer

class DEFER(torch.nn.Module):# dnn为辅助任务，defer为主任务
    def __init__(self,
                defer_embedding_dim=10,
                deep_embedding_dim=10,
                defer_column_name_path=None,
                defer_combine_schema_path=None,
                deep_column_name_path=None,
                deep_combine_schema_path=None,
                dnn_hidden_units=[1024,512,256,128],
                defer_hidden_units=[64, 32, 16],
                dnn_hidden_activations="ReLU",
                defer_hidden_activations="ReLU",
                net_dropout=0,
                batch_norm=False,
                embedding_regularizer=None,
                net_regularizer=None,
                use_bias=True,
                ftrl_l1=1.0,
                ftrl_l2=120.0,
                ftrl_alpha=0.5,
                ftrl_beta=1.0,
                **kwargs):
        super().__init__()
        self.defer_sparse = ms.EmbeddingSumConcat(defer_embedding_dim,
                                            defer_column_name_path,
                                            defer_combine_schema_path)
        self.defer_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.defer_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        self.defernn = MLPLayer(input_dim = self.defer_sparse.feature_count * defer_embedding_dim,
                                output_dim = 1,
                                hidden_units = defer_hidden_units,
                                hidden_activations = defer_hidden_activations,
                                final_activation = None,
                                dropout_rates = net_dropout,
                                batch_norm = batch_norm,
                                use_bias = use_bias,
                                input_norm = True)
        
        self.dnn_sparse = ms.EmbeddingSumConcat(deep_embedding_dim,
                                           deep_column_name_path,
                                           deep_combine_schema_path)
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
        
        for param in self.dnn.parameters():
            param.requires_grad = False
        for param in self.dnn_sparse.parameters():
            param.requires_grad = False

        self.final_activation1 = torch.nn.Sigmoid()
        self.final_activation2 = torch.nn.Sigmoid()

    def forward(self, x): # dnn为辅助任务，defer为主任务
        with torch.no_grad():
            dnn_out = self.dnn_sparse(x)
            dnn_out = self.dnn(dnn_out)
            dnn_out = self.final_activation1(dnn_out)
        
        defer_out = self.defer_sparse(x)
        defer_out = self.defernn(defer_out)
        defer_out = self.final_activation2(defer_out)

        return defer_out, dnn_out
