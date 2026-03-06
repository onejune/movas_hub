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

from .layers import MLPLayer

class LRFtrl(torch.nn.Module):
    def __init__(self,
                wide_embedding_dim=1,
                wide_column_name_path=None,
                wide_combine_schema_path=None,
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
        self._embedding_dim = wide_embedding_dim
        self.lr_sparse = ms.EmbeddingSumConcat(self._embedding_dim,
                                            wide_column_name_path,
                                            wide_combine_schema_path)
        self.lr_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.lr_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        self.final_activation = torch.nn.Sigmoid()
        feature_count = self.lr_sparse.feature_count
        feature_dim = self.lr_sparse.feature_count * self._embedding_dim
        self._bn = ms.nn.Normalization(feature_dim, momentum=0.01, eps=1e-5, affine=True)

    def forward(self, x):
        emb = self.lr_sparse(x)
        wide_out = self._bn(emb)
        wide_out = torch.sum(wide_out, dim=1, keepdim=True)
        return self.final_activation(wide_out)

class LRFtrl2(torch.nn.Module):
    def __init__(self,
                wide_embedding_dim=1,
                wide_column_name_path=None,
                wide_combine_schema_path=None,
                adam_learning_rate=0.1
            ):
        super().__init__()
        self._embedding_dim = wide_embedding_dim
        self.lr_sparse = ms.EmbeddingSumConcat(self._embedding_dim,
                                            wide_column_name_path,
                                            wide_combine_schema_path)
        self.lr_sparse.updater = ms.AdamTensorUpdater(adam_learning_rate)
        self.lr_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, x):
        wide_out = self.lr_sparse(x)
        wide_out = torch.sum(wide_out, dim=1, keepdim=True)
        return self.final_activation(wide_out)


class LRFtrl3(torch.nn.Module):
    def __init__(self,
                wide_embedding_dim=1,
                wide_column_name_path=None,
                wide_combine_schema_path=None,
                learning_rate=0.1
            ):
        super().__init__()
        self._embedding_dim = wide_embedding_dim
        self.lr_sparse = ms.EmbeddingSumConcat(self._embedding_dim,
                                            wide_column_name_path,
                                            wide_combine_schema_path)
        self.lr_sparse.updater = ms.AdaGradTensorUpdater(learning_rate, l2=0.5)
        self.lr_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, x):
        wide_out = self.lr_sparse(x)
        wide_out = torch.sum(wide_out, dim=1, keepdim=True)
        return self.final_activation(wide_out)
