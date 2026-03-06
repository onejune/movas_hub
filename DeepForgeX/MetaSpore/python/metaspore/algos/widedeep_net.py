
import torch
import metaspore as ms

from .layers import MLPLayer

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