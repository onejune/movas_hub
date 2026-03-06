import torch
import metaspore as ms
from .layers import MLPLayer


class FourChannelHidden(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.wc2 = torch.nn.Linear(int(in_size/4), int(in_size/4))
        self.wc3 = torch.nn.Linear(int(in_size), int(in_size - in_size/4 * 3))
        self.w = torch.nn.Linear(int(in_size + int(in_size/4) * 2) + int(in_size - int(in_size/4) * 3) + 3, out_size)
        self.act1 = torch.nn.Tanh()
        self.act = torch.nn.ReLU()
        self.fl = int(in_size/4)
        
    def forward(self, input_tensor, i1, i2, i3):
        f0 = input_tensor[:, :self.fl]
        f1 = input_tensor[:, self.fl:self.fl*2]
        f2 = input_tensor[:, self.fl*2:self.fl*3]
        f3 = input_tensor[:, self.fl*3:]

        c1 = self.act1(f0 * f1) * f1
        c2 = self.act1(self.wc2(f2) * f2)
        c3 = self.act1(f3 * self.wc3(input_tensor))

        s1 = torch.sum(c1, 1, True) + i1
        s2 = torch.sum(c2, 1, True) + i2
        s3 = torch.sum(c3, 1, True) + i3

        return self.act(self.w(torch.cat((input_tensor, c1, c2, c3, s1, s2, s3), 1))), s1, s2, s3


class GateHidden(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_size, out_size)
        self.layer2 = torch.nn.Linear(out_size, out_size)
        self.act1 = torch.nn.PReLU(out_size)
        self.act2 = torch.nn.Sigmoid()

    def forward(self, input_tensor):
        info = self.act1(self.layer1(input_tensor))
        gate = self.act2(self.layer2(info))
        return info * gate


class GateEmbedding(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_size, out_size)
        self.out_size = out_size
        self.act2 = torch.nn.Sigmoid()
    
    def forward(self, input_tensor):
        gate = self.act2(self.layer1(input_tensor))
        gate_reshape = torch.reshape(gate, (gate.shape[0], self.out_size, -1))
        input_reshape = torch.reshape(input_tensor, (input_tensor.shape[0], self.out_size, -1))
        return (gate_reshape * input_reshape).reshape(input_tensor.shape[0], -1)


class FourChannelGateModel(torch.nn.Module):
    def __init__(self, 
                 embedding_dim=16,
                 column_name_path=None,
                 combine_schema_path=None,
                 wide_column_name_path=None,
                 wide_combine_schema_path=None,
                 dnn_hidden_units=[1024, 512, 256],
                 dnn_hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 use_wide=False,
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
        self._embedding_dim = embedding_dim
        self.use_wide = use_wide
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(embedding_dim, wide_column_name_path, wide_combine_schema_path)
            self.lr_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
            feature_count = self.lr_sparse.feature_count
            feature_dim = self.lr_sparse.feature_count * self._embedding_dim 
            self._wideGateEmbedding = GateEmbedding(feature_dim, feature_count)

        self._sparse = ms.EmbeddingSumConcat(
            self._embedding_dim,
            column_name_path,
            combine_schema_path
        )
        self._sparse.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1, 
            l2=ftrl_l2, 
            alpha=ftrl_alpha, 
            beta=ftrl_beta
        )
        self._sparse.initializer = ms.NormalTensorInitializer(var=0.001)
        
        extra_attributes = {
            "enable_fresh_random_keep": True,
            "fresh_dist_range_from": 0,
            "fresh_dist_range_to": 1000,
            "fresh_dist_range_mean": 950,
        }
        self._sparse.extra_attributes = extra_attributes
        
        feature_count = self._sparse.feature_count
        feature_dim = self._sparse.feature_count * self._embedding_dim
        
        self._gateEmbedding = GateEmbedding(feature_dim, feature_count)
        self._h1 = torch.nn.Linear(feature_dim, dnn_hidden_units[0])
        self._h2 = FourChannelHidden(dnn_hidden_units[0], dnn_hidden_units[1])
        self._h3 = FourChannelHidden(dnn_hidden_units[1], dnn_hidden_units[2])
        self._h4 = torch.nn.Linear(dnn_hidden_units[2], 1)
        
        self._bn = ms.nn.Normalization(feature_dim, momentum=0.01, eps=1e-5, affine=True)
        self._zero = torch.zeros(1, 1)
        
        if final_net_activation == 'sigmoid':
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = None

    def forward(self, x):
        if self.use_wide:
            wide_emb = self.lr_sparse(x)
            wide_out = self._wideGateEmbedding(wide_emb)
            wide_out = torch.sum(wide_out, dim=1, keepdim=True)
        emb = self._sparse(x)
        bno = self._bn(emb)
        d = self._gateEmbedding(bno)
        o = self._h1(d)
        r, s1, s2, s3 = self._h2(o, self._zero, self._zero, self._zero)
        r, s1, s2, s3 = self._h3(r, s1, s2, s3)
        dnn_out = self._h4(r)
        
        final_out = torch.add(wide_out, dnn_out) if self.use_wide else dnn_out
        return self.final_activation(final_out) if self.final_activation else final_out

