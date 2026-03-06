import torch
import torch.nn.functional as F
import metaspore as ms
from ..layers import MLPLayer

# AdaSparse: Learning Adaptively Sparse Structures for Multi-Domain Click-Through Rate Prediction

class DomainPruner(torch.nn.Module):
    """
    域感知裁剪器
    """
    def __init__(self, input_dim, strategy="fusion", reduction_factor=4):
        super().__init__()
        self.strategy = strategy
        
        # 裁剪权重生成网络
        self.pruning_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim // reduction_factor),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim // reduction_factor, input_dim),
            torch.nn.Sigmoid()
        )
        
        # 软阈值参数
        #self.threshold = torch.nn.Parameter(torch.tensor(0.5))
        self.register_buffer('threshold', torch.tensor(0.5))
        
    def forward(self, x):
        # x: [batch_size, input_dim]
        pruning_weights = self.pruning_net(x)  # [batch_size, input_dim]
        
        if self.strategy == "binarization":
            # 二值化：硬裁剪
            mask = (pruning_weights > self.threshold).float()
            return x * mask
        elif self.strategy == "scaling":
            # 缩放：软加权
            return x * pruning_weights
        elif self.strategy == "fusion":
            # 混合：结合缩放和二值化
            scaled_x = x * pruning_weights
            mask = (pruning_weights > self.threshold).float()
            return scaled_x * mask
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


class AdaSparseMDLModel(torch.nn.Module):
    def __init__(self,
                 num_domains=2,
                 embedding_dim=10,
                 column_name_path=None,
                 combine_schema_path=None,
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
                 sparse_strategy="fusion",  # "binarization", "scaling", "fusion"
                 sparse_reg_weight=0.1,
                 **kwargs):
        super(AdaSparseMDLModel, self).__init__()
        
        self.num_domains = num_domains
        self.sparse_strategy = sparse_strategy
        self.sparse_reg_weight = sparse_reg_weight
        self.ftrl_l1 = ftrl_l1
        self.ftrl_l2 = ftrl_l2
        self.ftrl_alpha = ftrl_alpha
        self.ftrl_beta = ftrl_beta
        
        # 共享嵌入层 - 使用FTRL优化器
        self.embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path
        self.embedding_layer = ms.EmbeddingSumConcat(self.embedding_dim, self.column_name_path, self.combine_schema_path)
        self.embedding_layer.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.embedding_layer.initializer = ms.NormalTensorInitializer(var=0.01)
        self.input_dim = int(self.embedding_layer.feature_count * self.embedding_dim)
        
        assert self.input_dim > 0, f"input_dim={self.input_dim}, check your schema files!"
        #print(f"[INFO] input_dim = {self.input_dim}")

        # 域感知裁剪器 - 这些是密集参数，使用Adam优化器
        self.domain_pruners = torch.nn.ModuleList([
            DomainPruner(
                input_dim=self.input_dim,
                strategy=sparse_strategy
            ) for _ in range(num_domains)
        ])
        
        self.dnn = MLPLayer(input_dim=self.input_dim,
                           output_dim=1,
                           hidden_units=dnn_hidden_units,
                           hidden_activations=dnn_hidden_activations,
                           final_activation=None,
                           dropout_rates=net_dropout,
                           batch_norm=batch_norm,
                           use_bias=use_bias,
                           input_norm=False)
        
        if final_net_activation == 'sigmoid':
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = None
            
        # 用于存储当前批次的域ID
        self._current_domain_ids = None

    def do_extra_work(self, minibatch):
        # 假设 minibatch 是 pandas DataFrame 或 dict of tensors
        if isinstance(minibatch, dict):
            domain_id_col = minibatch['domain_id']
        elif hasattr(minibatch, 'domain_id'):  # 如 Spark Row 或 namedtuple
            domain_id_col = minibatch.domain_id
        else:
            # 如果是 pandas DataFrame
            domain_id_col = minibatch['domain_id'].values  # 或 .to_numpy()
        # 转为 LongTensor（PyTorch 推荐索引用 long/int64）
        if not isinstance(domain_id_col, torch.Tensor):
            domain_id_tensor = torch.tensor(domain_id_col, dtype=torch.long)
        else:
            domain_id_tensor = domain_id_col.long()
        #print('do_extra_work-domain_id_tensor:', domain_id_tensor)
        self._current_domain_ids = domain_id_tensor  # 保存到模型内部

    def forward(self, x):
        # 获取域ID
        domain_id = self._current_domain_ids
        
        # 共享嵌入
        embeddings = self.embedding_layer(x)  # [batch_size, feature_count * embedding_dim]
        
        # 根据域ID选择对应的裁剪器
        batch_size = embeddings.size(0)
        sparse_reg_loss = 0.0  # 累积稀疏正则项
        
        # 获取所有 unique domains in batch
        unique_domains = torch.unique(domain_id)
        outputs = torch.zeros(batch_size, 1, device=embeddings.device)

        for d in unique_domains:
            mask = (domain_id == d)
            indices = torch.where(mask)[0]
            emb_batch = embeddings[mask]  # [n_d, input_dim]
            pruner = self.domain_pruners[min(d.item(), self.num_domains - 1)]
            sparse_emb = pruner(emb_batch)
            
            # 计算 sparse reg loss for this domain
            weights = pruner.pruning_net(emb_batch)
            if self.sparse_strategy == "binarization":
                loss = torch.mean(torch.abs(torch.abs(weights - 0.5) - 0.5))
            else:
                loss = torch.mean(torch.abs(weights))
            sparse_reg_loss += loss * emb_batch.size(0)  # weighted by count

            dnn_out = self.dnn(sparse_emb)
            outputs[indices] = dnn_out

        self.sparse_reg_loss = sparse_reg_loss / batch_size
        
        final_out = outputs
        return self.final_activation(final_out) if self.final_activation else final_out

    def compute_loss(self, predictions, labels, minibatch, **kwargs):
        """Compute total loss including sparse regularization."""
        if torch.isnan(predictions).any():
            print("WARNING: NaN detected in model predictions during loss calculation.")
            predictions = torch.nan_to_num(predictions, nan=0.5, posinf=1.0 - 1e-7, neginf=1e-7)
        predictions = torch.clamp(predictions, min=1e-7, max=1 - 1e-7)
        main_loss = F.binary_cross_entropy(predictions, labels.float(), reduction = 'mean')
        total_loss = main_loss + self.sparse_reg_weight * self.sparse_reg_loss
        
        return total_loss, main_loss

