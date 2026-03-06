import torch
import torch.nn as nn
import torch.nn.functional as F
import metaspore as ms
from typing import Dict, List, Optional
from .layers import MLPLayer  # 假设MLPLayer在同目录下的layers模块中


class EventSequenceEncoder(nn.Module):
    """
    事件序列编码器，使用Transformer架构编码用户转化路径
    """
    def __init__(self, 
                 embed_dim: int,
                 max_seq_len: int = 10,
                 num_heads: int = 8,
                 num_layers: int = 2):
        super().__init__()
        
        # 注意：由于嵌入已由MetaSpore处理，这里不再需要event_embedding层
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.embed_dim = embed_dim
        
    def forward(self, event_embeds, mask=None):
        """
        Args:
            event_embeds: [batch_size, seq_len, embed_dim] 事件嵌入序列
            mask: [batch_size, seq_len] 事件掩码
        """
        batch_size, seq_len, embed_dim = event_embeds.shape
        
        # 位置嵌入
        positions = torch.arange(seq_len).expand(batch_size, seq_len).to(event_embeds.device)
        pos_embeds = self.position_embedding(positions)
        
        x = event_embeds + pos_embeds
        
        # 应用掩码
        if mask is not None:
            mask = ~mask.bool()  # True表示忽略的位置
            
        # Transformer编码
        output = self.transformer(x, src_key_padding_mask=mask)
        
        # 序列级表示（使用最后一个有效位置的表示）
        if mask is not None:
            valid_lengths = (~mask).sum(dim=1)  # [batch_size]
            batch_indices = torch.arange(batch_size).to(event_embeds.device)
            last_valid_pos = valid_lengths - 1
            sequence_repr = output[batch_indices, last_valid_pos]  # [batch_size, embed_dim]
        else:
            sequence_repr = output[:, -1, :]  # [batch_size, embed_dim]
            
        return sequence_repr


class PathFusionNetwork(torch.nn.Module):
    """
    基于MetaSpore的PathFusion Network (PFN) - 电商广告IVR建模主模型
    融合用户行为路径、广告主特征路径和上下文特征路径
    """
    def __init__(self, 
                 event_embed_dim: int,
                 advertiser_embed_dim: int,
                 context_embed_dim: int,
                 event_column_name_path: str,
                 event_combine_schema_path: str,
                 advertiser_column_name_path: str,
                 advertiser_combine_schema_path: str,
                 context_column_name_path: str,
                 context_combine_schema_path: str,
                 feature_dim: int,
                 num_intermediate_events: int = 5,
                 # 事件序列编码器参数
                 event_seq_max_len: int = 10,
                 event_seq_num_heads: int = 8,
                 event_seq_num_layers: int = 2,
                 # 其他网络参数
                 path_fusion_hidden_units: List[int] = [256, 128],
                 path_fusion_activation: str = 'ReLU',
                 path_fusion_dropout: float = 0.2,
                 # FTRL参数
                 ftrl_l1: float = 1.0,
                 ftrl_l2: float = 120.0,
                 ftrl_alpha: float = 0.5,
                 ftrl_beta: float = 1.0,
                 sparse_init_var: float = 1e-2):
        super().__init__()
        
        # 事件序列特征处理（使用EmbeddingLookup以支持序列）
        self.event_embedding_table = ms.EmbeddingLookup(
            event_embed_dim,
            event_column_name_path,
            event_combine_schema_path
        )
        self.event_embedding_table.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta
        )
        self.event_embedding_table.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        self.event_feature_nums = self.event_embedding_table.feature_count # 每个样本的事件序列长度（假设固定）
        
        # 事件序列编码器（嵌入已由MetaSpore处理）
        self.event_encoder = EventSequenceEncoder(
            embed_dim=event_embed_dim,
            max_seq_len=event_seq_max_len,
            num_heads=event_seq_num_heads,
            num_layers=event_seq_num_layers
        )
        
        # MetaSpore广告主特征嵌入
        self.advertiser_sparse = ms.EmbeddingSumConcat(
            advertiser_embed_dim,
            advertiser_column_name_path,
            advertiser_combine_schema_path
        )
        self.advertiser_sparse.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta
        )
        self.advertiser_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        self.advertiser_input_dim = int(self.advertiser_sparse.feature_count * advertiser_embed_dim)

        # MetaSpore上下文特征嵌入
        self.context_sparse = ms.EmbeddingSumConcat(
            context_embed_dim,
            context_column_name_path,
            context_combine_schema_path
        )
        self.context_sparse.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta
        )
        self.context_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        self.context_input_dim = int(self.context_sparse.feature_count * context_embed_dim)

        # 路径融合层 - 将事件序列特征、广告主特征和上下文特征融合
        fusion_input_dim = event_embed_dim + self.advertiser_input_dim + self.context_input_dim
        self.path_fusion = MLPLayer(
            input_dim=fusion_input_dim,
            output_dim=path_fusion_hidden_units[-1],  # 最后一层输出维度
            hidden_units=path_fusion_hidden_units[:-1],  # 除最后一层外的所有层
            hidden_activations=path_fusion_activation,
            final_activation=path_fusion_activation,
            dropout_rates=path_fusion_dropout,
            input_norm=True,
            batch_norm=True,
            use_bias=True
        )

        # 主任务预测头：impression -> purchase 预测
        self.main_task_head = MLPLayer(
            input_dim=path_fusion_hidden_units[-1],
            output_dim=1,
            hidden_units=[64, 32],
            hidden_activations=path_fusion_activation,
            final_activation='Sigmoid',  # 输出概率
            dropout_rates=path_fusion_dropout,
            input_norm=True,
            batch_norm=True,
            use_bias=True
        )

        # 辅助任务预测头：中间事件预测
        self.auxiliary_heads = nn.ModuleList([
            MLPLayer(
                input_dim=path_fusion_hidden_units[-1],
                output_dim=1,
                hidden_units=[64, 32],
                hidden_activations=path_fusion_activation,
                final_activation='Sigmoid',  # 输出概率
                dropout_rates=path_fusion_dropout,
                input_norm=True,
                batch_norm=True,
                use_bias=True
            ) for _ in range(num_intermediate_events)
        ])

        # 转化时间预测头（辅助任务）
        self.time_prediction_head = MLPLayer(
            input_dim=path_fusion_hidden_units[-1],
            output_dim=1,
            hidden_units=[64, 32],
            hidden_activations=path_fusion_activation,
            final_activation=None,  # 回归任务，线性输出
            dropout_rates=path_fusion_dropout,
            input_norm=True,
            batch_norm=True,
            use_bias=True
        )

        # 不确定性估计模块
        self.uncertainty_estimator = MLPLayer(
            input_dim=path_fusion_hidden_units[-1],
            output_dim=1,
            hidden_units=[32],
            hidden_activations=path_fusion_activation,
            final_activation='Sigmoid',  # 输出[0,1]范围的不确定性
            dropout_rates=path_fusion_dropout,
            input_norm=True,
            batch_norm=True,
            use_bias=True
        )

        # 动态任务权重学习
        self.task_weight_net = MLPLayer(
            input_dim=path_fusion_hidden_units[-1],
            output_dim=2 + num_intermediate_events,  # 主任务+时间预测+中间事件
            hidden_units=[64],
            hidden_activations=path_fusion_activation,
            final_activation=None,  # 输出原始logits，后续用softmax
            dropout_rates=path_fusion_dropout,
            input_norm=True,
            batch_norm=True,
            use_bias=True
        )

    def get_field_embedding_list(self, x, offset):
        """
        将一维特征张量按字段分组
        """
        x_reshape = [x[offset[i]:offset[i+1],:] for i in range(offset.shape[0]-1)]
        x_reshape.append(x[offset[offset.shape[0]-1]:x.shape[0],:])
        return x_reshape

    def get_seq_column_embedding(self, seq_column_index_list, x_reshape, column_nums):
        """
        提取序列列的嵌入
        """
        all_column_embedding = []
        for column_index in seq_column_index_list:
            column_embedding = x_reshape[column_index::column_nums]
            column_embedding = torch.nn.utils.rnn.pad_sequence(column_embedding, batch_first=True)
            all_column_embedding.append(column_embedding)
        all_column_embedding = torch.cat(all_column_embedding, dim=2)
        return all_column_embedding

    def do_extra_work(self, minibatch):
        """
        从minibatch中提取事件序列掩码
        假设掩码作为dense特征 'event_masks' 存在于 minibatch 中
        形状为 [batch_size * max_seq_len] 的一维张量，需要reshape
        """
        raw_masks = minibatch.get('event_masks', None)
        if raw_masks is not None:
            # 假设 raw_masks 是一个展平的一维张量 [B * S]
            # 需要reshape成 [B, S]
            # MetaSpore的dense特征通常是展平的，按样本排列
            # 但我们这里假设它已经被正确地作为 [B*S] 的张量传递进来
            # 或者，如果它是 [B, S] 的张量，则无需reshape
            # 为了安全，我们尝试reshape
            try:
                # minibatch中的dense特征有时是展平的(batch_size, -1)
                # 但event_masks如果是dense且shape明确，可能是(batch_size, seq_len)
                # 这里假设传入的是展平的 [B*S] 张量
                batch_size_from_masks = raw_masks.numel() // self.event_feature_nums
                self.event_masks = raw_masks.view(batch_size_from_masks, self.event_feature_nums)
            except RuntimeError as e:
                print(f"Warning: Failed to reshape event_masks: {e}")
                print(f"raw_masks shape: {raw_masks.shape}, expected total elements: {raw_masks.numel()}, "
                    f"calculated batch_size: {raw_masks.numel() // self.event_feature_nums if self.event_feature_nums > 0 else 'N/A'}")
                # 或者直接赋值，假设它已经是正确的形状
                self.event_masks = raw_masks
        else:
            # 如果没有提供掩码，默认全部为1（无填充）
            # 需要在知道batch_size后才能初始化
            pass # 在forward中处理

    def forward(self, x):
        """
        Args:
            x: MetaSpore输入特征张量
        """
        # --- 事件序列嵌入处理 ---
        event_x, event_offset = self.event_embedding_table(x)
        
        # 计算 batch_size
        # event_offset 的长度是 batch_size * seq_len + 1 (因为每个样本有seq_len个特征)
        # 所以 batch_size = (len(event_offset) - 1) / seq_len
        # 由于 event_feature_nums = max_seq_len
        batch_size = (event_offset.shape[0] - 1) // self.event_feature_nums
        
        # 初始化 event_masks 如果 do_extra_work 没有提供
        if not hasattr(self, 'event_masks') or self.event_masks is None:
            # 默认无填充
            self.event_masks = torch.ones((batch_size, self.event_feature_nums), dtype=torch.bool, device=event_x.device)
        elif self.event_masks.shape[0] != batch_size:
            # 检查batch_size是否匹配
            print(f"Warning: event_masks batch size ({self.event_masks.shape[0]}) does not match calculated batch size ({batch_size}). Using default masks.")
            self.event_masks = torch.ones((batch_size, self.event_feature_nums), dtype=torch.bool, device=event_x.device)

        # 将 event_x 重塑为 [batch_size, max_seq_len, event_embed_dim]
        # event_x 的 shape 是 [total_features, embed_dim] = [B * S, D]
        # 我们需要把它变成 [B, S, D]
        # MetaSpore 的 EmbeddingLookup 输出是按样本顺序排列的特征
        # 即 [sample0_feat0, sample0_feat1, ..., sample0_feat(S-1), sample1_feat0, ...]
        event_embeds_tensor = event_x.view(batch_size, self.event_feature_nums, -1) # -1 会自动推断为 embed_dim

        # --- 广告主和上下文特征处理 ---
        advertiser_features = self.advertiser_sparse(x)
        advertiser_out = advertiser_features.view(-1, self.advertiser_input_dim)
        
        context_features = self.context_sparse(x)
        context_out = context_features.view(-1, self.context_input_dim)
        
        # --- 事件序列编码 ---
        # 使用事件序列编码器处理嵌入后的序列和掩码
        event_sequence_features = self.event_encoder(event_embeds_tensor, self.event_masks) # 注意这里传入了mask
        
        # --- 特征融合与预测 ---
        fused_features = torch.cat([event_sequence_features, advertiser_out, context_out], dim=-1)
        fused_output = self.path_fusion(fused_features)
        
        main_pred = self.main_task_head(fused_output)
        aux_preds = [head(fused_output) for head in self.auxiliary_heads]
        time_pred = self.time_prediction_head(fused_output)
        uncertainty = self.uncertainty_estimator(fused_output)
        task_weights_logits = self.task_weight_net(fused_output)
        task_weights = torch.softmax(task_weights_logits, dim=-1)
        
        # 清理临时存储的mask，为下一个batch做准备
        if hasattr(self, 'event_masks'):
            delattr(self, 'event_masks')
            
        return {
            'predictions': {
                'main_task': main_pred,
                'auxiliary_tasks': aux_preds,
                'time_prediction': time_pred
            },
            'uncertainty': uncertainty,
            'task_weights': task_weights
        }


    def predict_main_task(self, x):
        """
        专门用于预测主任务（IVR）的接口
        """
        outputs = self.forward(x)
        return outputs['predictions']['main_task']

    def predict_all_tasks(self, x):
        """
        预测所有任务的接口
        """
        return self.forward(x)



