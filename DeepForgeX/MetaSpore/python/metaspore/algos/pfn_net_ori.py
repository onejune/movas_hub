import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import logging

class EventSequenceEncoder(nn.Module):
    """
    事件序列编码器，使用Transformer架构编码用户转化路径
    """
    def __init__(self, 
                 event_vocab_size: int,
                 embed_dim: int,
                 max_seq_len: int = 10,
                 num_heads: int = 8,
                 num_layers: int = 2):
        super().__init__()
        
        self.event_embedding = nn.Embedding(event_vocab_size, embed_dim)
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
        
    def forward(self, event_sequences, mask=None):
        """
        Args:
            event_sequences: [batch_size, seq_len] 事件序列
            mask: [batch_size, seq_len] 事件掩码
        """
        batch_size, seq_len = event_sequences.shape
        
        # 事件嵌入 + 位置嵌入
        event_embeds = self.event_embedding(event_sequences)
        positions = torch.arange(seq_len).expand(batch_size, seq_len).to(event_sequences.device)
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
            batch_indices = torch.arange(batch_size).to(event_sequences.device)
            last_valid_pos = valid_lengths - 1
            sequence_repr = output[batch_indices, last_valid_pos]  # [batch_size, embed_dim]
        else:
            sequence_repr = output[:, -1, :]  # [batch_size, embed_dim]
            
        return sequence_repr

class AdvertiserTypeEncoder(nn.Module):
    """
    广告主类型编码器，编码广告主的回传能力特征
    """
    def __init__(self, 
                 advertiser_type_vocab_size: int,
                 embed_dim: int):
        super().__init__()
        self.advertiser_embedding = nn.Embedding(advertiser_type_vocab_size, embed_dim)
        
    def forward(self, advertiser_types):
        return self.advertiser_embedding(advertiser_types)

class PathFusionLayer(nn.Module):
    """
    路径融合层，融合用户行为路径、广告主特征路径和上下文特征路径
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fusion_network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, fused_features):
        return self.fusion_network(fused_features)

class MultiTaskPredictor(nn.Module):
    """
    多任务预测器，包含主任务和辅助任务
    """
    def __init__(self, input_dim: int, num_intermediate_events: int):
        super().__init__()
        
        # 主任务：impression -> purchase 预测
        self.main_task_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 辅助任务：中间事件预测
        self.auxiliary_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(input_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_intermediate_events)
        ])
        
        # 转化时间预测（辅助任务）
        self.time_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, 1)
        )
        
    def forward(self, features):
        # 主任务预测
        main_pred = self.main_task_predictor(features)
        
        # 辅助任务预测
        aux_preds = []
        for predictor in self.auxiliary_predictors:
            aux_preds.append(predictor(features))
            
        # 时间预测
        time_pred = self.time_predictor(features)
        
        return {
            'main_task': main_pred,
            'auxiliary_tasks': aux_preds,
            'time_prediction': time_pred
        }

class UncertaintyEstimator(nn.Module):
    """
    不确定性估计模块，评估预测的可信度
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        return self.uncertainty_net(features)

class PathFusionNetwork(nn.Module):
    """
    PathFusion Network (PFN) - 电商广告IVR建模主模型
    融合用户行为路径、广告主特征路径和上下文特征路径
    """
    def __init__(self, 
                 event_vocab_size: int,
                 advertiser_type_vocab_size: int,
                 feature_dim: int,
                 embed_dim: int = 128,
                 num_intermediate_events: int = 5):
        super().__init__()
        
        # 用户行为路径编码器
        self.event_encoder = EventSequenceEncoder(
            event_vocab_size=event_vocab_size,
            embed_dim=embed_dim
        )
        
        # 广告主特征路径编码器
        self.advertiser_encoder = AdvertiserTypeEncoder(
            advertiser_type_vocab_size=advertiser_type_vocab_size,
            embed_dim=embed_dim
        )
        
        # 路径融合层
        fusion_input_dim = embed_dim * 2 + feature_dim  # 事件特征 + 广告主特征 + 额外特征
        self.path_fusion = PathFusionLayer(
            input_dim=fusion_input_dim,
            output_dim=embed_dim
        )
        
        # 多任务预测器
        self.predictor = MultiTaskPredictor(
            input_dim=embed_dim,
            num_intermediate_events=num_intermediate_events
        )
        
        # 不确定性估计
        self.uncertainty_estimator = UncertaintyEstimator(input_dim=embed_dim)
        
        # 动态任务权重学习
        self.task_weight_net = nn.Linear(embed_dim, 2 + num_intermediate_events)  # 主任务+时间预测+中间事件
        
    def forward(self, 
                event_sequences: torch.Tensor,
                advertiser_types: torch.Tensor,
                additional_features: torch.Tensor,
                event_masks: Optional[torch.Tensor] = None):
        """
        Args:
            event_sequences: [batch_size, seq_len] 事件序列
            advertiser_types: [batch_size] 广告主类型
            additional_features: [batch_size, feature_dim] 额外特征
            event_masks: [batch_size, seq_len] 事件掩码
        """
        # 编码用户行为路径
        user_behavior_path = self.event_encoder(event_sequences, event_masks)
        
        # 编码广告主特征路径
        advertiser_feature_path = self.advertiser_encoder(advertiser_types)
        
        # 路径融合：将三种路径信息融合
        fused_paths = torch.cat([
            user_behavior_path, 
            advertiser_feature_path, 
            additional_features
        ], dim=-1)
        
        # 路径融合计算
        fused_features = self.path_fusion(fused_paths)
        
        # 多任务预测
        predictions = self.predictor(fused_features)
        
        # 不确定性估计
        uncertainty = self.uncertainty_estimator(fused_features)
        
        # 任务权重
        task_weights = torch.softmax(self.task_weight_net(fused_features), dim=-1)
        
        return {
            'predictions': predictions,
            'uncertainty': uncertainty,
            'task_weights': task_weights
        }

class PFNLossFunction:
    """
    PathFusion Network的损失函数
    """
    def __init__(self, 
                 lambda_main: float = 1.0,
                 lambda_aux: float = 0.5,
                 lambda_reg: float = 0.01,
                 lambda_uncertainty: float = 0.1):
        self.lambda_main = lambda_main
        self.lambda_aux = lambda_aux
        self.lambda_reg = lambda_reg
        self.lambda_uncertainty = lambda_uncertainty
        
    def __call__(self, 
                 model_outputs: Dict,
                 targets: Dict,
                 model: PathFusionNetwork):
        """
        计算融合路径模型的综合损失
        """
        predictions = model_outputs['predictions']
        uncertainty = model_outputs['uncertainty']
        
        total_loss = 0.0
        
        # 主任务损失 (impression -> purchase)
        main_target = targets['main_task']  # [batch_size, 1]
        main_pred = predictions['main_task']
        main_loss = F.binary_cross_entropy(main_pred, main_target.float())
        total_loss += self.lambda_main * main_loss
        
        # 辅助任务损失
        if 'auxiliary_targets' in targets and len(predictions['auxiliary_tasks']) > 0:
            aux_targets = targets['auxiliary_targets']  # List of [batch_size, 1]
            aux_preds = predictions['auxiliary_tasks']
            
            aux_loss = 0.0
            for i, (pred, target) in enumerate(zip(aux_preds, aux_targets)):
                if target is not None:  # 某些广告主可能没有此中间事件
                    aux_loss += F.binary_cross_entropy(pred, target.float())
            
            total_loss += self.lambda_aux * aux_loss / len(aux_preds)
        
        # 时间预测损失
        if 'time_target' in targets and targets['time_target'] is not None:
            time_target = targets['time_target']
            time_pred = predictions['time_prediction']
            time_loss = F.mse_loss(time_pred, time_target.float())
            total_loss += self.lambda_aux * time_loss
        
        # 正则化损失
        reg_loss = sum(p.pow(2).sum() for p in model.parameters()) * self.lambda_reg
        total_loss += reg_loss
        
        # 不确定性损失
        uncertainty_loss = torch.mean(uncertainty)
        total_loss += self.lambda_uncertainty * uncertainty_loss
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'aux_loss': aux_loss if 'aux_loss' in locals() else 0.0,
            'reg_loss': reg_loss,
            'uncertainty_loss': uncertainty_loss
        }

class PFNTrainingPipeline:
    """
    PathFusion Network训练管道
    """
    def __init__(self, 
                 model: PathFusionNetwork,
                 loss_fn: PFNLossFunction,
                 learning_rate: float = 0.001):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
    def train_step(self, batch_data):
        """
        训练步骤
        """
        self.model.train()
        
        # 解析批次数据
        event_sequences = batch_data['event_sequences']
        advertiser_types = batch_data['advertiser_types']
        additional_features = batch_data['additional_features']
        event_masks = batch_data.get('event_masks')
        targets = batch_data['targets']
        
        # 前向传播
        outputs = self.model(
            event_sequences=event_sequences,
            advertiser_types=advertiser_types,
            additional_features=additional_features,
            event_masks=event_masks
        )
        
        # 计算损失
        losses = self.loss_fn(outputs, targets, self.model)
        
        # 反向传播
        self.optimizer.zero_grad()
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return losses
    
    def evaluate(self, eval_loader):
        """
        评估模型
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in eval_loader:
                outputs = self.model(
                    event_sequences=batch_data['event_sequences'],
                    advertiser_types=batch_data['advertiser_types'],
                    additional_features=batch_data['additional_features'],
                    event_masks=batch_data.get('event_masks')
                )
                
                losses = self.loss_fn(outputs, batch_data['targets'], self.model)
                total_loss += losses['total_loss'].item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches
        self.scheduler.step(avg_loss)
        
        return avg_loss

def create_sample_data():
    """
    创建示例数据
    """
    batch_size = 32
    seq_len = 10
    
    # 模拟数据
    data = {
        'event_sequences': torch.randint(0, 10, (batch_size, seq_len)),  # 事件ID: 0=impression, 1=purchase, 2=atc, 3=content_view...
        'advertiser_types': torch.randint(0, 5, (batch_size,)),  # 广告主类型
        'additional_features': torch.randn(batch_size, 64),  # 额外特征
        'event_masks': torch.ones(batch_size, seq_len),  # 掩码
        'targets': {
            'main_task': torch.randint(0, 2, (batch_size, 1)).float(),  # 主任务标签
            'auxiliary_targets': [torch.randint(0, 2, (batch_size, 1)).float() for _ in range(3)],  # 辅助任务标签
            'time_target': torch.rand(batch_size, 1) * 100  # 时间目标
        }
    }
    print(data)
    return data

def main():
    """
    主函数示例 - PathFusion Network (PFN) 初始化
    """
    # 模型配置
    config = {
        'event_vocab_size': 20,  # 包含impression, purchase, atc, content_view等
        'advertiser_type_vocab_size': 10,  # 广告主类型数量
        'feature_dim': 64,  # 额外特征维度
        'embed_dim': 128,
        'num_intermediate_events': 5  # 中间事件数量
    }
    
    # 创建PathFusion Network模型
    model = PathFusionNetwork(**config)
    
    # 创建损失函数
    loss_fn = PFNLossFunction(
        lambda_main=1.0,
        lambda_aux=0.5,
        lambda_reg=0.01,
        lambda_uncertainty=0.1
    )
    
    # 创建训练管道
    pipeline = PFNTrainingPipeline(model, loss_fn, learning_rate=0.001)
    
    # 示例训练循环
    print("PathFusion Network (PFN) 训练开始...")
    for epoch in range(5):  # 减少训练轮次以便演示
        # 获取示例数据
        sample_batch = create_sample_data()
        
        # 训练一步
        losses = pipeline.train_step(sample_batch)
        
        print(f"Epoch {epoch+1}, Total Loss: {losses['total_loss'].item():.4f}, "
              f"Main Loss: {losses['main_loss'].item():.4f}")
    
    print("\nPathFusion Network (PFN) 初始化完成！")
    print("模型已准备好处理异构广告主的转化路径建模任务。")

if __name__ == "__main__":
    main()



