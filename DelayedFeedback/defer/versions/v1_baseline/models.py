"""
Defer 模型 - PyTorch 版本

模型类型:
- MLP_SIG: 基础二分类模型 (Baseline, Vanilla, FNW, FNC, Oracle)
- MLP_EXP_DELAY: DFM 指数延迟模型
- MLP_tn_dp: ES-DFM 真负/延迟正分类模型
- MLP_FSIW: FSIW 重要性权重模型
- MLP_3class: 三分类模型
- MLP_likeli: 似然模型
- MLP_wintime_sep: 窗口时间分离模型
- MLP_winadapt: 自适应窗口模型 (Defer 核心)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 特征配置
# 数值特征: 无 (已移除)
NUM_FEATURES = 0

# 类别特征词表大小 (根据实际数据 hash 范围设置，需 >= max_index + 1)
# business_type, country, adx, make, bundle, demand_pkgname, offerid, campaignid, model
# 实际 max: [43, 198, 62, 229, 511, 243, 510, 509, 255]
CATE_BIN_SIZE = (64, 256, 64, 256, 512, 256, 512, 512, 256)  # 9 个类别特征的词表大小
EMBEDDING_DIM = 8


class FeatureProcessor(nn.Module):
    """
    特征处理层
    - 只使用类别特征 Embedding
    """
    def __init__(self, num_features=NUM_FEATURES, cate_bin_sizes=CATE_BIN_SIZE, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.num_features = num_features
        self.cate_bin_sizes = cate_bin_sizes
        self.embedding_dim = embedding_dim
        
        # 类别特征的 Embedding 层
        self.cate_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim)
            for vocab_size in cate_bin_sizes
        ])
        
        # 输出维度 = 类别特征数 * embedding_dim (不再包含数值特征)
        self.output_dim = len(cate_bin_sizes) * embedding_dim
    
    def forward(self, num_feats, cate_feats):
        """
        Args:
            num_feats: [batch_size, ?] 数值特征 (忽略)
            cate_feats: [batch_size, 9] 类别特征 (整数索引)
        
        Returns:
            [batch_size, output_dim] 拼接后的特征
        """
        features = []
        
        for i, emb_layer in enumerate(self.cate_embeddings):
            # 确保索引在有效范围内
            cate_idx = cate_feats[:, i].long() % self.cate_bin_sizes[i]
            emb = emb_layer(cate_idx)  # [batch_size, embedding_dim]
            features.append(emb)
        
        return torch.cat(features, dim=1)


class MLP(nn.Module):
    """
    基础 MLP 模型
    
    结构:
        FeatureProcessor -> FC(256) -> BN -> LeakyReLU 
                        -> FC(256) -> BN -> LeakyReLU
                        -> FC(128) -> BN -> LeakyReLU
                        -> FC(output_dim)
    """
    def __init__(self, model_name, l2_reg=1e-6):
        super().__init__()
        self.model_name = model_name
        self.l2_reg = l2_reg
        
        # 特征处理
        self.feature_layer = FeatureProcessor()
        input_dim = self.feature_layer.output_dim
        
        # 共享层
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        
        # 根据模型类型构建不同的输出层
        if model_name == "MLP_wintime_sep":
            # 双塔结构: CV 塔 + Time 塔
            self.bn21 = nn.BatchNorm1d(256)
            self.fc31 = nn.Linear(256, 128)
            self.bn31 = nn.BatchNorm1d(128)
            self.fc41 = nn.Linear(128, 1)  # CV logits
            
            self.bn22 = nn.BatchNorm1d(256)
            self.fc32 = nn.Linear(256, 128)
            self.bn32 = nn.BatchNorm1d(128)
            self.fc42 = nn.Linear(128, 1)  # Time logits
        else:
            self.bn2 = nn.BatchNorm1d(256)
            self.fc3 = nn.Linear(256, 128)
            self.bn3 = nn.BatchNorm1d(128)
            
            # 输出层维度
            if model_name == "MLP_EXP_DELAY":
                self.fc4 = nn.Linear(128, 2)  # [cvr_logit, log_lambda]
            elif model_name == "MLP_tn_dp":
                self.fc4 = nn.Linear(128, 2)  # [tn_logit, dp_logit]
            elif model_name == "MLP_3class":
                self.fc4 = nn.Linear(128, 3)  # [class0, class1, class2]
            elif model_name == "MLP_FSIW":
                self.fc4 = nn.Linear(128, 1)
            elif model_name == "MLP_likeli":
                self.fc4 = nn.Linear(128, 2)  # [cv_logit, time_logit]
            elif model_name == "MLP_winadapt":
                self.fc4 = nn.Linear(128, 4)  # [cv, p15, p30, p60]
            else:  # MLP_SIG
                self.fc4 = nn.Linear(128, 1)
        
        print(f"构建模型: {model_name}")
    
    def forward(self, num_feats, cate_feats):
        """
        Args:
            num_feats: [batch_size, 8] 数值特征
            cate_feats: [batch_size, 9] 类别特征
        
        Returns:
            dict: 包含不同 logits 的字典
        """
        # 特征处理
        h = self.feature_layer(num_feats, cate_feats)
        
        # 共享层
        # 对齐 TF 原版: Dense(activation=leaky_relu) → BN
        # 即: Linear → leaky_relu → BN
        h = F.leaky_relu(self.fc1(h))
        h = self.bn1(h)
        h = F.leaky_relu(self.fc2(h))

        if self.model_name == "MLP_wintime_sep":
            # CV 塔
            h1 = self.bn21(h)
            h1 = F.leaky_relu(self.fc31(h1))
            h1 = self.bn31(h1)
            cv_logits = self.fc41(h1)

            # Time 塔
            h2 = self.bn22(h)
            h2 = F.leaky_relu(self.fc32(h2))
            h2 = self.bn32(h2)
            time_logits = self.fc42(h2)

            logits = torch.cat([cv_logits, time_logits], dim=1)
            return {
                "logits": logits,
                "cv_logits": cv_logits,
                "time_logits": time_logits
            }

        # 标准结构: Linear → leaky_relu → BN
        h = self.bn2(h)
        h = F.leaky_relu(self.fc3(h))
        h = self.bn3(h)
        logits = self.fc4(h)
        
        # 根据模型类型返回不同格式
        if self.model_name == "MLP_EXP_DELAY":
            return {
                "logits": logits[:, 0:1],
                "log_lamb": logits[:, 1:2],
                "cvr_logits": logits[:, 0:1],
                "delay_logits": logits[:, 1:2]
            }
        elif self.model_name == "MLP_tn_dp":
            # ES-DFM: tn_logits 预测真负概率，dp_logits 预测延迟正概率
            tn_logits = logits[:, 0:1]
            dp_logits = logits[:, 1:2]
            # CVR 预测: P(转化) = 1 - P(真负) + P(延迟正)
            # 简化: P(转化) ≈ 1 - P(真负) = sigmoid(-tn_logits)
            # 但更准确的是考虑延迟正: P(转化) = (1 - P(tn)) * (1 + P(dp))
            # 这里用重要性加权的思路: cv_logits = -tn_logits (基础) + dp_logits (调整)
            cv_logits = -tn_logits + 0.5 * dp_logits
            return {
                "logits": logits,
                "cv_logits": cv_logits,
                "tn_logits": tn_logits,
                "dp_logits": dp_logits
            }
        elif self.model_name == "MLP_likeli":
            return {
                "logits": logits,
                "cv_logits": logits[:, 0:1],
                "time_logits": logits[:, 1:2]
            }
        elif self.model_name == "MLP_winadapt":
            return {
                "logits": logits,
                "cv_logits": logits[:, 0:1],
                "time15_logits": logits[:, 1:2],
                "time30_logits": logits[:, 2:3],
                "time60_logits": logits[:, 3:4]
            }
        else:
            return {"logits": logits}
    
    def get_l2_loss(self):
        """计算 L2 正则化损失"""
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2_reg * l2_loss


def get_model(name, l2_reg=1e-6):
    """获取模型实例"""
    return MLP(name, l2_reg)


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    # 测试各种模型
    batch_size = 32
    num_feats = torch.randn(batch_size, 8)
    cate_feats = torch.randint(0, 100, (batch_size, 9))
    
    model_names = [
        "MLP_SIG", "MLP_EXP_DELAY", "MLP_tn_dp", 
        "MLP_3class", "MLP_FSIW", "MLP_likeli",
        "MLP_wintime_sep", "MLP_winadapt"
    ]
    
    for name in model_names:
        model = get_model(name)
        output = model(num_feats, cate_feats)
        print(f"{name}: {list(output.keys())}")
        for k, v in output.items():
            print(f"  {k}: {v.shape}")
        print()
