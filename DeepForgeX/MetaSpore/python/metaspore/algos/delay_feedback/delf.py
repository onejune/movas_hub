import torch
import metaspore as ms
from ..layers import MLPLayer
import random

'''
《Overcoming Delayed Feedback in Conversion Prediction for Online Advertising》（CIKM 2021）

'''

class DELFWideDeep(torch.nn.Module):
    """
    基于WideDeep架构的延迟反馈建模模型
    包含两个分支：转化倾向分支和转化时间分支
    """
    def __init__(self,
                 use_wide=True,
                 wide_embedding_dim=10,
                 deep_embedding_dim=10,
                 wide_column_name_path=None,
                 wide_combine_schema_path=None,
                 deep_column_name_path=None,
                 deep_combine_schema_path=None,
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
                 **kwargs):
        super().__init__()
        
        # 共享的WideDeep骨干网络
        self.use_wide = use_wide
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(wide_embedding_dim,
                                                  wide_column_name_path,
                                                  wide_combine_schema_path)
            self.lr_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        
        self.dnn_sparse = ms.EmbeddingSumConcat(deep_embedding_dim,
                                               deep_column_name_path,
                                               deep_combine_schema_path)
        self.dnn_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha=ftrl_alpha, beta=ftrl_beta)
        self.dnn_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        
        # 共享的深度网络
        shared_dnn_input_dim = self.dnn_sparse.feature_count * deep_embedding_dim
        self.shared_dnn = MLPLayer(input_dim=shared_dnn_input_dim,
                                  output_dim=None,  # 不直接输出到1，而是输出中间特征
                                  hidden_units=dnn_hidden_units[:-1],  # 去掉最后一层
                                  hidden_activations=dnn_hidden_activations,
                                  final_activation=None,
                                  dropout_rates=net_dropout,
                                  batch_norm=batch_norm,
                                  use_bias=use_bias,
                                  input_norm=True)
        
        # 最终输出层维度
        final_input_dim = dnn_hidden_units[-2] if len(dnn_hidden_units) > 1 else shared_dnn_input_dim
        
        # 转化倾向分支 (Propensity Tower)
        self.propensity_dnn = MLPLayer(input_dim=final_input_dim,
                                      output_dim=1,
                                      hidden_units=[dnn_hidden_units[-1]],  # 最后一层
                                      hidden_activations=dnn_hidden_activations,
                                      final_activation=None,
                                      dropout_rates=net_dropout,
                                      batch_norm=batch_norm,
                                      use_bias=use_bias)
        
        # 转化时间分支 (Time Tower) - 输出Weibull分布参数
        self.time_k_dnn = MLPLayer(input_dim=final_input_dim,
                                  output_dim=1,
                                  hidden_units=[dnn_hidden_units[-1]],  # 最后一层
                                  hidden_activations=dnn_hidden_activations,
                                  final_activation="Softplus",  # 确保k>0
                                  dropout_rates=net_dropout,
                                  batch_norm=batch_norm,
                                  use_bias=use_bias)
        
        self.time_lambda_dnn = MLPLayer(input_dim=final_input_dim,
                                       output_dim=1,
                                       hidden_units=[dnn_hidden_units[-1]],  # 最后一层
                                       hidden_activations=dnn_hidden_activations,
                                       final_activation="Softplus",  # 确保lambda>0
                                       dropout_rates=net_dropout,
                                       batch_norm=batch_norm,
                                       use_bias=use_bias)
        
        # Wide部分（如果使用）
        if self.use_wide:
            self.wide_linear = torch.nn.Linear(self.lr_sparse.feature_count * wide_embedding_dim, 1, bias=False)
        
        if final_net_activation == 'sigmoid':
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = None

    def weibull_cdf(self, k, lam, t):
        """Weibull分布的累积分布函数"""
        return 1 - torch.exp(-(t / lam) ** k)

    def calculate_ipw_weights(self, p_prop, k, lam, obs_window=3.0):
        """
        计算 IPW 权重，支持标量或 Tensor 的 obs_window
        
        Args:
            p_prop: [B] 或 [B,1]
            k, lam: [B] 或 [B,1]
            obs_window: float or Tensor of shape [B]
        """
        # 确保输入为 [B]
        if p_prop.dim() > 1:
            p_prop = p_prop.squeeze(-1)
        if k.dim() > 1:
            k = k.squeeze(-1)
        if lam.dim() > 1:
            lam = lam.squeeze(-1)
        if not isinstance(obs_window, torch.Tensor):
            obs_window = torch.full_like(p_prop, obs_window)

        # P(T > obs_window | Y=1, x)
        prob_time_gt_w = torch.exp(- (obs_window / lam) ** k + 1e-8)  # 加小量防 inf

        # 分子: P(Y=1, T > w | x)
        numerator = p_prop * prob_time_gt_w

        # 分母: P(observed=0 | x) = P(Y=0|x) + P(Y=1, T > w | x)
        denominator = (1 - p_prop) + numerator

        # IPW 权重
        weights = numerator / (denominator + 1e-8)
        return weights

    def forward(self, x):
        # 获取共享嵌入特征
        if self.use_wide:
            wide_out = self.lr_sparse(x)
            wide_out = torch.sum(wide_out, dim=1, keepdim=True)
        
        dnn_out = self.dnn_sparse(x)
        dnn_out = self.shared_dnn(dnn_out)
        
        # 转化倾向分支
        propensity_out = self.propensity_dnn(dnn_out)
        
        # 转化时间分支（Weibull参数）
        k_out = self.time_k_dnn(dnn_out)
        lambda_out = self.time_lambda_dnn(dnn_out)
        
        # 如果使用wide部分，需要将wide特征也传给各分支
        if self.use_wide:
            wide_features = torch.sum(self.lr_sparse(x), dim=1)
            # 将wide特征与deep特征拼接后分别送入各分支
            combined_features = torch.cat([wide_features, dnn_out], dim=1)
            
            # 重新计算各分支（这里简化为只用deep特征，实际可根据需要调整）
            propensity_out = self.propensity_dnn(dnn_out)
            k_out = self.time_k_dnn(dnn_out)
            lambda_out = self.time_lambda_dnn(dnn_out)
            
            # wide部分对最终结果的贡献
            wide_contribution = self.wide_linear(wide_features.unsqueeze(1)).squeeze(1)
        else:
            wide_contribution = 0.0
        
        # 返回三个输出：转化倾向概率、形状参数k、尺度参数lambda
        propensity_prob = self.final_activation(propensity_out) if self.final_activation else propensity_out
        k_param = torch.clamp(k_out, min=0.01)  # 确保k>0
        lambda_param = torch.clamp(lambda_out, min=0.01)  # 确保lambda>0
        
        return propensity_prob, k_param, lambda_param, wide_contribution

    def predict(self, yhat, minibatch = None):
        """ 预测在 decision_window 内的转化概率 """
        p_prop, k, lam, _ = yhat
        decision_window = minibatch['attributeWindow']
        # 转为 numpy -> tensor，并放到与模型输出相同的设备上
        decision_window = torch.tensor(
            decision_window.values,
            dtype=torch.float32,
            device=p_prop.device
        )
        cdf_decision = self.weibull_cdf(k, lam, decision_window)
        
        if random.random() < 0.0001:
            print('-------> p_prop:', p_prop, 'cdf_decision:', cdf_decision, 'ivr:', p_prop * cdf_decision)
            print('-------> decision_window:', decision_window)
            print('-------> ipw_weights:', self.calculate_ipw_weights(p_prop, k, lam, obs_window=3.0))
        #注意，如果需要采样校准，只需对 p_prop校准即可
        return p_prop * cdf_decision

