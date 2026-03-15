import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import regularizers
import numpy as np

# 特征配置 (与 PyTorch 版本对齐)
# 数值特征: 无 (已移除)
NUM_FEATURES = 0

# 类别特征词表大小 (与 PyTorch 版本对齐)
# business_type, country, adx, make, bundle, demand_pkgname, offerid, campaignid, model
CATE_BIN_SIZE = (64, 256, 64, 256, 512, 256, 512, 512, 256)  # 9 个类别特征的词表大小
EMBEDDING_DIM = 8


class FeatureProcessor(layers.Layer):
    """
    特征处理层 (与 PyTorch 版本对齐)
    - 只使用类别特征 Embedding
    """
    def __init__(self, cate_bin_sizes=CATE_BIN_SIZE, embedding_dim=EMBEDDING_DIM, **kwargs):
        super(FeatureProcessor, self).__init__(**kwargs)
        self.cate_bin_sizes = cate_bin_sizes
        self.embedding_dim = embedding_dim
        
        # 类别特征的 embedding 层
        self.cate_embeddings = []
        for i, vocab_size in enumerate(cate_bin_sizes):
            self.cate_embeddings.append(
                layers.Embedding(
                    input_dim=vocab_size, 
                    output_dim=embedding_dim,
                    name=f'cate_emb_{i}'
                )
            )
    
    def call(self, inputs, training=None):
        features = []
        
        # 只处理类别特征 (0-8)，与 PyTorch 版本对齐
        for i in range(9):
            key = str(i)
            if key in inputs:
                # 类别特征通过 hash 转换为索引
                cate_feat = inputs[key]
                if cate_feat.dtype == tf.string:
                    # 字符串转 hash
                    hash_val = tf.strings.to_hash_bucket_fast(cate_feat, self.cate_bin_sizes[i])
                else:
                    hash_val = tf.cast(cate_feat, tf.int32) % self.cate_bin_sizes[i]
                # 通过 embedding 层
                emb = self.cate_embeddings[i](hash_val)
                emb = tf.reshape(emb, (-1, self.embedding_dim))
                features.append(emb)
        
        # 拼接所有特征
        if features:
            return tf.concat(features, axis=1)
        else:
            raise ValueError("No features found in inputs")


class MLP(Model):

    def __init__(self, name, params):
        super(MLP, self).__init__()
        self.model_name = name
        self.params = params
        
        # 使用自定义特征处理层 (与 PyTorch 版本对齐)
        self.feature_layer = FeatureProcessor(
            cate_bin_sizes=CATE_BIN_SIZE,
            embedding_dim=EMBEDDING_DIM
        )

        self.fc1 = layers.Dense(256, activation=tf.nn.leaky_relu,
                                kernel_regularizer=regularizers.l2(params["l2_reg"]))
        self.bn1 = layers.BatchNormalization()
        self.fc2 = layers.Dense(256, activation=tf.nn.leaky_relu,
                                kernel_regularizer=regularizers.l2(params["l2_reg"]))

        if self.model_name == "MLP_wintime_sep":
            self.bn21 = layers.BatchNormalization()
            self.fc31 = layers.Dense(128, activation=tf.nn.leaky_relu,
                                kernel_regularizer=regularizers.l2(params["l2_reg"]))
            self.bn31 = layers.BatchNormalization()
            self.bn22 = layers.BatchNormalization()
            self.fc32 = layers.Dense(128, activation=tf.nn.leaky_relu,
                                kernel_regularizer=regularizers.l2(params["l2_reg"]))
            self.bn32 = layers.BatchNormalization()
        else:
            self.bn2 = layers.BatchNormalization()
            self.fc3 = layers.Dense(128, activation=tf.nn.leaky_relu,
                                kernel_regularizer=regularizers.l2(params["l2_reg"]))
            self.bn3 = layers.BatchNormalization()

        print("build model {}".format(name))
        if self.model_name == "MLP_EXP_DELAY":
            self.fc4 = layers.Dense(2)
        elif self.model_name == "MLP_tn_dp":
            self.fc4 = layers.Dense(2)
        elif self.model_name == "MLP_3class":
            self.fc4 = layers.Dense(3)
        elif self.model_name == "MLP_FSIW":
            self.fc4 = layers.Dense(1)
        elif self.model_name == "MLP_likeli":
            self.fc4 = layers.Dense(2)
        elif self.model_name == "MLP_wintime_sep":
            self.fc41 = layers.Dense(1)
            self.fc42 = layers.Dense(1)
        elif self.model_name == "MLP_winadapt":
            self.fc4 = layers.Dense(4)
        else:
            self.fc4 = layers.Dense(1)

    def call(self, x, training=False):
        h = self.feature_layer(x, training=training)
        h = self.fc1(h)
        h = self.bn1(h, training=training)
        h = self.fc2(h)
        
        if self.model_name == "MLP_wintime_sep":
            h1 = self.bn21(h, training=training)
            h1 = self.fc31(h1)
            h1 = self.bn31(h1, training=training)
            h1 = self.fc41(h1)
            
            h2 = self.bn22(h, training=training)
            h2 = self.fc32(h2)
            h2 = self.bn32(h2, training=training)
            h2 = self.fc42(h2)
            
            logits = tf.concat([h1, h2], axis=1)
            return {"logits": logits, "cv_logits": h1, "time_logits": h2}
        elif self.model_name == "MLP_likeli":
            h = self.bn2(h, training=training)
            h = self.fc3(h)
            h = self.bn3(h, training=training)
            logits = self.fc4(h)
            return {"logits": logits, "cv_logits": logits[:, 0:1], "time_logits": logits[:, 1:2]}
        elif self.model_name == "MLP_winadapt":
            h = self.bn2(h, training=training)
            h = self.fc3(h)
            h = self.bn3(h, training=training)
            logits = self.fc4(h)
            return {
                "logits": logits,
                "cv_logits": logits[:, 0:1],
                "time15_logits": logits[:, 1:2],
                "time30_logits": logits[:, 2:3],
                "time60_logits": logits[:, 3:4]
            }
        elif self.model_name == "MLP_tn_dp":
            h = self.bn2(h, training=training)
            h = self.fc3(h)
            h = self.bn3(h, training=training)
            logits = self.fc4(h)
            return {
                "logits": logits,
                "tn_logits": logits[:, 0:1],
                "dp_logits": logits[:, 1:2]
            }
        elif self.model_name == "MLP_EXP_DELAY":
            h = self.bn2(h, training=training)
            h = self.fc3(h)
            h = self.bn3(h, training=training)
            logits = self.fc4(h)
            # DFM 模型需要 log_lamb 用于指数延迟分布
            return {
                "logits": logits[:, 0:1],  # CVR logits
                "log_lamb": logits[:, 1:2],  # log(lambda) for exponential delay
                "cvr_logits": logits[:, 0:1],
                "delay_logits": logits[:, 1:2]
            }
        elif self.model_name == "MLP_3class":
            h = self.bn2(h, training=training)
            h = self.fc3(h)
            h = self.bn3(h, training=training)
            logits = self.fc4(h)
            return {"logits": logits}
        else:
            h = self.bn2(h, training=training)
            h = self.fc3(h)
            h = self.bn3(h, training=training)
            logits = self.fc4(h)
            return {"logits": logits}


def get_model(name, params):
    return MLP(name, params)
