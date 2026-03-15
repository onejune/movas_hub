"""
模型工具函数
"""
import numpy as np
import tensorflow as tf


def build_model_with_dummy_data(model):
    """
    使用虚拟数据构建模型，以便可以加载权重
    """
    # 创建虚拟输入数据
    batch_size = 2
    dummy_input = {}
    
    # 8 个数值特征
    for i in range(8):
        dummy_input[str(i)] = np.random.random(batch_size).astype(np.float32)
    
    # 9 个类别特征
    for i in range(8, 17):
        dummy_input[str(i)] = np.array([str(j) for j in range(batch_size)])
    
    # 调用模型以构建它
    _ = model(dummy_input, training=False)
    
    return model


def load_model_weights(model, ckpt_path):
    """
    先构建模型，然后加载权重
    """
    build_model_with_dummy_data(model)
    model.load_weights(ckpt_path)
    return model
