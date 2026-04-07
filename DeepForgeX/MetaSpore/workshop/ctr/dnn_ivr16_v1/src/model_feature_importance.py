#!/usr/bin/env python3
"""
基于模型的特征重要性分析
方法：
1. Embedding L2 范数 - 分析 sparse embedding 的权重大小
2. 第一层权重分析 - 分析 DNN 第一层对每个特征维度的权重
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# MetaSpore 路径
METASPORE_PATH = "/mnt/workspace/walter.wan/git_project/github_onejune/movas_hub/DeepForgeX/MetaSpore"
sys.path.insert(0, METASPORE_PATH)

MODEL_DIR = "/mnt/workspace/walter.wan/model_experiment/dnn/dnn_ivr16_v1/output/model_2026-03-02"
SCHEMA_PATH = "/mnt/workspace/walter.wan/git_project/github_onejune/movas_hub/DeepForgeX/MetaSpore/workshop/ctr/dnn_ivr16_v1/conf/combine_schema"
OUTPUT_DIR = "/mnt/workspace/walter.wan/git_project/github_onejune/movas_hub/DeepForgeX/MetaSpore/workshop/ctr/dnn_ivr16_v1/output/model_importance"

EMBEDDING_DIM = 8  # 每个特征的 embedding 维度


def load_features(schema_path):
    """加载特征列表"""
    features = []
    with open(schema_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                features.append(line)
    return features


def load_sparse_embedding(model_dir):
    """加载 sparse embedding 权重"""
    meta_path = os.path.join(model_dir, "dnn_sparse__sparse_meta.json")
    data_path_0 = os.path.join(model_dir, "dnn_sparse__sparse_0.dat")
    data_path_1 = os.path.join(model_dir, "dnn_sparse__sparse_1.dat")
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    print(f"Sparse meta: {json.dumps(meta, indent=2)}")
    
    # 读取 embedding 数据
    # 格式通常是: key (int64) + value (float32 * embedding_dim)
    embeddings = {}
    
    for data_path in [data_path_0, data_path_1]:
        if not os.path.exists(data_path):
            continue
            
        file_size = os.path.getsize(data_path)
        print(f"\n读取 {data_path}, 大小: {file_size / 1024 / 1024:.2f} MB")
        
        # 尝试读取为 float32 数组
        data = np.fromfile(data_path, dtype=np.float32)
        print(f"  数据形状: {data.shape}, 前10个值: {data[:10]}")
        
        # 假设是连续的 embedding 向量
        n_embeddings = len(data) // EMBEDDING_DIM
        print(f"  估计 embedding 数量: {n_embeddings}")
        
        if n_embeddings > 0:
            reshaped = data[:n_embeddings * EMBEDDING_DIM].reshape(n_embeddings, EMBEDDING_DIM)
            embeddings[data_path] = reshaped
    
    return embeddings, meta


def load_dense_weights(model_dir):
    """加载 DNN 权重"""
    weights = {}
    
    # 第一层权重: dnn.dnn.0.weight
    weight_path = os.path.join(model_dir, "dnn.dnn.0.weight__dense_data.dat")
    meta_path = os.path.join(model_dir, "dnn.dnn.0.weight__dense_meta.json")
    
    if os.path.exists(weight_path) and os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print(f"\nDNN 第一层 meta: {json.dumps(meta, indent=2)}")
        
        data = np.fromfile(weight_path, dtype=np.float32)
        print(f"第一层权重大小: {data.shape}")
        
        # 权重形状应该是 [hidden_size, input_size]
        # hidden_size = 512, input_size = n_features * embedding_dim
        weights['layer0'] = data
        weights['layer0_meta'] = meta
    
    return weights


def analyze_embedding_importance(embeddings, features, embedding_dim=8):
    """
    分析 embedding 重要性
    方法: 计算每个特征对应 embedding 的 L2 范数均值
    """
    print("\n" + "=" * 50)
    print("Embedding L2 范数分析")
    print("=" * 50)
    
    results = []
    
    # 合并所有 embedding
    all_embeddings = []
    for path, emb in embeddings.items():
        all_embeddings.append(emb)
    
    if not all_embeddings:
        print("没有找到 embedding 数据")
        return pd.DataFrame()
    
    all_emb = np.concatenate(all_embeddings, axis=0)
    print(f"总 embedding 数量: {len(all_emb)}")
    
    # 计算所有 embedding 的 L2 范数
    l2_norms = np.linalg.norm(all_emb, axis=1)
    print(f"L2 范数统计: min={l2_norms.min():.4f}, max={l2_norms.max():.4f}, "
          f"mean={l2_norms.mean():.4f}, std={l2_norms.std():.4f}")
    
    # 由于我们不知道具体哪个 embedding 对应哪个特征
    # 只能给出整体统计
    return {
        'total_embeddings': len(all_emb),
        'l2_min': float(l2_norms.min()),
        'l2_max': float(l2_norms.max()),
        'l2_mean': float(l2_norms.mean()),
        'l2_std': float(l2_norms.std()),
        'l2_percentiles': {
            'p10': float(np.percentile(l2_norms, 10)),
            'p25': float(np.percentile(l2_norms, 25)),
            'p50': float(np.percentile(l2_norms, 50)),
            'p75': float(np.percentile(l2_norms, 75)),
            'p90': float(np.percentile(l2_norms, 90)),
        }
    }


def analyze_first_layer_importance(weights, features, embedding_dim=8):
    """
    分析第一层权重对每个特征的重要性
    方法: 第一层权重 [hidden, input] 中，每个特征占 embedding_dim 个输入维度
          计算每个特征对应维度的权重 L2 范数
    """
    print("\n" + "=" * 50)
    print("第一层权重分析")
    print("=" * 50)
    
    if 'layer0' not in weights:
        print("没有找到第一层权重")
        return pd.DataFrame()
    
    layer0 = weights['layer0']
    meta = weights.get('layer0_meta', {})
    
    # 获取形状
    # MetaSpore 存储格式可能是 [out_features, in_features] 或展平的
    n_features = len(features)
    expected_input_dim = n_features * embedding_dim
    
    print(f"特征数: {n_features}, 预期输入维度: {expected_input_dim}")
    print(f"权重数据大小: {len(layer0)}")
    
    # 尝试推断形状
    # 如果是 512 * input_dim 的展平数据
    hidden_size = 512
    if len(layer0) == hidden_size * expected_input_dim:
        w = layer0.reshape(hidden_size, expected_input_dim)
        print(f"权重形状: {w.shape}")
    else:
        # 尝试其他形状
        for hs in [512, 256, 128, 64]:
            if len(layer0) % hs == 0:
                in_dim = len(layer0) // hs
                print(f"尝试形状: [{hs}, {in_dim}]")
                if in_dim % embedding_dim == 0:
                    n_feat = in_dim // embedding_dim
                    print(f"  对应特征数: {n_feat}")
                    if n_feat == n_features or abs(n_feat - n_features) < 10:
                        w = layer0.reshape(hs, in_dim)
                        hidden_size = hs
                        break
        else:
            print("无法推断权重形状")
            return pd.DataFrame()
    
    # 计算每个特征的重要性
    results = []
    actual_n_features = w.shape[1] // embedding_dim
    
    for i, feat in enumerate(features[:actual_n_features]):
        start = i * embedding_dim
        end = start + embedding_dim
        
        # 取出该特征对应的权重 [hidden_size, embedding_dim]
        feat_weights = w[:, start:end]
        
        # 计算重要性指标
        l2_norm = np.linalg.norm(feat_weights)
        mean_abs = np.abs(feat_weights).mean()
        max_abs = np.abs(feat_weights).max()
        
        results.append({
            'feature': feat,
            'position': i,
            'l2_norm': round(float(l2_norm), 4),
            'mean_abs_weight': round(float(mean_abs), 4),
            'max_abs_weight': round(float(max_abs), 4),
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('l2_norm', ascending=False)
    
    # 添加排名和分类
    df['rank'] = range(1, len(df) + 1)
    
    # 基于 L2 范数分位数分类
    p75 = df['l2_norm'].quantile(0.75)
    p50 = df['l2_norm'].quantile(0.50)
    p25 = df['l2_norm'].quantile(0.25)
    
    def classify(l2):
        if l2 >= p75:
            return 'HIGH'
        elif l2 >= p50:
            return 'MEDIUM_HIGH'
        elif l2 >= p25:
            return 'MEDIUM_LOW'
        else:
            return 'LOW'
    
    df['importance_level'] = df['l2_norm'].apply(classify)
    
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("基于模型的特征重要性分析")
    print("=" * 60)
    
    # 加载特征
    features = load_features(SCHEMA_PATH)
    print(f"特征数: {len(features)}")
    
    # 加载模型权重
    print("\n--- 加载 Sparse Embedding ---")
    embeddings, emb_meta = load_sparse_embedding(MODEL_DIR)
    
    print("\n--- 加载 Dense 权重 ---")
    weights = load_dense_weights(MODEL_DIR)
    
    # 分析
    print("\n" + "=" * 60)
    print("开始分析")
    print("=" * 60)
    
    # Embedding 分析
    emb_stats = analyze_embedding_importance(embeddings, features, EMBEDDING_DIM)
    if emb_stats:
        with open(os.path.join(OUTPUT_DIR, "embedding_stats.json"), 'w') as f:
            json.dump(emb_stats, f, indent=2)
        print(f"\nEmbedding 统计已保存")
    
    # 第一层权重分析
    layer_importance = analyze_first_layer_importance(weights, features, EMBEDDING_DIM)
    if len(layer_importance) > 0:
        layer_importance.to_csv(os.path.join(OUTPUT_DIR, "layer0_importance.csv"), index=False)
        print(f"\n第一层重要性分析已保存")
        
        # 打印 Top 20
        print("\n--- Top 20 重要特征 (基于第一层权重) ---")
        print(layer_importance.head(20).to_string(index=False))
        
        # 打印 Bottom 20
        print("\n--- Bottom 20 特征 ---")
        print(layer_importance.tail(20).to_string(index=False))
        
        # 统计
        print("\n--- 重要性分布 ---")
        print(layer_importance['importance_level'].value_counts())
    
    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
