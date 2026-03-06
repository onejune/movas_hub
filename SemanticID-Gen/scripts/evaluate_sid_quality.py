"""
SID 质量评估脚本
输入: embedding .npy + SID .json
输出: 质量评估报告（控制台 + JSON）
"""

import argparse
import json
import re
import numpy as np


def parse_sid_token(token_str):
    """解析 SID token，提取数字。例: '<a_42>' -> 42"""
    m = re.search(r'(\d+)', token_str)
    return int(m.group(1)) if m else 0


def sids_to_array(sids_dict):
    """将 SID 字典转为 numpy 数组。
    输入: {"0": ['<a_42>', '<b_80>', '<c_160>'], ...}
    输出: (N, num_layers) numpy array
    """
    item_ids = sorted(sids_dict.keys(), key=lambda x: int(x) if x.isdigit() else x)
    N = len(item_ids)
    first_sid = sids_dict[item_ids[0]]
    num_layers = len(first_sid) if isinstance(first_sid, list) else 1
    
    sids_array = np.zeros((N, num_layers), dtype=np.int32)
    for i, item_id in enumerate(item_ids):
        sid = sids_dict[item_id]
        if isinstance(sid, list):
            sids_array[i] = [parse_sid_token(t) for t in sid[:num_layers]]
        else:
            sids_array[i, 0] = parse_sid_token(str(sid))
    
    return sids_array


def compute_cosine_similarity_matrix(embeddings):
    """计算余弦相似度矩阵 (N × N)"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normed = embeddings / norms
    cos_sim = normed @ normed.T
    return cos_sim


def metric_semantic_grouping(cos_sim, sids, K=10):
    """核心指标: 语义分组准确率"""
    N = cos_sim.shape[0]
    num_layers = sids.shape[1]
    
    # 去掉自身
    sim_matrix = cos_sim.copy()
    np.fill_diagonal(sim_matrix, -np.inf)
    
    # 找 embedding 最像的 K 个邻居
    topk_indices = np.argsort(-sim_matrix, axis=1)[:, :K]
    
    results = {}
    
    for layer_idx in range(1, num_layers + 1):
        # 构造组标签
        group_labels = []
        for i in range(N):
            label = '_'.join(str(sids[i, l]) for l in range(layer_idx))
            group_labels.append(label)
        group_labels = np.array(group_labels)
        
        # 计算准确率
        accuracies = np.zeros(N)
        for i in range(N):
            my_label = group_labels[i]
            neighbor_labels = group_labels[topk_indices[i]]
            accuracies[i] = (neighbor_labels == my_label).sum() / K
        
        mean_acc = accuracies.mean()
        
        # 随机基线
        unique_labels, counts = np.unique(group_labels, return_counts=True)
        random_baseline = ((counts / N) ** 2).sum()
        num_groups = len(unique_labels)
        lift = mean_acc / random_baseline if random_baseline > 0 else 0
        
        results[f'level_{layer_idx}_accuracy'] = round(float(mean_acc), 4)
        results[f'level_{layer_idx}_num_groups'] = num_groups
        results[f'level_{layer_idx}_random_baseline'] = round(float(random_baseline), 4)
        results[f'level_{layer_idx}_lift'] = round(float(lift), 1)
    
    return results


def compute_collision_rate(sids):
    """计算碰撞率: 有多少比例的商品 SID 完全相同"""
    N = sids.shape[0]
    # 将每行转为字符串用于去重
    sid_strings = ['_'.join(map(str, row)) for row in sids]
    unique_sids = len(set(sid_strings))
    collision_rate = (N - unique_sids) / N
    return collision_rate


def evaluate_sid_quality(embeddings, sids_dict, K=10):
    """评估 SID 质量
    
    参数:
        embeddings: (N, D) numpy array
        sids_dict: dict, {"0": ['<a_42>', '<b_80>', '<c_160>'], ...}
        K: 近邻数量
    
    返回:
        dict: 包含各项质量指标
    """
    print("\n" + "=" * 70)
    print("开始评估 SID 质量...")
    print("=" * 70)
    
    # 转换 SID 格式
    sids = sids_to_array(sids_dict)
    N = embeddings.shape[0]
    num_layers = sids.shape[1]
    
    # 计算余弦相似度矩阵
    print(f"\n[1/3] 计算 {N} 个商品的余弦相似度矩阵...")
    cos_sim = compute_cosine_similarity_matrix(embeddings)
    
    # 语义分组准确率
    print(f"\n[2/3] 计算语义分组准确率 (K={K})...")
    grouping_results = metric_semantic_grouping(cos_sim, sids, K=K)
    
    # 碰撞率
    print(f"\n[3/3] 计算碰撞率...")
    collision_rate = compute_collision_rate(sids)
    
    # 汇总结果
    results = {
        'num_items': N,
        'num_layers': num_layers,
        'collision_rate': round(float(collision_rate), 4),
        **grouping_results
    }
    
    # 打印总结
    print("\n" + "=" * 70)
    print("SID 质量评估总结")
    print("=" * 70)
    print(f"  商品数:        {N}")
    print(f"  SID层数:       {num_layers}")
    print(f"  碰撞率:        {collision_rate:.2%}")
    print(f"")
    print(f"  ★ 语义分组准确率 (K={K}):")
    
    for layer_idx in range(1, num_layers + 1):
        acc = grouping_results[f'level_{layer_idx}_accuracy']
        baseline = grouping_results[f'level_{layer_idx}_random_baseline']
        lift = grouping_results[f'level_{layer_idx}_lift']
        num_groups = grouping_results[f'level_{layer_idx}_num_groups']
        print(f"     前{layer_idx}层: {acc:.1%}  (随机基线 {baseline:.1%}, 提升 {lift}x, {num_groups}个组)")
    
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='评估 SID 质量')
    parser.add_argument('--emb_npy', type=str, required=True, help='Embedding .npy 文件')
    parser.add_argument('--sid_json', type=str, required=True, help='SID index .json 文件')
    parser.add_argument('--output', type=str, default=None, help='输出评估结果 JSON（可选）')
    parser.add_argument('--K', type=int, default=10, help='近邻数量（默认10）')
    args = parser.parse_args()

    # 加载数据
    print(f"加载 Embedding: {args.emb_npy}")
    embeddings = np.load(args.emb_npy)
    print(f"  shape: {embeddings.shape}")

    print(f"\n加载 SID: {args.sid_json}")
    with open(args.sid_json, 'r', encoding='utf-8') as f:
        sids_dict = json.load(f)
    print(f"  共 {len(sids_dict)} 条 SID")

    # 评估
    results = evaluate_sid_quality(embeddings, sids_dict, K=args.K)

    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ 评估结果已保存: {args.output}")


if __name__ == '__main__':
    main()
