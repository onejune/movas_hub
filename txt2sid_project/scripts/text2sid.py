"""
端到端 SID 生成脚本: 输入商品文本 → 直接输出 SID
支持单条文本、批量JSON文件、交互模式
"""

import argparse
import html
import json
import os
import re
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import re

from models.rqvae import RQVAE


# ========================= 文本预处理 =========================

def clean_text(raw_text):
    if isinstance(raw_text, list):
        new_raw_text = []
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r'</?\w+[^>]*>', '', raw)
            raw = re.sub(r'["\n\r]*', '', raw)
            new_raw_text.append(raw.strip())
        cleaned_text = ' '.join(new_raw_text)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = raw_text.strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r'</?\w+[^>]*>', '', cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)

    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'

    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text


def prepare_text_from_item_json(item_json_path, features=('title', 'description')):
    with open(item_json_path, 'r', encoding='utf-8') as f:
        item2feature = json.load(f)

    item_text_list = []
    for item_id_str, data in item2feature.items():
        parts = []
        for key in features:
            if key in data:
                cleaned = clean_text(data[key]).strip()
                if cleaned:
                    parts.append(cleaned)
        text = ' '.join(parts) if parts else 'unknown item.'

        try:
            item_id = int(item_id_str)
        except ValueError:
            item_id = item_id_str
        item_text_list.append((item_id, text))

    return item_text_list


# ========================= 加载模型 =========================

def load_qwen(model_path, device):
    print(f"[1/2] 加载 Qwen 模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  → Qwen 已加载到 {device}")
    return tokenizer, model


def load_rqvae(ckpt_path, device):
    print(f"[2/2] 加载 RQ-VAE: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt['args']
    state_dict = ckpt['state_dict']

    first_key = [k for k in state_dict if 'encoder' in k and 'weight' in k][0]
    in_dim = state_dict[first_key].shape[1]

    model = RQVAE(
        in_dim=in_dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    num_layers = len(args.num_emb_list)
    codebook_sizes = args.num_emb_list
    print(f"  → RQ-VAE 已加载: {num_layers}层, codebook大小={codebook_sizes}, embedding输入维度={in_dim}")
    return model


# ========================= 核心函数 =========================

@torch.no_grad()
def texts_to_embeddings(texts, tokenizer, qwen_model, device, max_len=512, batch_size=8):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        encoded = tokenizer(
            batch, max_length=max_len, truncation=True,
            return_tensors='pt', padding=True,
        ).to(device)

        outputs = qwen_model(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
        )

        last_hidden = outputs.last_hidden_state
        mask = encoded.attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_emb = torch.sum(last_hidden * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_emb = sum_emb / sum_mask

        all_embeddings.append(mean_emb.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


@torch.no_grad()
def embeddings_to_sids(embeddings, rqvae_model, device, batch_size=64):
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
    all_sids = []

    emb_tensor = torch.from_numpy(embeddings).float()

    for i in range(0, len(emb_tensor), batch_size):
        batch = emb_tensor[i:i + batch_size].to(device)
        indices = rqvae_model.get_indices(batch, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()

        for index in indices:
            sid = [prefix[j].format(int(idx)) for j, idx in enumerate(index)]
            all_sids.append(sid)

    return all_sids


def text_to_sid(texts, tokenizer, qwen_model, rqvae_model, device,
                max_len=512, qwen_batch_size=8, rqvae_batch_size=64):
    embeddings = texts_to_embeddings(
        texts, tokenizer, qwen_model, device,
        max_len=max_len, batch_size=qwen_batch_size,
    )
    sids = embeddings_to_sids(
        embeddings, rqvae_model, device,
        batch_size=rqvae_batch_size,
    )
    return sids


# ========================= SID 质量评估 =========================

def parse_sid_token(token_str):
    m = re.search(r'(\d+)', token_str)
    return int(m.group(1)) if m else 0

def sids_to_array(sids_list):
    N = len(sids_list)
    sids_array = np.zeros((N, 3), dtype=np.int32)
    for i, sid in enumerate(sids_list):
        sids_array[i] = [parse_sid_token(t) for t in sid[:3]]
    return sids_array

def compute_cosine_similarity_matrix(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normed = embeddings / norms
    cos_sim = normed @ normed.T
    return cos_sim

def evaluate_sid_quality(embeddings, sids_list, K=10):
    print("\n" + "=" * 70)
    print("开始评估 SID 质量...")
    print("=" * 70)
    sids = sids_to_array(sids_list)
    N = embeddings.shape[0]
    print(f"\n[1/3] 计算 {N} 个商品的余弦相似度矩阵...")
    cos_sim = compute_cosine_similarity_matrix(embeddings)
    print(f"\n[2/3] 计算语义分组准确率 (K={K})...")
    grouping_results = metric_semantic_grouping(cos_sim, sids, K=K)
    print(f"\n[3/3] 计算碰撞率...")
    collision_rate = compute_collision_rate(sids)
    results = {'num_items': N, 'collision_rate': collision_rate, **grouping_results}
    print("\n" + "=" * 70)
    print("SID 质量评估总结")
    print("=" * 70)
    print(f"  商品数:        {N}")
    print(f"  碰撞率:        {collision_rate:.2%}")
    print(f"  ★ 语义分组准确率 (K={K}):")
    print(f"     前1层: {grouping_results['level_1_accuracy']:.1%}  (随机基线 {grouping_results['level_1_random_baseline']:.1%}, 提升 {grouping_results['level_1_lift']}x)")
    print(f"     前2层: {grouping_results['level_2_accuracy']:.1%}  (随机基线 {grouping_results['level_2_random_baseline']:.1%}, 提升 {grouping_results['level_2_lift']}x)")
    print(f"     前3层: {grouping_results['level_3_accuracy']:.1%}  (随机基线 {grouping_results['level_3_random_baseline']:.1%}, 提升 {grouping_results['level_3_lift']}x)")
    print("=" * 70)
    return results

def metric_semantic_grouping(cos_sim, sids, K=10):
    N = cos_sim.shape[0]
    sim_matrix = cos_sim.copy()
    np.fill_diagonal(sim_matrix, -np.inf)
    topk_indices = np.argsort(-sim_matrix, axis=1)[:, :K]
    results = {}
    for level_name, num_layers in [('前1层', 1), ('前2层', 2), ('前3层', 3)]:
        group_labels = []
        for i in range(N):
            label = '_'.join(str(sids[i, l]) for l in range(num_layers))
            group_labels.append(label)
        group_labels = np.array(group_labels)
        accuracies = np.zeros(N)
        for i in range(N):
            my_label = group_labels[i]
            neighbor_labels = group_labels[topk_indices[i]]
            accuracies[i] = (neighbor_labels == my_label).sum() / K
        mean_acc = accuracies.mean()
        unique_labels, counts = np.unique(group_labels, return_counts=True)
        random_baseline = ((counts / N) ** 2).sum()
        num_groups = len(unique_labels)
        lift = mean_acc / random_baseline if random_baseline > 0 else 0
        results[f'level_{num_layers}_accuracy'] = round(float(mean_acc), 4)
        results[f'level_{num_layers}_num_groups'] = num_groups
        results[f'level_{num_layers}_random_baseline'] = round(float(random_baseline), 4)
        results[f'level_{num_layers}_lift'] = round(float(lift), 1)
    return results

def compute_collision_rate(sids):
    N = sids.shape[0]
    sid_strings = ['_'.join(map(str, row)) for row in sids]
    unique_sids = len(set(sid_strings))
    collision_rate = (N - unique_sids) / N
    return collision_rate


# ========================= 命令行参数 =========================

def parse_args():
    parser = argparse.ArgumentParser(description='端到端 SID 生成: 文本 → SID')
    parser.add_argument('--qwen_path', type=str, required=True)
    parser.add_argument('--rqvae_ckpt', type=str, required=True)
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--item_json', type=str, default=None)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--qwen_batch_size', type=int, default=4)
    parser.add_argument('--rqvae_batch_size', type=int, default=64)
    parser.add_argument('--limit', type=int, default=None)
    return parser.parse_args()


# ========================= 主程序 =========================

def main():
    args = parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    tokenizer, qwen_model = load_qwen(args.qwen_path, device)
    rqvae_model = load_rqvae(args.rqvae_ckpt, device)
    print("模型加载完成!\n")

    if args.text:
        text = clean_text(args.text)
        sids = text_to_sid(
            [text], tokenizer, qwen_model, rqvae_model, device,
            max_len=args.max_len, qwen_batch_size=1, rqvae_batch_size=1,
        )
        print(f"输入: {args.text}")
        print(f"SID:  {sids[0]}")

    elif args.item_json:
        print(f"读取: {args.item_json}")
        item_text_list = prepare_text_from_item_json(args.item_json)
        print(f"共 {len(item_text_list)} 个商品\n")

        if args.limit and args.limit < len(item_text_list):
            item_text_list = item_text_list[:args.limit]
            print(f"(--limit {args.limit}) 只处理前 {args.limit} 条\n")

        ids = [x[0] for x in item_text_list]
        texts = [x[1] for x in item_text_list]

        all_sids = []
        all_embeddings = []
        total = len(texts)
        batch = args.qwen_batch_size

        for start in tqdm(range(0, total, batch), desc="生成 SID"):
            end = min(start + batch, total)
            batch_texts = texts[start:end]

            batch_embeddings = texts_to_embeddings(
                batch_texts, tokenizer, qwen_model, device,
                max_len=args.max_len, batch_size=batch,
            )
            batch_sids = embeddings_to_sids(
                batch_embeddings, rqvae_model, device,
                batch_size=args.rqvae_batch_size,
            )

            all_sids.extend(batch_sids)
            if args.evaluate:
                all_embeddings.append(batch_embeddings)

        result = {str(item_id): sid for item_id, sid in zip(ids, all_sids)}

        output_file = args.output_file or args.item_json.replace('.item.json', '.text2sid.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n已保存 {len(result)} 条 SID → {output_file}")

        print("\n样例:")
        for i, (item_id, sid) in enumerate(list(result.items())[:5]):
            print(f"  [{item_id}] {sid}  ←  {texts[i][:60]}...")

        if args.evaluate:
            embeddings_matrix = np.concatenate(all_embeddings, axis=0)
            eval_results = evaluate_sid_quality(embeddings_matrix, all_sids, K=10)

            eval_output = output_file.replace('.json', '.eval.json')
            with open(eval_output, 'w') as f:
                json.dump(eval_results, f, indent=2)
            print(f"\n评估结果已保存 → {eval_output}")

    elif args.interactive:
        print("=" * 50)
        print("交互模式: 输入商品文本, 回车得到 SID (输入 q 退出)")
        print("=" * 50)
        while True:
            try:
                text = input("\n商品文本> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ('q', 'quit', 'exit'):
                break
            if not text:
                continue

            text = clean_text(text)
            sids = text_to_sid(
                [text], tokenizer, qwen_model, rqvae_model, device,
                max_len=args.max_len, qwen_batch_size=1, rqvae_batch_size=1,
            )
            print(f"  → SID: {sids[0]}")
        print("\n再见!")

    else:
        print("请指定输入方式: --text / --item_json / --interactive")


if __name__ == '__main__':
    main()
