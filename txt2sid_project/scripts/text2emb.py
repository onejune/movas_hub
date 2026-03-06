"""
文本 → Embedding 向量
支持 CPU 和 GPU，通过 --device 参数切换
"""

import argparse
import json
import os
import html
import re
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def clean_text(raw_text):
    """清洗文本：去除HTML标签等"""
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


def load_item_data(item_json_path, features=('title', 'description')):
    """加载商品数据"""
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


def load_model(model_path, device):
    """加载模型到指定设备"""
    print(f"Loading model from: {model_path}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        low_cpu_mem_usage=True
    )
    model = model.to(device)
    model.eval()

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully")
    return tokenizer, model


@torch.no_grad()
def generate_embeddings(item_text_list, tokenizer, model, device, max_len=512, batch_size=8):
    """生成 embedding"""
    all_ids, all_texts = zip(*item_text_list)
    total_items = len(all_texts)

    print(f"Total items: {total_items}")
    print(f"Batch size: {batch_size}")

    all_results = []

    for i in tqdm(range(0, total_items, batch_size), desc="Generating embeddings"):
        batch_texts = list(all_texts[i:i + batch_size])
        batch_ids = all_ids[i:i + batch_size]

        encoded = tokenizer(
            batch_texts,
            max_length=max_len,
            truncation=True,
            return_tensors='pt',
            padding=True
        ).to(device)

        outputs = model(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask)

        # Mean Pooling
        last_hidden = outputs.last_hidden_state
        mask_expanded = encoded.attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_output = sum_embeddings / sum_mask

        mean_output = mean_output.cpu().float().numpy()

        for idx, emb in zip(batch_ids, mean_output):
            all_results.append((idx, emb))

    # 按 item_id 排序
    all_results.sort(key=lambda x: x[0])
    final_embeddings = np.stack([x[1] for x in all_results], axis=0)

    return final_embeddings


def main():
    parser = argparse.ArgumentParser(description='文本 → Embedding')
    parser.add_argument('--item_json', type=str, required=True, help='输入 item.json 路径')
    parser.add_argument('--model_path', type=str, required=True, help='Qwen 模型路径')
    parser.add_argument('--output', type=str, required=True, help='输出 .npy 路径')
    parser.add_argument('--device', type=str, default='cpu', help='设备: cpu / cuda:0')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--max_len', type=int, default=512, help='最大文本长度')
    parser.add_argument('--plm_name', type=str, default='qwen', help='模型名（用于文件命名）')
    parser.add_argument('--num_samples', type=int, default=-1, help='只处理前N个（-1=全部）')

    args = parser.parse_args()
    device = torch.device(args.device)

    # 1. 加载数据
    item_text_list = load_item_data(args.item_json)
    print(f"Loaded {len(item_text_list)} items")

    if args.num_samples > 0:
        item_text_list = item_text_list[:args.num_samples]
        print(f"[TEST MODE] Only processing first {args.num_samples} samples")

    # 2. 加载模型
    tokenizer, model = load_model(args.model_path, device)

    # 3. 生成 embedding
    embeddings = generate_embeddings(
        item_text_list, tokenizer, model, device,
        max_len=args.max_len, batch_size=args.batch_size
    )
    print(f"Embedding shape: {embeddings.shape}")

    # 4. 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, embeddings)
    print(f"Saved to: {args.output}")


if __name__ == '__main__':
    main()
