"""
汇总表格生成: 把 文本、Embedding、SID 合并成一张表
输入:
  - item.json (商品文本)
  - embedding .npy
  - SID index .json
输出:
  - summary.csv  (Excel/WPS可直接打开)
  - summary.json (程序读取)
"""

import argparse
import json
import os
import csv
import html
import re

import numpy as np


def clean_text(raw_text):
    """清洗文本"""
    if isinstance(raw_text, list):
        parts = []
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r'</?\w+[^>]*>', '', raw)
            raw = re.sub(r'["\n\r]*', '', raw)
            parts.append(raw.strip())
        return ' '.join(parts)
    elif isinstance(raw_text, dict):
        return str(raw_text)[1:-1].strip()
    else:
        text = html.unescape(str(raw_text).strip())
        text = re.sub(r'</?\w+[^>]*>', '', text)
        text = re.sub(r'["\n\r]*', '', text)
        return text


def main():
    parser = argparse.ArgumentParser(description='生成汇总表格: 文本 + Embedding + SID')
    parser.add_argument('--item_json', type=str, required=True, help='商品数据 item.json')
    parser.add_argument('--emb_npy', type=str, default=None, help='Embedding .npy 文件（可选）')
    parser.add_argument('--sid_json', type=str, default=None, help='SID index .json 文件（可选）')
    parser.add_argument('--output', type=str, required=True, help='输出路径（不含扩展名，会生成 .csv 和 .json）')
    parser.add_argument('--emb_dims', type=int, default=8, help='CSV中显示的Embedding维度数（默认前8维）')
    parser.add_argument('--limit', type=int, default=None, help='只输出前N条')
    args = parser.parse_args()

    # ---- 1. 加载文本 ----
    print(f"加载商品数据: {args.item_json}")
    with open(args.item_json, 'r', encoding='utf-8') as f:
        item_data = json.load(f)

    item_ids = sorted(item_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
    total = len(item_ids)
    print(f"  共 {total} 个商品")

    # ---- 2. 加载 Embedding（可选）----
    embeddings = None
    if args.emb_npy and os.path.exists(args.emb_npy):
        print(f"加载 Embedding: {args.emb_npy}")
        embeddings = np.load(args.emb_npy)
        print(f"  shape: {embeddings.shape}")
    else:
        print("未提供 Embedding 文件，跳过")

    # ---- 3. 加载 SID（可选）----
    sids = None
    if args.sid_json and os.path.exists(args.sid_json):
        print(f"加载 SID: {args.sid_json}")
        with open(args.sid_json, 'r', encoding='utf-8') as f:
            sids = json.load(f)
        print(f"  共 {len(sids)} 条 SID")
    else:
        print("未提供 SID 文件，跳过")

    # ---- 4. 组装表格 ----
    if args.limit:
        item_ids = item_ids[:args.limit]

    rows = []
    for idx, item_id in enumerate(item_ids):
        data = item_data[item_id]

        # 文本
        title = clean_text(data.get('title', ''))
        description = clean_text(data.get('description', ''))
        text = f"{title} {description}".strip()
        if len(text) > 200:
            text_short = text[:200] + '...'
        else:
            text_short = text

        row = {
            'item_id': item_id,
            'title': title,
            'text': text_short,
        }

        # Embedding
        if embeddings is not None and idx < len(embeddings):
            emb = embeddings[idx]
            row['emb_dim'] = len(emb)
            row['emb_norm'] = round(float(np.linalg.norm(emb)), 4)
            row['emb_mean'] = round(float(emb.mean()), 6)
            # 前几维
            for d in range(min(args.emb_dims, len(emb))):
                row[f'emb_{d}'] = round(float(emb[d]), 6)
            # 完整 embedding（JSON输出用）
            row['embedding_full'] = emb.tolist()

        # SID
        if sids is not None and item_id in sids:
            sid = sids[item_id]
            row['sid'] = ' '.join(sid) if isinstance(sid, list) else str(sid)
            if isinstance(sid, list):
                for i, token in enumerate(sid):
                    row[f'sid_layer_{i+1}'] = token

        rows.append(row)

    # ---- 5. 输出 CSV ----
    output_csv = args.output + '.csv'
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)

    # 确定CSV列（不包含 embedding_full）
    csv_columns = ['item_id', 'title', 'text']
    if embeddings is not None:
        csv_columns += ['emb_dim', 'emb_norm', 'emb_mean']
        csv_columns += [f'emb_{d}' for d in range(min(args.emb_dims, embeddings.shape[1] if embeddings is not None else 0))]
    if sids is not None:
        csv_columns.append('sid')
        # 根据第一条SID确定层数
        first_sid = sids.get(item_ids[0], [])
        if isinstance(first_sid, list):
            for i in range(len(first_sid)):
                csv_columns.append(f'sid_layer_{i+1}')

    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ CSV 已保存: {output_csv}")
    print(f"   共 {len(rows)} 行, {len(csv_columns)} 列")

    # ---- 6. 输出 JSON（包含完整embedding）----
    output_json = args.output + '.json'
    # JSON版本不截断embedding
    json_rows = []
    for row in rows:
        json_row = {k: v for k, v in row.items()}
        json_rows.append(json_row)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_rows, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON 已保存: {output_json}")

    # ---- 7. 打印预览 ----
    print(f"\n{'='*80}")
    print("表格预览（前5条）:")
    print(f"{'='*80}")
    preview_cols = ['item_id', 'title', 'emb_dim', 'emb_norm', 'sid']
    preview_cols = [c for c in preview_cols if c in csv_columns]

    # 表头
    header = ' | '.join(f'{c:>12s}' if c != 'title' else f'{c:<30s}' for c in preview_cols)
    print(header)
    print('-' * len(header))

    for row in rows[:5]:
        vals = []
        for c in preview_cols:
            v = row.get(c, '')
            if c == 'title':
                v = str(v)[:30]
                vals.append(f'{v:<30s}')
            elif c == 'sid':
                vals.append(f'{str(v):>12s}')
            else:
                vals.append(f'{str(v):>12s}')
        print(' | '.join(vals))

    print(f"{'='*80}")


if __name__ == '__main__':
    main()
