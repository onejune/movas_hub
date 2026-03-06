"""
从训练好的 RQ-VAE 生成 SID（含 Sinkhorn 消碰）
输入: embedding .npy + RQ-VAE checkpoint
输出: SID index .json
"""
import collections
import json
import logging

import argparse
import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE

import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Generate SID index.json from a trained RQ-VAE checkpoint.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to best_collision_model.pth")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save *.index.json")
    parser.add_argument("--dataset_name", type=str, default=None, help="Output filename prefix")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu / cuda:0")
    parser.add_argument("--data_path", type=str, default=None, help="Override embedding .npy path")
    parser.add_argument("--model_name", type=str, default="", help="Model name suffix for output file")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_collision_refine_steps", type=int, default=20)
    parser.add_argument("--last_layer_sk_epsilon", type=float, default=0.003)
    return parser.parse_args()


def main():
    cli = parse_cli_args()

    device = torch.device(cli.device)
    os.makedirs(cli.output_dir, exist_ok=True)

    ckpt = torch.load(cli.ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    data_path = cli.data_path or args.data_path
    data = EmbDataset(data_path)

    dataset_name = cli.dataset_name
    if not dataset_name:
        dataset_name = os.path.basename(str(data_path)).split(".")[0]

    if cli.model_name:
        output_file = os.path.join(cli.output_dir, f"{dataset_name}.index-{cli.model_name}.json")
    else:
        output_file = os.path.join(cli.output_dir, f"{dataset_name}.index.json")

    model = RQVAE(
        in_dim=data.dim,
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
    print(model)

    data_loader = DataLoader(
        data,
        num_workers=cli.num_workers,
        batch_size=cli.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for d in tqdm(data_loader, desc="Generate indices (use_sk=False)"):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    # refine collisions (Sinkhorn) only on last layer by default
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = float(cli.last_layer_sk_epsilon)

    tt = 0
    while True:
        if tt >= cli.max_collision_refine_steps or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print(f"Refine step {tt+1}: {len(collision_item_groups)} collision groups")
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))
                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    print("All indices number: ", len(all_indices))
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate", (tot_item - tot_indice) / tot_item)

    all_indices_dict = {item: list(indices) for item, indices in enumerate(all_indices.tolist())}

    with open(output_file, "w") as fp:
        json.dump(all_indices_dict, fp)
    print(f"Saved SID index to: {output_file}")


if __name__ == "__main__":
    main()
