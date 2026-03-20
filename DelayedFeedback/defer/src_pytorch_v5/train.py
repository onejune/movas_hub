#!/usr/bin/env python3
"""
Defer PyTorch v4 训练脚本

v4 严格对齐 TF 原版 delay_win_time_loss + delay_win_select_loss:
1. 多窗口独立建模: 24h/48h/72h 三个窗口
2. 每个窗口有 loss_win_now + loss_win_spm
3. 全局 loss_cv_spm 建模最终转化
4. TF 原版的 stop_gradient 策略

使用 parquet 数据 (249 特征)，与 TF v2 版本对齐
支持方法: Vanilla, Oracle, FNW, FNC, DFM, ES-DFM, WinAdapt
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from tqdm import tqdm
from datetime import datetime

from data import load_data, prepare_data, create_dataloader, TIME_WINDOWS
from models import get_model
from loss import (cross_entropy_loss, fnw_loss, fnc_loss, dfm_loss, 
                  esdfm_importance_weight_loss, winadapt_loss, winadapt_loss_v4)


# 配置
DATA_DIR = "/mnt/workspace/walter.wan/open_research/defer/data_v4"  # v3 数据含 delay_time, dp_label 等
LOG_DIR = "/mnt/workspace/walter.wan/open_research/defer/logs_pytorch_v5"

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 所有支持的方法
ALL_METHODS = ['vanilla', 'oracle', 'fnw', 'fnc', 'dfm', 'esdfm', 'winadapt']


def train_epoch_simple(model, dataloader, optimizer, device):
    """训练一个 epoch - 简单方法 (Vanilla, Oracle)"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        cate_feats = batch['cate_feats'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(cate_feats)
        logits = logits.squeeze(-1)
        
        loss = cross_entropy_loss(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def train_epoch_fnw(model, dataloader, optimizer, device, C=24.0):
    """训练一个 epoch - FNW/FNC 方法"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Training FNW"):
        cate_feats = batch['cate_feats'].to(device)
        labels = batch['label']
        
        label_vals = labels['label'].to(device)
        elapsed = labels['elapsed_time'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(cate_feats).squeeze(-1)
        loss = fnw_loss(logits, label_vals, elapsed, C=C)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def train_epoch_dfm(model, dataloader, optimizer, device):
    """训练一个 epoch - DFM 方法"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Training DFM"):
        cate_feats = batch['cate_feats'].to(device)
        labels = batch['label']
        
        label_vals = labels['label'].to(device)
        delay_time = labels['delay_time'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(cate_feats)
        loss = dfm_loss(outputs['cv_logits'], outputs['log_lamb'], label_vals, delay_time)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def train_epoch_esdfm_pretrain(model, dataloader, optimizer, device):
    """
    ES-DFM 第一阶段: 预训练 tn/dp 分类器
    
    目标: 学会区分真负样本 (tn) 和延迟正样本 (dp)
    - tn_logits: 预测是否为真负样本
    - dp_logits: 预测是否为延迟正样本
    """
    from loss import esdfm_pretrain_loss
    
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="ES-DFM Pretrain (Stage 1)"):
        cate_feats = batch['cate_feats'].to(device)
        labels = batch['label']
        
        tn_label = labels['tn_label'].to(device)
        dp_label = labels['dp_label'].to(device)
        pos_label = labels['pos_label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(cate_feats)
        
        loss = esdfm_pretrain_loss(
            outputs['tn_logits'], outputs['dp_logits'],
            tn_label, dp_label, pos_label
        )
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def train_epoch_esdfm_finetune(model, dataloader, optimizer, device):
    """
    ES-DFM 第二阶段: 用 importance weight 训练 CVR
    
    此时 tn/dp 分类器已冻结，只更新 cv_logits 相关参数
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="ES-DFM Finetune (Stage 2)"):
        cate_feats = batch['cate_feats'].to(device)
        labels = batch['label']
        
        label_vals = labels['label'].to(device)
        pos_label = labels['pos_label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(cate_feats)
        
        # tn/dp 已冻结，detach 防止梯度回传
        tn_prob = torch.sigmoid(outputs['tn_logits'].detach())
        dp_prob = torch.sigmoid(outputs['dp_logits'].detach())
        
        loss = esdfm_importance_weight_loss(
            outputs['cv_logits'], tn_prob, dp_prob, label_vals, pos_label
        )
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def train_epoch_esdfm(model, dataloader, optimizer, device):
    """训练一个 epoch - ES-DFM 方法 (旧版单阶段，保留兼容)"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Training ES-DFM"):
        cate_feats = batch['cate_feats'].to(device)
        labels = batch['label']
        
        label_vals = labels['label'].to(device)
        tn_label = labels['tn_label'].to(device)
        dp_label = labels['dp_label'].to(device)
        pos_label = labels['pos_label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(cate_feats)
        
        # ES-DFM 使用重要性加权
        tn_prob = torch.sigmoid(outputs['tn_logits'])
        dp_prob = torch.sigmoid(outputs['dp_logits'])
        
        loss = esdfm_importance_weight_loss(
            outputs['cv_logits'], tn_prob, dp_prob, label_vals, pos_label
        )
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def train_epoch_winadapt(model, dataloader, optimizer, device):
    """WinAdapt 训练一个 epoch - 4 输出头版本"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Training WinAdapt"):
        cate_feats = batch['cate_feats'].to(device)
        labels = batch['label']  # dict
        
        # 将标签移到设备
        labels_device = {k: v.to(device) for k, v in labels.items()}
        
        optimizer.zero_grad()
        
        # WinAdapt 模型输出是 dict
        outputs = model(cate_feats)
        loss = winadapt_loss(outputs, labels_device)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, device, method='oracle'):
    """评估模型"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            cate_feats = batch['cate_feats'].to(device)
            labels = batch['label']
            
            # 处理 labels
            if isinstance(labels, dict):
                labels = labels.get('label', labels.get('label_oracle', labels.get('label_168h')))
            
            outputs = model(cate_feats)
            
            # 获取 cv_logits
            if isinstance(outputs, dict):
                logits = outputs['cv_logits']
            else:
                logits = outputs.squeeze(-1)
            
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # 计算指标
    auc = roc_auc_score(all_labels, all_probs)
    prauc = average_precision_score(all_labels, all_probs)
    logloss = log_loss(all_labels, np.clip(all_probs, 1e-7, 1-1e-7))
    
    return {
        'auc': auc,
        'prauc': prauc,
        'logloss': logloss,
    }


def train_model_esdfm_twostage(train_df, test_df, feature_cols, encoder, args):
    """
    ES-DFM 两阶段训练
    
    Stage 1: 预训练 tn/dp 分类器 (学会区分真负 vs 延迟正)
    Stage 2: 固定 tn/dp，用 importance weight 训练 CVR
    """
    print("\n" + "="*60)
    print("训练 ES-DFM 模型 (两阶段)")
    print("="*60)
    
    start_time = time.time()
    
    # 准备数据
    train_features, train_labels, test_features, test_labels = prepare_data(
        train_df, test_df, feature_cols, encoder, 
        method='esdfm', window=args.window
    )
    
    # 创建 DataLoader
    train_loader = create_dataloader(
        train_features, train_labels, 
        batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, method='esdfm'
    )
    test_loader = create_dataloader(
        test_features, test_labels,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, method='oracle'
    )
    
    # 创建模型
    vocab_sizes = encoder.get_vocab_sizes_list(feature_cols)
    model = get_model('esdfm', vocab_sizes, embed_dim=args.embed_dim)
    model = model.to(DEVICE)
    
    results_history = []
    best_auc = 0.0
    
    # ========================================
    # Stage 1: 预训练 tn/dp 分类器
    # ========================================
    print("\n--- Stage 1: 预训练 tn/dp 分类器 ---")
    
    # Stage 1 优化器 (训练所有参数)
    optimizer_stage1 = optim.Adam(model.parameters(), lr=args.lr)
    
    pretrain_epochs = getattr(args, 'esdfm_pretrain_epochs', 1)
    for epoch in range(pretrain_epochs):
        epoch_start = time.time()
        
        train_loss = train_epoch_esdfm_pretrain(model, train_loader, optimizer_stage1, DEVICE)
        
        # 评估 (此阶段 cv_logits 未训练，AUC 可能较低)
        metrics = evaluate(model, test_loader, DEVICE, method='esdfm')
        epoch_time = time.time() - epoch_start
        
        print(f"Stage1 Epoch {epoch+1}/{pretrain_epochs}: "
              f"loss={train_loss:.4f}, "
              f"AUC={metrics['auc']:.4f}, "
              f"PR-AUC={metrics['prauc']:.4f}, "
              f"time={epoch_time:.1f}s")
        
        results_history.append({
            'stage': 1,
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **metrics,
            'epoch_time': epoch_time,
        })
    
    # ========================================
    # Stage 2: 固定 tn/dp，训练 CVR
    # ========================================
    print("\n--- Stage 2: 固定 tn/dp，训练 CVR ---")
    
    # 冻结 tn/dp 相关参数 (output_layer 的后两个输出)
    # ESDFMModel 继承自 DeferModel，output_layer 是 Linear(64, 3)
    # logits[:, 0] = cv, logits[:, 1] = tn, logits[:, 2] = dp
    # 我们需要冻结 tn/dp 的梯度，但共享 embedding 和 MLP 仍需更新
    # 方案: 在 train_epoch_esdfm_finetune 中对 tn/dp logits 做 detach
    
    # Stage 2 优化器 (仍然训练所有参数，但 loss 只来自 cv_logits)
    optimizer_stage2 = optim.Adam(model.parameters(), lr=args.lr)
    
    finetune_epochs = getattr(args, 'esdfm_finetune_epochs', 1)
    for epoch in range(finetune_epochs):
        epoch_start = time.time()
        
        train_loss = train_epoch_esdfm_finetune(model, train_loader, optimizer_stage2, DEVICE)
        
        metrics = evaluate(model, test_loader, DEVICE, method='esdfm')
        epoch_time = time.time() - epoch_start
        
        print(f"Stage2 Epoch {epoch+1}/{finetune_epochs}: "
              f"loss={train_loss:.4f}, "
              f"AUC={metrics['auc']:.4f}, "
              f"PR-AUC={metrics['prauc']:.4f}, "
              f"LogLoss={metrics['logloss']:.4f}, "
              f"time={epoch_time:.1f}s")
        
        results_history.append({
            'stage': 2,
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **metrics,
            'epoch_time': epoch_time,
        })
        
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
    
    total_time = time.time() - start_time
    
    return {
        'method': 'esdfm',
        'best_auc': best_auc,
        'final_metrics': metrics,
        'total_time': total_time,
        'history': results_history,
    }


def train_model(method, train_df, test_df, feature_cols, encoder, args):
    """训练模型"""
    # ES-DFM 使用两阶段训练
    if method.lower() == 'esdfm':
        return train_model_esdfm_twostage(train_df, test_df, feature_cols, encoder, args)
    
    print("\n" + "="*60)
    print(f"训练 {method.upper()} 模型")
    print("="*60)
    
    start_time = time.time()
    method_lower = method.lower()
    
    # 准备数据
    train_features, train_labels, test_features, test_labels = prepare_data(
        train_df, test_df, feature_cols, encoder, 
        method=method_lower, window=args.window
    )
    
    # 创建 DataLoader
    train_loader = create_dataloader(
        train_features, train_labels, 
        batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, method=method_lower
    )
    test_loader = create_dataloader(
        test_features, test_labels,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, method='oracle'
    )
    
    # 创建模型
    vocab_sizes = encoder.get_vocab_sizes_list(feature_cols)
    model = get_model(method_lower, vocab_sizes, embed_dim=args.embed_dim)
    model = model.to(DEVICE)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 选择训练函数
    train_fn_map = {
        'vanilla': train_epoch_simple,
        'oracle': train_epoch_simple,
        'fnw': lambda m, d, o, dev: train_epoch_fnw(m, d, o, dev, C=args.window),
        'fnc': lambda m, d, o, dev: train_epoch_fnw(m, d, o, dev, C=args.window),
        'dfm': train_epoch_dfm,
        'esdfm': train_epoch_esdfm,  # 不会走到这里，但保留兼容
        'winadapt': train_epoch_winadapt,
    }
    train_fn = train_fn_map[method_lower]
    
    # 训练
    best_auc = 0.0
    results_history = []
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss = train_fn(model, train_loader, optimizer, DEVICE)
        
        # 评估
        metrics = evaluate(model, test_loader, DEVICE, method=method_lower)
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"loss={train_loss:.4f}, "
              f"AUC={metrics['auc']:.4f}, "
              f"PR-AUC={metrics['prauc']:.4f}, "
              f"LogLoss={metrics['logloss']:.4f}, "
              f"time={epoch_time:.1f}s")
        
        results_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **metrics,
            'epoch_time': epoch_time,
        })
        
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
    
    total_time = time.time() - start_time
    
    return {
        'method': method,
        'best_auc': best_auc,
        'final_metrics': metrics,
        'total_time': total_time,
        'history': results_history,
    }


def main():
    parser = argparse.ArgumentParser(description='Defer PyTorch v2 Training')
    parser.add_argument('--method', type=str, default='all',
                        choices=['vanilla', 'oracle', 'fnw', 'fnc', 'dfm', 'esdfm', 'winadapt', 'all'])
    parser.add_argument('--window', type=int, default=24,
                        help='Window for vanilla/fnw/fnc method (hours)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs (default: 1)')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embed_dim', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    # ES-DFM 两阶段训练参数
    parser.add_argument('--esdfm_pretrain_epochs', type=int, default=1,
                        help='ES-DFM Stage 1 (pretrain tn/dp) epochs')
    parser.add_argument('--esdfm_finetune_epochs', type=int, default=1,
                        help='ES-DFM Stage 2 (finetune CVR) epochs')
    args = parser.parse_args()
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 记录开始时间
    run_start = datetime.now()
    print(f"\n{'='*60}")
    print(f"Defer PyTorch v2 Training")
    print(f"开始时间: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {DEVICE}")
    print(f"{'='*60}")
    
    # 加载数据
    encoder_path = os.path.join(LOG_DIR, 'encoder.json')
    train_df, test_df, feature_cols, label_cols, encoder, meta = load_data(
        DATA_DIR, encoder_path=encoder_path
    )
    
    # 保存编码器
    if not os.path.exists(encoder_path):
        encoder.save(encoder_path)
    
    results = {}
    
    # 训练各方法
    if args.method == 'all':
        methods = ALL_METHODS
    else:
        methods = [args.method]
    
    for method in methods:
        try:
            result = train_model(method, train_df, test_df, feature_cols, encoder, args)
            results[method] = result
        except Exception as e:
            print(f"\n❌ {method.upper()} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            results[method] = {'method': method, 'error': str(e)}
    
    # 打印结果汇总
    run_end = datetime.now()
    total_run_time = (run_end - run_start).total_seconds()
    
    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)
    print(f"\n数据信息:")
    print(f"  训练样本: {meta['train_rows']:,}")
    print(f"  测试样本: {meta['test_rows']:,}")
    print(f"  特征数: {meta['num_features']}")
    print(f"  时间窗口: {meta['time_windows']}")
    
    print(f"\n训练配置:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Embedding dim: {args.embed_dim}")
    
    print(f"\n模型结果:")
    print(f"{'Method':<12} {'AUC':<10} {'PR-AUC':<10} {'LogLoss':<10} {'Time':<10}")
    print("-" * 52)
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    for method, result in sorted(valid_results.items(), key=lambda x: -x[1].get('best_auc', 0)):
        m = result.get('final_metrics', {})
        t = result.get('total_time', 0)
        print(f"{method:<12} {m.get('auc', 0):<10.4f} {m.get('prauc', 0):<10.4f} {m.get('logloss', 0):<10.4f} {t:<10.1f}s")
    
    # 打印失败的方法
    failed = {k: v for k, v in results.items() if 'error' in v}
    if failed:
        print(f"\n失败的方法:")
        for method, result in failed.items():
            print(f"  {method}: {result['error']}")
    
    print(f"\n总耗时: {total_run_time:.1f}s ({total_run_time/60:.1f}min)")
    print(f"结束时间: {run_end.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 保存结果
    output = {
        'run_start': run_start.isoformat(),
        'run_end': run_end.isoformat(),
        'total_time': total_run_time,
        'device': str(DEVICE),
        'args': vars(args),
        'data_info': meta,
        'results': {k: {
            'method': v['method'],
            'best_auc': v.get('best_auc'),
            'final_metrics': v.get('final_metrics'),
            'total_time': v.get('total_time'),
            'error': v.get('error'),
        } for k, v in results.items()},
    }
    
    result_path = os.path.join(LOG_DIR, 'results.json')
    with open(result_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {result_path}")


if __name__ == '__main__':
    main()
