"""
Defer 训练脚本 - PyTorch 版本

用法:
    # 预训练
    python train.py --method Pretrain --mode pretrain --data_path ../data/business_data.txt
    
    # 流式训练
    python train.py --method ES-DFM --mode stream --data_path ../data/business_data.txt
"""

import argparse
import os
import sys
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from models import get_model
from loss import get_loss_fn
from data import load_data, get_stream_data, DataDF, SECONDS_AN_HOUR
from metrics import cal_auc, cal_prauc, cal_llloss, cal_ece, cal_pcoc


# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")


def train_epoch(model, dataloader, optimizer, loss_fn, params):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        num_feats = batch['num_feats'].to(DEVICE)
        cate_feats = batch['cate_feats'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(num_feats, cate_feats)
        targets = {"label": labels}
        
        loss_dict = loss_fn(targets, outputs, params)
        loss = loss_dict["loss"]
        
        # L2 正则化
        loss = loss + model.get_l2_loss()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, params, sample_ratio=0.2):
    """评估模型 (内存优化版 - 采样评估)"""
    import gc
    import random
    model.eval()
    
    # 采样评估，减少内存占用
    total_batches = len(dataloader)
    sample_every = max(1, int(1.0 / sample_ratio))
    
    all_probs = []
    all_labels = []
    all_business_types = []  # 记录 business_type 用于分组评估
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # 只评估部分 batch (采样)
            if i % sample_every != 0:
                continue
                
            cate_feats = batch['cate_feats'].to(DEVICE)
            labels = batch['label']
            
            outputs = model(None, cate_feats)
            # 对于 winadapt 等多输出模型，使用 cv_logits 作为主预测
            if "cv_logits" in outputs:
                logits = outputs["cv_logits"]
            else:
                logits = outputs["logits"]
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            all_probs.extend(probs.tolist())
            
            # 处理标签维度
            if labels.dim() > 1:
                labels = labels[:, 0]
            all_labels.extend(labels.numpy().flatten().tolist())
            
            # 记录 business_type (cate_feats 第 0 列)
            bt = batch['cate_feats'][:, 0].numpy().flatten()
            all_business_types.extend(bt.tolist())
            
            # 及时释放
            del cate_feats, outputs, logits, probs
            
            # 每 100 个 batch 强制 gc
            if i % 100 == 0:
                gc.collect()
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_business_types = np.array(all_business_types)
    
    # 计算指标
    auc = cal_auc(all_labels, all_probs)
    prauc = cal_prauc(all_labels, all_probs)
    llloss = cal_llloss(all_labels, all_probs)
    ece = cal_ece(all_labels, all_probs)
    pcoc = cal_pcoc(all_labels, all_probs)
    
    # 分组评估 (按 business_type 窗口长度)
    # 短窗口 (≤24h), 中窗口 (24-72h), 长窗口 (>72h)
    SHORT_BTS = [39, 48, 45, 24, 49, 37, 7, 56, 31, 20, 62, 17, 59]
    MEDIUM_BTS = [12, 47, 53]
    LONG_BTS = [60, 22, 43]
    
    group_metrics = {}
    for group_name, group_bts in [('short', SHORT_BTS), ('medium', MEDIUM_BTS), ('long', LONG_BTS)]:
        mask = np.isin(all_business_types, group_bts)
        if mask.sum() > 100:
            g_probs = all_probs[mask]
            g_labels = all_labels[mask]
            group_metrics[f'{group_name}_auc'] = cal_auc(g_labels, g_probs)
            group_metrics[f'{group_name}_n'] = int(mask.sum())
    
    # 清理内存
    del all_probs, all_labels, all_business_types
    gc.collect()
    
    result = {
        "auc": auc,
        "prauc": prauc,
        "llloss": llloss,
        "ece": ece,
        "pcoc": pcoc,
    }
    result.update(group_metrics)
    return result


def pretrain(params):
    """预训练"""
    print("\n" + "="*60)
    print(f"预训练: {params['method']}")
    print("="*60)
    
    # 加载数据
    data = load_data(params['data_path'], params.get('cache_path'))
    
    # 划分训练/测试集
    pre_train_start = params.get('pre_train_start', 0)
    pre_train_end = params.get('pre_train_end', 30)
    pre_test_start = params.get('pre_test_start', 30)
    pre_test_end = params.get('pre_test_end', 40)
    
    train_data = data.sub_days(pre_train_start, pre_train_end)
    test_data = data.sub_days(pre_test_start, pre_test_end)
    
    print(f"训练集: {len(train_data)} 条 (第 {pre_train_start}-{pre_train_end} 天)")
    print(f"测试集: {len(test_data)} 条 (第 {pre_test_start}-{pre_test_end} 天)")
    
    # 构造 DataLoader
    train_loader = train_data.to_dataloader(
        batch_size=params['batch_size'],
        shuffle=True
    )
    test_loader = test_data.to_dataloader(
        batch_size=params['batch_size'],
        shuffle=False
    )
    
    # 创建模型
    model_name = params.get('model', 'MLP_SIG')
    model = get_model(model_name, params.get('l2_reg', 1e-6)).to(DEVICE)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    
    # 损失函数
    loss_fn = get_loss_fn(params.get('loss', 'cross_entropy_loss'))
    
    # 训练
    best_auc = 0
    for epoch in range(params['epoch']):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, params)
        
        # 评估 (强制 GC 避免内存累积)
        import gc
        gc.collect()
        metrics = evaluate(model, test_loader, params)
        gc.collect()
        
        print(f"Epoch {epoch+1}/{params['epoch']}: "
              f"loss={train_loss:.4f}, "
              f"auc={metrics['auc']:.4f}, "
              f"prauc={metrics['prauc']:.4f}, "
              f"llloss={metrics['llloss']:.4f}")
        
        # 保存最佳模型
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            save_path = params.get('save_path') or params.get('model_ckpt_path')
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"  模型已保存: {save_path}")
    
    print(f"\n预训练完成, 最佳 AUC: {best_auc:.4f}")
    return model


def stream_train(params):
    """流式训练"""
    print("\n" + "="*60)
    print(f"流式训练: {params['method']}")
    print("="*60)
    
    # 加载数据
    data = load_data(params['data_path'], params.get('cache_path'))
    
    # 获取流式数据
    train_stream, test_stream, test_stream_nowin = get_stream_data(data, params['method'], params)
    print(f"训练流: {len(train_stream)} 个小时")
    print(f"测试流: {len(test_stream)} 个小时")
    print(f"nowin 测试流: {len(test_stream_nowin)} 个小时")
    
    # 创建模型
    model_name = get_model_name(params['method'])
    model = get_model(model_name, params.get('l2_reg', 1e-6)).to(DEVICE)
    
    # 加载预训练权重 (允许部分匹配)
    pretrain_path = get_pretrain_path(params)
    if pretrain_path and Path(pretrain_path).exists():
        print(f"加载预训练权重: {pretrain_path}")
        pretrain_state = torch.load(pretrain_path, map_location=DEVICE)
        model_state = model.state_dict()
        # 只加载匹配的权重
        matched = {k: v for k, v in pretrain_state.items() 
                   if k in model_state and v.shape == model_state[k].shape}
        model_state.update(matched)
        model.load_state_dict(model_state)
        print(f"  加载了 {len(matched)}/{len(pretrain_state)} 个权重")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    
    # 损失函数
    loss_fn = get_loss_fn(get_loss_name(params['method']))
    
    # 指标移动平均
    metrics_ma = {
        'auc': [], 'prauc': [], 'llloss': [], 'ece': [], 'pcoc': [],
        'ctr': [], 'pctr': [],
        'short_auc': [], 'medium_auc': [], 'long_auc': [],
        'nowin_auc': [], 'nowin_prauc': [], 'nowin_llloss': [], 'nowin_ece': []
    }

    # 流式训练
    for epoch, (train_hour, test_hour, nowin_hour) in enumerate(zip(train_stream, test_stream, test_stream_nowin)):
        # 训练
        train_loader = train_hour.to_dataloader(
            batch_size=params['batch_size'],
            shuffle=False  # 流式训练不 shuffle
        )
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, params)
        
        # 评估
        test_loader = test_hour.to_dataloader(
            batch_size=params['batch_size'],
            shuffle=False
        )
        metrics = evaluate(model, test_loader, params)

        # nowin 评估 (对齐 TF 原版 test_stream_nowin)
        nowin_loader = nowin_hour.to_dataloader(
            batch_size=params['batch_size'],
            shuffle=False
        )
        nowin_metrics = evaluate(model, nowin_loader, params)

        # 更新移动平均
        for k, v in metrics.items():
            if k in metrics_ma:
                metrics_ma[k].append(v)
        for k in ['auc', 'prauc', 'llloss', 'ece']:
            if k in nowin_metrics:
                metrics_ma[f'nowin_{k}'].append(nowin_metrics[k])
        
        # 打印
        if (epoch + 1) % 10 == 0 or epoch == 0:
            ma_auc = np.mean(metrics_ma['auc'][-100:])
            ma_prauc = np.mean(metrics_ma['prauc'][-100:])
            ma_llloss = np.mean(metrics_ma['llloss'][-100:])
            ma_ece = np.mean(metrics_ma['ece'][-100:])
            ma_pcoc = np.mean(metrics_ma['pcoc'][-100:])
            
            print(f"Epoch {epoch+1}: "
                  f"auc={metrics['auc']:.4f} (ma={ma_auc:.4f}), "
                  f"prauc={metrics['prauc']:.4f} (ma={ma_prauc:.4f}), "
                  f"llloss={metrics['llloss']:.4f} (ma={ma_llloss:.4f}), "
                  f"ece={metrics['ece']:.4f} (ma={ma_ece:.4f}), "
                  f"pcoc={metrics['pcoc']:.4f} (ma={ma_pcoc:.4f})")
    
    # 最终结果 (过滤掉 inf 值)
    def safe_mean(arr):
        arr = np.array(arr)
        valid = arr[np.isfinite(arr)]
        return np.mean(valid) if len(valid) > 0 else float('nan')
    
    print("\n" + "="*60)
    print("最终结果 (移动平均):")
    print(f"  AUC:     {safe_mean(metrics_ma['auc']):.4f}")
    print(f"  PR-AUC:  {safe_mean(metrics_ma['prauc']):.4f}")
    print(f"  LogLoss: {safe_mean(metrics_ma['llloss']):.4f}")
    print(f"  ECE:     {safe_mean(metrics_ma['ece']):.4f}")
    print(f"  PCOC:    {safe_mean(metrics_ma['pcoc']):.4f}")
    print("-"*60)
    print("分组 AUC (按延迟窗口):")
    print(f"  短窗口 (≤24h):  {safe_mean(metrics_ma['short_auc']):.4f}")
    print(f"  中窗口 (24-72h): {safe_mean(metrics_ma['medium_auc']):.4f}")
    print(f"  长窗口 (>72h):  {safe_mean(metrics_ma['long_auc']):.4f}")
    print("-"*60)
    print("nowin 评估 (含延迟正样本):")
    print(f"  nowin AUC:     {safe_mean(metrics_ma['nowin_auc']):.4f}")
    print(f"  nowin PR-AUC:  {safe_mean(metrics_ma['nowin_prauc']):.4f}")
    print(f"  nowin LogLoss: {safe_mean(metrics_ma['nowin_llloss']):.4f}")
    print(f"  nowin ECE:     {safe_mean(metrics_ma['nowin_ece']):.4f}")
    print("="*60)
    
    return model


def get_model_name(method):
    """根据方法获取模型名称"""
    method_to_model = {
        "Pretrain": "MLP_SIG",
        "Vanilla": "MLP_SIG",
        "Oracle": "MLP_SIG",
        "FNW": "MLP_SIG",
        "FNC": "MLP_SIG",
        "DFM": "MLP_EXP_DELAY",
        "ES-DFM": "MLP_tn_dp",
        "FSIW": "MLP_FSIW",
        "3class": "MLP_3class",
        "delay_win_time": "MLP_likeli",
        "delay_win_adapt": "MLP_winadapt",
    }
    return method_to_model.get(method, "MLP_SIG")


def get_loss_name(method):
    """根据方法获取损失函数名称"""
    method_to_loss = {
        "Pretrain": "cross_entropy_loss",
        "Vanilla": "cross_entropy_loss",
        "Oracle": "cross_entropy_loss",
        "FNW": "fnw_loss",
        "FNC": "fnc_loss",
        "DFM": "exp_delay_loss",
        "ES-DFM": "esdfm_importance_weight_loss",
        "FSIW": "fsiw_loss",
        "3class": "delay_3class_loss",
        "delay_win_time": "delay_win_time_loss",
        "delay_win_adapt": "delay_win_select_loss",
    }
    return method_to_loss.get(method, "cross_entropy_loss")


def get_pretrain_path(params):
    """获取预训练模型路径"""
    method = params['method']
    
    if method in ["Vanilla", "Oracle", "FNW", "FNC"]:
        return params.get('pretrain_baseline_path')
    elif method == "ES-DFM":
        return params.get('pretrain_esdfm_path')
    elif method == "DFM":
        return params.get('pretrain_dfm_path')
    elif method == "delay_win_adapt":
        return params.get('pretrain_winadapt_path')
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Defer PyTorch 训练')
    
    # 基础参数
    parser.add_argument('--method', type=str, required=True,
                        choices=['Pretrain', 'Vanilla', 'Oracle', 'FNW', 'FNC',
                                'DFM', 'ES-DFM', 'FSIW', '3class',
                                'delay_win_time', 'delay_win_adapt'],
                        help='训练方法')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['pretrain', 'stream'],
                        help='训练模式')
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据文件路径')
    
    # 模型参数
    parser.add_argument('--model_ckpt_path', type=str, default=None,
                        help='模型保存路径')
    parser.add_argument('--pretrain_baseline_path', type=str, default=None,
                        help='Baseline 预训练模型路径')
    parser.add_argument('--pretrain_esdfm_path', type=str, default=None,
                        help='ES-DFM 预训练模型路径')
    parser.add_argument('--pretrain_dfm_path', type=str, default=None,
                        help='DFM 预训练模型路径')
    parser.add_argument('--pretrain_winadapt_path', type=str, default=None,
                        help='WinAdapt 预训练模型路径')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_reg', type=float, default=1e-6)
    parser.add_argument('--C', type=float, default=6,
                        help='观察窗口大小 (小时)')
    parser.add_argument('--win1', type=float, default=6,
                        help='第一档窗口 (小时)')
    parser.add_argument('--win2', type=float, default=24,
                        help='第二档窗口 (小时)')
    parser.add_argument('--win3', type=float, default=72,
                        help='第三档窗口 (小时)')
    
    # 时间参数 - 预训练
    parser.add_argument('--pre_train_start', type=int, default=0)
    parser.add_argument('--pre_train_end', type=int, default=30)
    parser.add_argument('--pre_test_start', type=int, default=30)
    parser.add_argument('--pre_test_end', type=int, default=40)
    # 时间参数 - 流式训练
    parser.add_argument('--stream_start', type=int, default=30)
    parser.add_argument('--stream_mid', type=int, default=30)
    parser.add_argument('--stream_end', type=int, default=60)
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None,
                        help='预训练模型保存路径')
    
    args = parser.parse_args()
    params = vars(args)
    
    # 设置随机种子
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    
    print("参数配置:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # 运行
    if params['mode'] == 'pretrain':
        pretrain(params)
    else:
        stream_train(params)


if __name__ == '__main__':
    main()
