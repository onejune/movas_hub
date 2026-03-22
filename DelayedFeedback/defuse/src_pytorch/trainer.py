"""
DEFUSE 训练器

支持:
- 天级增量训练
- 两阶段训练 (Pretrain + Finetune)
- 多方法对比 (DEFUSE vs ES-DFM)
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from .models import DEFUSEModel
from .loss import pretrain_loss, defuse_loss, esdfm_loss
from .data import IncrementalDataLoader


class Trainer:
    """DEFUSE 训练器"""
    
    def __init__(self,
                 data_loader: IncrementalDataLoader,
                 output_dir: str,
                 batch_size: int = 4096,
                 learning_rate: float = 0.001,
                 num_workers: int = 4,
                 device: str = 'cpu'):
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.device = torch.device(device)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型和优化器
        self.model: Optional[DEFUSEModel] = None
        self.optimizer: Optional[optim.Optimizer] = None
        
        # 训练状态
        self.current_method = None
        self.train_history = []
        self.results = {}
    
    def _create_model(self) -> DEFUSEModel:
        """创建模型"""
        vocab_sizes = self.data_loader.get_vocab_sizes()
        model = DEFUSEModel(vocab_sizes).to(self.device)
        print(f"Model created: {model.get_num_params():,} parameters")
        return model
    
    def _create_dataloader(self, X: np.ndarray, y: Dict[str, np.ndarray], 
                          shuffle: bool = True) -> DataLoader:
        """创建 DataLoader"""
        tensors = [torch.from_numpy(X)]
        tensors.append(torch.from_numpy(y['label']))
        
        if 'tn_label' in y:
            tensors.append(torch.from_numpy(y['tn_label']))
            tensors.append(torch.from_numpy(y['dp_label']))
        
        dataset = TensorDataset(*tensors)
        return DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def _train_pretrain_epoch(self, dataloader: DataLoader) -> float:
        """训练一个 pretrain epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Pretrain")
        for batch in pbar:
            x = batch[0].to(self.device)
            tn_label = batch[2].to(self.device)
            dp_label = batch[3].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x)
            
            loss = pretrain_loss(
                outputs['tn_logits'], 
                outputs['dp_logits'],
                tn_label, 
                dp_label
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        return total_loss / max(num_batches, 1)
    
    def _train_finetune_epoch(self, dataloader: DataLoader, 
                              loss_fn: Callable) -> float:
        """训练一个 finetune epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        method_name = "DEFUSE" if loss_fn == defuse_loss else "ES-DFM"
        pbar = tqdm(dataloader, desc=method_name)
        
        for batch in pbar:
            x = batch[0].to(self.device)
            label = batch[1].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x)
            
            # 使用 detached tn/dp 概率
            with torch.no_grad():
                tn_prob = torch.sigmoid(outputs['tn_logits'])
                dp_prob = torch.sigmoid(outputs['dp_logits'])
            
            loss = loss_fn(outputs['cv_logits'], tn_prob, dp_prob, label)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        return total_loss / max(num_batches, 1)
    
    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                x = batch[0].to(self.device)
                label = batch[1]
                
                outputs = self.model(x)
                preds = torch.sigmoid(outputs['cv_logits']).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(label.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 计算指标
        auc = roc_auc_score(all_labels, all_preds)
        pr_auc = average_precision_score(all_labels, all_preds)
        logloss = log_loss(all_labels, np.clip(all_preds, 1e-7, 1-1e-7))
        
        return {
            'auc': auc,
            'pr_auc': pr_auc,
            'logloss': logloss
        }
    
    def train_incremental(self, 
                          method: str = 'defuse',
                          pretrain_epochs: int = 1,
                          finetune_epochs: int = 1) -> Dict[str, float]:
        """
        天级增量训练
        
        Args:
            method: 'defuse' 或 'esdfm'
            pretrain_epochs: 预训练 epochs
            finetune_epochs: 微调 epochs
        
        Returns:
            评估指标
        """
        self.current_method = method
        loss_fn = defuse_loss if method == 'defuse' else esdfm_loss
        
        print(f"\n{'='*60}")
        print(f"Training {method.upper()} (incremental)")
        print(f"{'='*60}")
        
        # 创建模型
        self.model = self._create_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Stage 1: Pretrain (天级增量)
        print(f"\n--- Stage 1: Pretrain tn/dp heads ({pretrain_epochs} epoch) ---")
        for epoch in range(pretrain_epochs):
            batch_idx = 0
            for X, y in self.data_loader.iterate_train():
                batch_idx += 1
                print(f"\nPretrain batch {batch_idx}")
                
                dataloader = self._create_dataloader(X, y, shuffle=True)
                loss = self._train_pretrain_epoch(dataloader)
                
                self.train_history.append({
                    'stage': 'pretrain',
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': loss
                })
                print(f"Pretrain batch {batch_idx} loss: {loss:.4f}")
        
        # Stage 2: Finetune CVR (天级增量)
        print(f"\n--- Stage 2: Finetune CVR with {method.upper()} ({finetune_epochs} epoch) ---")
        for epoch in range(finetune_epochs):
            batch_idx = 0
            for X, y in self.data_loader.iterate_train():
                batch_idx += 1
                print(f"\nFinetune batch {batch_idx}")
                
                dataloader = self._create_dataloader(X, y, shuffle=True)
                loss = self._train_finetune_epoch(dataloader, loss_fn)
                
                self.train_history.append({
                    'stage': 'finetune',
                    'method': method,
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': loss
                })
                print(f"Finetune batch {batch_idx} loss: {loss:.4f}")
        
        # 评估
        print("\n--- Evaluation ---")
        X_test, y_test = self.data_loader.load_test()
        test_loader = self._create_dataloader(X_test, y_test, shuffle=False)
        metrics = self._evaluate(test_loader)
        
        print(f"\nResults ({method.upper()}):")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"  LogLoss: {metrics['logloss']:.4f}")
        
        # 保存结果
        self.results[method] = metrics
        self._save_results()
        self._save_model(method)
        
        return metrics
    
    def compare_methods(self, 
                        methods: List[str] = None,
                        pretrain_epochs: int = 1,
                        finetune_epochs: int = 1) -> Dict[str, Dict[str, float]]:
        """
        对比多个方法
        
        Args:
            methods: 方法列表, 默认 ['defuse', 'esdfm']
            pretrain_epochs: 预训练 epochs
            finetune_epochs: 微调 epochs
        
        Returns:
            各方法的评估指标
        """
        methods = methods or ['defuse', 'esdfm']
        
        for method in methods:
            self.train_incremental(method, pretrain_epochs, finetune_epochs)
        
        # 打印对比结果
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Method':<15} {'AUC':>10} {'PR-AUC':>10} {'LogLoss':>10}")
        print("-"*60)
        
        for method, metrics in self.results.items():
            print(f"{method.upper():<15} {metrics['auc']:>10.4f} "
                  f"{metrics['pr_auc']:>10.4f} {metrics['logloss']:>10.4f}")
        
        return self.results
    
    def _save_results(self):
        """保存结果"""
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'results': self.results,
                'history': self.train_history,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"Results saved to {results_path}")
    
    def _save_model(self, method: str):
        """保存模型"""
        model_path = self.output_dir / f"model_{method}.pt"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
