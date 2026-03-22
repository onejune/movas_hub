#!/usr/bin/env python3
"""
DEFUSE Criteo Benchmark - Trainer

Implements pretraining and streaming training for all methods.
"""
import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset

from config import Config, SECONDS_A_DAY, SECONDS_AN_HOUR
from data import DataDF, load_criteo_data, get_streaming_data
from models import get_model
from loss import get_loss_fn
from metrics import evaluate, ScalarMovingAverage


class CriteoDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for Criteo data."""
    
    def __init__(self, data: DataDF, cat_hash_sizes: Tuple[int, ...]):
        self.num_features = torch.tensor(
            data.x[[str(i) for i in range(8)]].values, 
            dtype=torch.float32
        )
        
        # Hash categorical features
        cat_data = []
        for i, hash_size in enumerate(cat_hash_sizes):
            col = data.x[str(i + 8)].apply(lambda x: hash(str(x)) % hash_size)
            cat_data.append(col.values)
        self.cat_features = torch.tensor(np.stack(cat_data, axis=1), dtype=torch.long)
        
        self.labels = torch.tensor(data.labels, dtype=torch.float32)
        self.delay_labels = torch.tensor(
            data.delay_labels if data.delay_labels is not None else data.labels,
            dtype=torch.float32
        )
        self.inw_labels = torch.tensor(
            data.inw_labels if data.inw_labels is not None else data.labels,
            dtype=torch.float32
        )
        
        # For pretraining
        self.tn_labels = torch.tensor(
            (data.pay_ts < 0).astype(np.int32),
            dtype=torch.float32
        )
        self.dp_labels = torch.tensor(
            ((data.pay_ts > 0) & (data.pay_ts - data.click_ts > SECONDS_AN_HOUR)).astype(np.int32),
            dtype=torch.float32
        )
        self.pos_labels = torch.tensor(
            (data.pay_ts > 0).astype(np.int32),
            dtype=torch.float32
        )
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "num_features": self.num_features[idx],
            "cat_features": self.cat_features[idx],
            "label": self.labels[idx],
            "delay_label": self.delay_labels[idx],
            "inw_label": self.inw_labels[idx],
            "tn_label": self.tn_labels[idx],
            "dp_label": self.dp_labels[idx],
            "pos_label": self.pos_labels[idx],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }


class Trainer:
    """Trainer for DEFUSE methods."""
    
    def __init__(self, config: Config, device: str = "cpu"):
        self.config = config
        self.device = device
        self.cat_hash_sizes = config.data.cat_bin_sizes
        
        # Create output directory
        os.makedirs(config.data.output_dir, exist_ok=True)
    
    def pretrain(self, 
                 data: DataDF, 
                 model_type: str,
                 loss_name: str,
                 epochs: int = 1) -> nn.Module:
        """
        Pretrain auxiliary models (tn/dp, dp, dfm).
        
        Args:
            data: Training data
            model_type: Model type (MLP_tn_dp, MLP_dp, MLP_EXP_DELAY)
            loss_name: Loss function name
            epochs: Number of epochs
        
        Returns:
            Trained model
        """
        print(f"Pretraining {model_type} with {loss_name} loss...")
        
        # Create model
        model = get_model(
            model_type,
            hidden_dims=self.config.model.hidden_dims,
            embed_dim=self.config.data.embed_dim,
            use_bn=self.config.model.use_batch_norm,
            l2_reg=self.config.model.l2_reg
        ).to(self.device)
        
        # Create dataset and dataloader
        dataset = CriteoDataset(data, self.cat_hash_sizes)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.config.train.num_workers,
            collate_fn=collate_fn
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.model.l2_reg
        )
        
        # Loss function
        loss_fn = get_loss_fn(loss_name)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in tqdm(dataloader, desc=f"Pretrain Epoch {epoch+1}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                outputs = model(batch["num_features"], batch["cat_features"])
                loss = loss_fn(outputs, batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
        
        return model
    
    def train_epoch(self,
                    model: nn.Module,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn,
                    aux_models: Optional[Dict[str, nn.Module]] = None) -> float:
        """Train one epoch."""
        model.train()
        if aux_models:
            for m in aux_models.values():
                m.eval()
        
        total_loss = 0.0
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(batch["num_features"], batch["cat_features"])
            
            # Add auxiliary model outputs
            if aux_models:
                with torch.no_grad():
                    if "esdfm" in aux_models:
                        aux_out = aux_models["esdfm"](batch["num_features"], batch["cat_features"])
                        outputs["tn_logits"] = aux_out["tn_logits"]
                        outputs["dp_logits"] = aux_out["dp_logits"]
                    if "defer" in aux_models:
                        aux_out = aux_models["defer"](batch["num_features"], batch["cat_features"])
                        outputs["dp_logits"] = aux_out["logits"]
            
            loss = loss_fn(outputs, batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self,
                 model: nn.Module,
                 dataloader: DataLoader,
                 method: str = "default") -> Dict[str, float]:
        """Evaluate model."""
        model.eval()
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if hasattr(model, 'predict'):
                    outputs = model.predict(batch["num_features"], batch["cat_features"])
                    probs = outputs.get("prob", torch.sigmoid(outputs.get("logits", outputs["logits_inw"])))
                else:
                    outputs = model(batch["num_features"], batch["cat_features"])
                    probs = torch.sigmoid(outputs["logits"])
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch["label"].cpu().numpy())
        
        all_probs = np.concatenate(all_probs).flatten()
        all_labels = np.concatenate(all_labels).flatten()
        
        return evaluate(all_labels, all_probs, method)
    
    def stream_train(self,
                     data: DataDF,
                     method: str,
                     aux_models: Optional[Dict[str, nn.Module]] = None) -> Dict[str, float]:
        """
        Streaming training and evaluation.
        
        Args:
            data: Full dataset
            method: Method name
            aux_models: Auxiliary models (esdfm, defer)
        
        Returns:
            Final metrics
        """
        print(f"\nStreaming training for {method}...")
        
        # Determine model type
        if method == "Bi-DEFUSE":
            model_type = "Bi-DEFUSE"
        elif method == "DFM":
            model_type = "MLP_EXP_DELAY"
        else:
            model_type = "MLP_SIG"
        
        # Create model
        model = get_model(
            model_type,
            hidden_dims=self.config.model.hidden_dims,
            embed_dim=self.config.data.embed_dim,
            use_bn=self.config.model.use_batch_norm,
            l2_reg=self.config.model.l2_reg
        ).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.model.l2_reg
        )
        
        # Loss function
        loss_map = {
            "Vanilla": "vanilla",
            "FNW": "fnw",
            "FNC": "fnc",
            "DFM": "dfm",
            "DEFER": "defer",
            "ES-DFM": "esdfm",
            "DEFUSE": "defuse",
            "Bi-DEFUSE": "bidefuse",
        }
        loss_fn = get_loss_fn(loss_map.get(method, "vanilla"))
        
        # Moving averages for metrics
        auc_ma = ScalarMovingAverage()
        prauc_ma = ScalarMovingAverage()
        logloss_ma = ScalarMovingAverage()
        
        # Streaming training
        ob_win = self.config.data.observation_window
        attr_win = self.config.data.attribution_window
        start_hour = self.config.data.train_start_day * 24
        end_hour = self.config.data.train_end_day * 24
        test_start_hour = self.config.data.test_start_day * 24
        
        for hour, (train_data, test_data) in enumerate(
            get_streaming_data(data, method, ob_win, attr_win, start_hour, end_hour)
        ):
            if len(train_data) == 0:
                continue
            
            # Create dataloaders
            train_dataset = CriteoDataset(train_data, self.cat_hash_sizes)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.train.batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_fn
            )
            
            # Train
            loss = self.train_epoch(model, train_loader, optimizer, loss_fn, aux_models)
            
            # Evaluate on test data (only after test_start_hour)
            if hour >= test_start_hour - start_hour:
                test_dataset = CriteoDataset(test_data, self.cat_hash_sizes)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.config.train.batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate_fn
                )
                
                metrics = self.evaluate(model, test_loader, method)
                batch_size = len(test_data)
                
                auc_ma.add(metrics["auc"] * batch_size, batch_size)
                prauc_ma.add(metrics["pr_auc"] * batch_size, batch_size)
                logloss_ma.add(metrics["logloss"] * batch_size, batch_size)
                
                if hour % 24 == 0:  # Print daily
                    print(f"Hour {hour}, Loss: {loss:.4f}, "
                          f"AUC: {metrics['auc']:.4f} (MA: {auc_ma.get():.4f}), "
                          f"PR-AUC: {metrics['pr_auc']:.4f} (MA: {prauc_ma.get():.4f})")
        
        final_metrics = {
            "auc": auc_ma.get(),
            "pr_auc": prauc_ma.get(),
            "logloss": logloss_ma.get()
        }
        
        print(f"\n{method} Final Results:")
        print(f"  AUC: {final_metrics['auc']:.4f}")
        print(f"  PR-AUC: {final_metrics['pr_auc']:.4f}")
        print(f"  LogLoss: {final_metrics['logloss']:.4f}")
        
        return final_metrics
    
    def run_all_methods(self, data: DataDF) -> Dict[str, Dict[str, float]]:
        """
        Run all methods and collect results.
        
        Args:
            data: Full dataset
        
        Returns:
            Results dictionary
        """
        results = {}
        
        # 1. Pretrain auxiliary models
        print("\n" + "="*60)
        print("Phase 1: Pretraining Auxiliary Models")
        print("="*60)
        
        # Pretrain tn/dp model (for ES-DFM, DEFUSE)
        pretrain_data = data.construct_tn_dp_data(
            self.config.data.observation_window,
            self.config.data.attribution_window
        )
        esdfm_model = self.pretrain(pretrain_data, "MLP_tn_dp", "pretrain_tn_dp")
        
        # Pretrain dp model (for DEFER)
        defer_model = self.pretrain(pretrain_data, "MLP_dp", "pretrain_dp")
        
        # 2. Run all methods
        print("\n" + "="*60)
        print("Phase 2: Streaming Training and Evaluation")
        print("="*60)
        
        methods = self.config.methods
        
        for method in methods:
            print(f"\n{'='*40}")
            print(f"Running {method}")
            print(f"{'='*40}")
            
            aux_models = None
            if method in ["ES-DFM", "DEFUSE"]:
                aux_models = {"esdfm": esdfm_model}
            elif method == "DEFER":
                aux_models = {"defer": defer_model}
            elif method == "Bi-DEFUSE":
                aux_models = {"esdfm": esdfm_model}
            
            try:
                results[method] = self.stream_train(data, method, aux_models)
            except Exception as e:
                print(f"Error running {method}: {e}")
                results[method] = {"auc": 0, "pr_auc": 0, "logloss": 0}
        
        # 3. Print summary
        print("\n" + "="*60)
        print("Final Results Summary")
        print("="*60)
        print(f"{'Method':<12} | {'AUC':>8} | {'PR-AUC':>8} | {'LogLoss':>8}")
        print("-" * 45)
        for method, metrics in results.items():
            print(f"{method:<12} | {metrics['auc']:>8.4f} | {metrics['pr_auc']:>8.4f} | {metrics['logloss']:>8.4f}")
        
        # Save results
        results_path = os.path.join(self.config.data.output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
        
        return results
