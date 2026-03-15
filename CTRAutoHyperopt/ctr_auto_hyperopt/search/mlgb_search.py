"""
MLGB-specific Model Search with Extended Model Support (33+ models)
"""

import sys
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, log_loss

import optuna
from optuna.trial import Trial

# MLGB path
MLGB_PATH = '/mnt/workspace/walter.wan/open_research/mlgb'
MLGB_AVAILABLE = False
mlgb_ranking = None

try:
    if os.path.exists(MLGB_PATH):
        sys.path.insert(0, MLGB_PATH)
        from mlgb.torch.models import ranking as mlgb_ranking
        MLGB_AVAILABLE = True
except ImportError:
    pass


class MLGBSearch:
    """
    MLGB Model Search - 33+ CTR ranking models.
    
    Normal Models (26):
        LR, MLP, PLM, DLRM, MaskNet,
        DCM, DCN, EDCN,
        FM, FFM, HOFM, FwFM, FEFM, FmFM, AFM, LFM, IM, IFM, DIFM,
        FNN, PNN, PIN, ONN, AFN,
        NFM, WDL, DeepFM, DeepFEFM, DeepIM, FLEN,
        CCPM, FGCNN, XDeepFM, FiBiNet, AutoInt
    
    Sequential Models (8):
        GRU4Rec, Caser, SASRec, BERT4Rec,
        BST, DIN, DIEN, DSIN
    """
    
    # Model categories
    SIMPLE_MODELS = ['LR', 'FM', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FmFM', 'LFM', 'IM']
    DNN_MODELS = ['MLP', 'PLM', 'DLRM', 'MaskNet', 'FNN', 'PNN', 'PIN', 'ONN', 'AFN', 
                  'NFM', 'WDL', 'DeepFM', 'DeepFEFM', 'DeepIM', 'FLEN']
    CROSS_MODELS = ['DCM', 'DCN', 'EDCN']
    CIN_MODELS = ['XDeepFM', 'CCPM', 'FGCNN']
    ATTENTION_MODELS = ['AFM', 'IFM', 'DIFM', 'FiBiNet', 'AutoInt']
    SEQUENTIAL_MODELS = ['GRU4Rec', 'Caser', 'SASRec', 'BERT4Rec', 'BST', 'DIN', 'DIEN', 'DSIN']
    
    ALL_MODELS = SIMPLE_MODELS + DNN_MODELS + CROSS_MODELS + CIN_MODELS + ATTENTION_MODELS
    
    def __init__(
        self,
        models: List[str] = None,
        metric: str = 'auc',
        time_budget: int = 600,
        n_trials: int = 50,
        n_epochs: int = 3,
        batch_size: int = 1024,
        device: str = 'cpu',
        seed: int = 42,
        verbose: int = 1,
    ):
        if not MLGB_AVAILABLE:
            raise ImportError(f"MLGB not available. Check path: {MLGB_PATH}")
        
        self.models = models or self.ALL_MODELS
        self.metric = metric
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.verbose = verbose
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def search(self, data: dict) -> Tuple[List[dict], optuna.Study]:
        """
        Search MLGB models with Optuna.
        
        Args:
            data: Dict with X_train, y_train, X_val, y_val, sparse_features, vocab_sizes
        
        Returns:
            (results_list, optuna_study)
        """
        feature_names = self._build_feature_names(data)
        
        # Prepare tensors
        X_train = torch.tensor(data['X_train'].values, dtype=torch.long)
        y_train = torch.tensor(data['y_train'], dtype=torch.float32)
        X_val = torch.tensor(data['X_val'].values, dtype=torch.long)
        y_val_np = data['y_val']
        
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
        best_results = {}
        
        def objective(trial: Trial) -> float:
            model_name = trial.suggest_categorical('model', self.models)
            
            try:
                model = self._create_model(model_name, feature_names, trial)
                model = model.to(self.device)
                
                lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = torch.nn.BCELoss()
                
                # Training
                model.train()
                for epoch in range(self.n_epochs):
                    for xb, yb in train_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        optimizer.zero_grad()
                        pred = model(xb).squeeze()
                        loss = criterion(pred, yb)
                        loss.backward()
                        optimizer.step()
                
                # Evaluation
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_val.to(self.device)).squeeze().cpu().numpy()
                
                score = roc_auc_score(y_val_np, y_pred)
                
                # Track best per model
                if model_name not in best_results or score > best_results[model_name]['score']:
                    best_results[model_name] = {
                        'score': score,
                        'params': trial.params.copy()
                    }
                
                return score
                
            except Exception as e:
                if self.verbose >= 2:
                    print(f"  Trial failed ({model_name}): {e}")
                return 0.5
        
        # Run optimization
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.time_budget,
            show_progress_bar=(self.verbose >= 1),
            catch=(Exception,)
        )
        
        # Collect results
        results = []
        for model_name, info in sorted(best_results.items(), key=lambda x: -x[1]['score']):
            results.append({
                'model_name': model_name,
                'score': info['score'],
                'params': info['params'],
                'model': None,
            })
            if self.verbose >= 1:
                print(f"  {model_name}: AUC={info['score']:.6f}")
        
        return results, study
    
    def _build_feature_names(self, data: dict) -> tuple:
        """Build MLGB feature_names tuple: (sparse_features, dense_features, sequence_features)"""
        sparse_features = []
        for col in data['sparse_features']:
            vocab_size = data['vocab_sizes'].get(col, int(data['X_train'][col].max()) + 1)
            sparse_features.append({
                'name': col,
                'vocabulary_size': vocab_size,
            })
        
        # Dense features (if any)
        dense_features = []
        for col in data.get('dense_features', []):
            dense_features.append({'name': col})
        
        return (tuple(sparse_features), tuple(dense_features), ())
    
    def _create_model(self, model_name: str, feature_names: tuple, trial: Trial):
        """Create MLGB model with trial hyperparameters."""
        model_class = getattr(mlgb_ranking, model_name, None)
        if model_class is None:
            raise ValueError(f"Unknown MLGB model: {model_name}")
        
        # Base params
        embed_dim = trial.suggest_categorical('embed_dim', [8, 16, 32, 64])
        
        kwargs = {
            'feature_names': feature_names,
            'task': 'binary',
            'device': self.device,
            'embed_dim': embed_dim,
        }
        
        # DNN params for models that support it
        if model_name not in self.SIMPLE_MODELS:
            n_layers = trial.suggest_int('dnn_n_layers', 1, 4)
            dnn_hidden_units = tuple([
                trial.suggest_categorical(f'dnn_hidden_{i}', [64, 128, 256, 512])
                for i in range(n_layers)
            ])
            kwargs['dnn_hidden_units'] = dnn_hidden_units
            kwargs['dnn_dropout'] = trial.suggest_float('dnn_dropout', 0.0, 0.5)
            kwargs['dnn_activation'] = trial.suggest_categorical('dnn_activation', ['relu', 'tanh', 'gelu'])
        
        # Model-specific params
        if model_name in ['DCN', 'EDCN']:
            kwargs['dcn_version'] = trial.suggest_categorical('dcn_version', ['v1', 'v2'])
        
        if model_name == 'EDCN':
            kwargs['edcn_layer_num'] = trial.suggest_int('edcn_layer_num', 1, 4)
            kwargs['bdg_mode'] = trial.suggest_categorical('bdg_mode', [
                'EDCN:pointwise_addition', 'EDCN:hadamard_product', 
                'EDCN:concatenation', 'EDCN:attention_pooling'
            ])
        
        if model_name == 'XDeepFM':
            cin_size = trial.suggest_categorical('cin_layer_size', [64, 128, 256])
            cin_layers = trial.suggest_int('cin_layers', 1, 3)
            kwargs['cin_layer_size'] = tuple([cin_size] * cin_layers)
        
        if model_name in ['AutoInt']:
            kwargs['autoint_layer_num'] = trial.suggest_int('autoint_layer_num', 1, 4)
            kwargs['autoint_head_num'] = trial.suggest_categorical('autoint_head_num', [1, 2, 4, 8])
        
        if model_name == 'AFM':
            kwargs['afm_attention_factor'] = trial.suggest_categorical('afm_attention_factor', [4, 8, 16, 32])
        
        if model_name == 'FiBiNet':
            kwargs['fbi_mode'] = trial.suggest_categorical('fbi_mode', [
                'FiBiNet:all', 'FiBiNet:each', 'FiBiNet:interaction'
            ])
        
        if model_name == 'MaskNet':
            kwargs['masknet_mask_mode'] = trial.suggest_categorical('masknet_mask_mode', [
                'MaskNet:serial', 'MaskNet:parallel'
            ])
        
        if model_name in ['PNN', 'PIN']:
            kwargs['pnn_product_mode'] = trial.suggest_categorical('pnn_product_mode', [
                'PNN:inner', 'PNN:outer', 'PNN:both'
            ])
        
        if model_name == 'FGCNN':
            kwargs['fgcnn_filter_num'] = trial.suggest_categorical('fgcnn_filter_num', [4, 8, 14])
            kwargs['fgcnn_pooling_size'] = trial.suggest_categorical('fgcnn_pooling_size', [2, 3])
        
        return model_class(**kwargs)


def quick_benchmark(data: dict, models: List[str] = None, time_budget: int = 300) -> dict:
    """
    Quick benchmark of MLGB models.
    
    Example:
        results = quick_benchmark(data, models=['MLP', 'DeepFM', 'DCN', 'XDeepFM'])
    """
    searcher = MLGBSearch(
        models=models or ['MLP', 'DeepFM', 'DCN', 'XDeepFM', 'AutoInt'],
        time_budget=time_budget,
        n_trials=30,
        verbose=1,
    )
    results, study = searcher.search(data)
    
    print("\n" + "="*50)
    print("Benchmark Results")
    print("="*50)
    for r in results:
        print(f"{r['model_name']:15s} AUC: {r['score']:.6f}")
    
    return {'results': results, 'study': study}
