"""
Deep Learning Model Search using Optuna + DeepCTR/MLGB
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

# DeepCTR imports (try vendored first, then system)
DEEPCTR_AVAILABLE = False
try:
    # Try vendored version first
    _third_party_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'third_party', 'DeepCTR-Torch')
    if os.path.exists(_third_party_path):
        sys.path.insert(0, _third_party_path)
    from deepctr_torch.inputs import SparseFeat, get_feature_names
    from deepctr_torch.models import (
        DeepFM, DCN, DCNMix, xDeepFM, AutoInt,
        NFM, AFM, PNN, WDL, FiBiNET, AFN
    )
    DEEPCTR_AVAILABLE = True
except ImportError:
    pass

# MLGB imports (try vendored first, then system)
MLGB_AVAILABLE = False
mlgb_ranking = None
try:
    # Try vendored version first
    _mlgb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'third_party', 'mlgb')
    if os.path.exists(_mlgb_path):
        sys.path.insert(0, _mlgb_path)
    from mlgb.torch.models import ranking as mlgb_ranking
    MLGB_AVAILABLE = True
except ImportError:
    pass


class DLSearch:
    """Search Deep Learning models using Optuna."""
    
    # DeepCTR model registry
    DEEPCTR_MODELS = {
        'DeepFM': DeepFM if DEEPCTR_AVAILABLE else None,
        'DCN': DCN if DEEPCTR_AVAILABLE else None,
        'DCNMix': DCNMix if DEEPCTR_AVAILABLE else None,
        'xDeepFM': xDeepFM if DEEPCTR_AVAILABLE else None,
        'AutoInt': AutoInt if DEEPCTR_AVAILABLE else None,
        'NFM': NFM if DEEPCTR_AVAILABLE else None,
        'AFM': AFM if DEEPCTR_AVAILABLE else None,
        'PNN': PNN if DEEPCTR_AVAILABLE else None,  # fix: PNN was missing from registry
        'WDL': WDL if DEEPCTR_AVAILABLE else None,
        'FiBiNET': FiBiNET if DEEPCTR_AVAILABLE else None,
        'AFN': AFN if DEEPCTR_AVAILABLE else None,
    }
    
    # MLGB model list (33 ranking models)
    MLGB_MODELS = [
        'LR', 'MLP', 'PLM', 'DLRM', 'MaskNet',
        'DCM', 'DCN', 'EDCN',
        'FM', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FmFM', 'AFM', 'LFM', 'IM', 'IFM', 'DIFM',
        'FNN', 'PNN', 'PIN', 'ONN', 'AFN',
        'NFM', 'WDL', 'DeepFM', 'DeepFEFM', 'DeepIM', 'FLEN',
        'CCPM', 'FGCNN', 'XDeepFM', 'FiBiNet', 'AutoInt',
    ]
    
    def __init__(
        self,
        models: List[str],
        backend: str = 'deepctr',
        metric: str = 'auc',
        time_budget: int = 300,
        n_trials: int = 30,
        n_epochs: int = 3,
        device: str = 'cpu',
        seed: int = 42,
        verbose: int = 1,
    ):
        self.models = models
        self.backend = backend
        self.metric = metric
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.n_epochs = n_epochs
        self.device = device
        self.seed = seed
        self.verbose = verbose
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def search(self, data: dict) -> Tuple[List[dict], optuna.Study]:
        """Search DL models."""
        if self.backend == 'deepctr':
            return self._search_deepctr(data)
        elif self.backend == 'mlgb':
            return self._search_mlgb(data)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _search_deepctr(self, data: dict) -> Tuple[List[dict], optuna.Study]:
        """Search using DeepCTR models."""
        if not DEEPCTR_AVAILABLE:
            raise ImportError("DeepCTR not available. pip install deepctr-torch")
        
        feature_columns = data.get('feature_columns')
        if feature_columns is None:
            feature_columns = self._build_deepctr_features(data)
        
        def objective(trial: Trial) -> float:
            model_name = trial.suggest_categorical('model', self.models)
            try:
                model = self._create_deepctr_model(model_name, feature_columns, trial)
                lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])
                
                model.compile(
                    optimizer=torch.optim.Adam(model.parameters(), lr=lr),
                    loss='binary_crossentropy',
                    metrics=['binary_crossentropy', 'auc']
                )
                model.fit(
                    data['train_input'], data['y_train'],
                    batch_size=batch_size, epochs=self.n_epochs, verbose=0,
                    validation_data=(data['val_input'], data['y_val'])
                )
                y_pred = model.predict(data['val_input'], batch_size=batch_size)
                score = roc_auc_score(data['y_val'], y_pred)
                # fix: save model to trial user_attrs so _collect_results can retrieve it
                trial.set_user_attr('model_obj', model)
                return score
            except Exception as e:
                if self.verbose >= 2:
                    print(f"  Trial failed ({model_name}): {e}")
                return 0.5
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.time_budget,
                      show_progress_bar=(self.verbose >= 1), catch=(Exception,))
        
        return self._collect_results(study), study
    
    def _search_mlgb(self, data: dict) -> Tuple[List[dict], optuna.Study]:
        """Search using MLGB models."""
        if not MLGB_AVAILABLE:
            raise ImportError("MLGB not available. Check path.")
        
        feature_names = self._build_mlgb_features(data)
        X_train = torch.tensor(data['X_train'].values, dtype=torch.long)
        y_train = torch.tensor(data['y_train'], dtype=torch.float32)
        X_val = torch.tensor(data['X_val'].values, dtype=torch.long)
        y_val = torch.tensor(data['y_val'], dtype=torch.float32)
        
        def objective(trial: Trial) -> float:
            model_name = trial.suggest_categorical('model', self.models)
            try:
                model = self._create_mlgb_model(model_name, feature_names, trial)
                lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])
                
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = torch.nn.BCELoss()
                
                train_ds = TensorDataset(X_train, y_train)
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                
                model.train()
                for epoch in range(self.n_epochs):
                    for xb, yb in train_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        optimizer.zero_grad()
                        pred = model(xb).squeeze()
                        loss = criterion(pred, yb)
                        loss.backward()
                        optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_val.to(self.device)).squeeze().cpu().numpy()
                return roc_auc_score(y_val.numpy(), y_pred)
            except Exception as e:
                if self.verbose >= 2:
                    print(f"  Trial failed ({model_name}): {e}")
                return 0.5
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.time_budget,
                      show_progress_bar=(self.verbose >= 1), catch=(Exception,))
        
        return self._collect_results(study), study
    
    def _build_deepctr_features(self, data: dict) -> List:
        """Build DeepCTR feature columns."""
        feature_columns = []
        for col in data['feature_names']:
            vocab_size = int(data['X_train'][col].max()) + 1
            feature_columns.append(SparseFeat(col, vocabulary_size=vocab_size, embedding_dim=16))
        return feature_columns
    
    def _build_mlgb_features(self, data: dict) -> tuple:
        """Build MLGB feature_names tuple."""
        sparse_features = []
        for col in data['sparse_features']:
            vocab_size = int(data['X_train'][col].max()) + 1
            sparse_features.append({'name': col, 'vocabulary_size': vocab_size})
        return (tuple(sparse_features), (), ())  # (sparse, dense, sequence)
    
    def _create_deepctr_model(self, model_name: str, feature_columns: List, trial: Trial):
        """Create DeepCTR model with trial params."""
        embed_dim = trial.suggest_categorical('embed_dim', [8, 16, 32])
        updated_columns = [
            SparseFeat(f.name, vocabulary_size=f.vocabulary_size, embedding_dim=embed_dim)
            for f in feature_columns
        ]
        
        n_layers = trial.suggest_int('dnn_n_layers', 1, 4)
        dnn_hidden_units = tuple([
            trial.suggest_categorical(f'dnn_hidden_{i}', [64, 128, 256, 512])
            for i in range(n_layers)
        ])
        dnn_dropout = trial.suggest_float('dnn_dropout', 0.0, 0.5)
        
        model_class = self.DEEPCTR_MODELS.get(model_name)
        if model_class is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        kwargs = {
            'linear_feature_columns': updated_columns,
            'dnn_feature_columns': updated_columns,
            'dnn_hidden_units': dnn_hidden_units,
            'dnn_dropout': dnn_dropout,
            'device': self.device,
        }
        
        if model_name in ['DCN', 'DCNMix']:
            kwargs['cross_num'] = trial.suggest_int('cross_num', 1, 4)
        elif model_name == 'xDeepFM':
            kwargs['cin_layer_size'] = trial.suggest_categorical(
                'cin_layer_size', [(64,), (128,), (64, 64), (128, 128)])
        elif model_name == 'AutoInt':
            kwargs['att_layer_num'] = trial.suggest_int('att_layer_num', 1, 4)
            kwargs['att_head_num'] = trial.suggest_categorical('att_head_num', [1, 2, 4])
        elif model_name == 'FiBiNET':
            kwargs['bilinear_type'] = trial.suggest_categorical(
                'bilinear_type', ['all', 'each', 'interaction'])
        elif model_name == 'AFM':
            kwargs.pop('dnn_hidden_units', None)
            kwargs.pop('dnn_dropout', None)
            kwargs['attention_factor'] = trial.suggest_categorical('attention_factor', [4, 8, 16])
        elif model_name == 'PNN':
            # PNN only takes dnn_feature_columns, not linear_feature_columns
            kwargs.pop('linear_feature_columns', None)
            kwargs['use_inner'] = True
            kwargs['use_outter'] = False
        
        return model_class(**kwargs)
    
    def _create_mlgb_model(self, model_name: str, feature_names: tuple, trial: Trial):
        """Create MLGB model with trial params."""
        embed_dim = trial.suggest_categorical('embed_dim', [8, 16, 32])
        n_layers = trial.suggest_int('dnn_n_layers', 1, 4)
        dnn_hidden_units = tuple([
            trial.suggest_categorical(f'dnn_hidden_{i}', [64, 128, 256, 512])
            for i in range(n_layers)
        ])
        dnn_dropout = trial.suggest_float('dnn_dropout', 0.0, 0.5)
        dnn_activation = trial.suggest_categorical('dnn_activation', ['relu', 'tanh', 'gelu'])
        
        model_class = getattr(mlgb_ranking, model_name, None)
        if model_class is None:
            raise ValueError(f"Unknown MLGB model: {model_name}")
        
        kwargs = {
            'feature_names': feature_names,
            'task': 'binary',
            'device': self.device,
            'embed_dim': embed_dim,
        }
        
        # Model-specific params
        if model_name not in ['LR', 'FM', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FmFM', 'LFM', 'IM']:
            kwargs['dnn_hidden_units'] = dnn_hidden_units
            kwargs['dnn_dropout'] = dnn_dropout
            kwargs['dnn_activation'] = dnn_activation
        
        if model_name in ['DCN', 'EDCN']:
            kwargs['dcn_version'] = trial.suggest_categorical('dcn_version', ['v1', 'v2'])
        elif model_name == 'XDeepFM':
            kwargs['cin_layer_size'] = trial.suggest_categorical(
                'cin_layer_size', [(64,), (128,), (64, 64), (128, 128)])
        elif model_name == 'AutoInt':
            kwargs['autoint_layer_num'] = trial.suggest_int('autoint_layer_num', 1, 4)
            kwargs['autoint_head_num'] = trial.suggest_categorical('autoint_head_num', [1, 2, 4])
        elif model_name == 'AFM':
            kwargs['afm_attention_factor'] = trial.suggest_categorical('afm_attention_factor', [4, 8, 16])
        
        return model_class(**kwargs)
    
    def _collect_results(self, study: optuna.Study) -> List[dict]:
        """Collect results from study."""
        model_best = {}
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                model = trial.params.get('model', 'unknown')
                if model not in model_best or trial.value > model_best[model]['score']:
                    model_best[model] = {
                        'score': trial.value,
                        'params': trial.params,
                        # fix: retrieve saved model object (may be None if trial failed mid-way)
                        'model_obj': trial.user_attrs.get('model_obj', None),
                    }
        
        results = []
        for model_name, info in model_best.items():
            results.append({
                'model_name': model_name,
                'score': info['score'],
                'params': info['params'],
                'model': info['model_obj'],  # fix: was always None before
            })
            if self.verbose >= 1:
                print(f"  {model_name}: {self.metric.upper()}={info['score']:.6f}")
        
        return results
