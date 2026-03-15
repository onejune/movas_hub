"""
AutoCTR - Main API for Automated CTR Model Selection & Hyperparameter Optimization
"""

import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, log_loss

import optuna
from optuna.trial import Trial

from .search.ml_search import MLSearch
from .search.dl_search import DLSearch
from .data.preprocessing import DataProcessor

warnings.filterwarnings('ignore')


class AutoCTR:
    """
    Automated Model Selection & Hyperparameter Optimization for CTR Prediction.
    
    Features:
    - Search across 50+ CTR models (ML + DL)
    - Optuna-based Bayesian optimization with pruning
    - Auto feature engineering (crossing, encoding)
    - Unified API for training and prediction
    
    Example:
        >>> auto_ctr = AutoCTR(metric='auc', time_budget=600)
        >>> auto_ctr.fit(X_train, y_train, X_val, y_val)
        >>> y_pred = auto_ctr.predict(X_test)
        >>> print(f"Best Model: {auto_ctr.best_model_name}, AUC: {auto_ctr.best_score:.4f}")
    """
    
    # Available model pools
    ML_MODELS = ['lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree']
    
    DL_MODELS_DEEPCTR = [
        'DeepFM', 'DCN', 'DCNMix', 'xDeepFM', 'AutoInt',
        'NFM', 'AFM', 'PNN', 'WDL', 'FiBiNET', 'AFN'
    ]
    
    DL_MODELS_MLGB = [
        'LR', 'MLP', 'PLM', 'DLRM', 'MaskNet',
        'DCM', 'DCN', 'EDCN',
        'FM', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FmFM', 'AFM', 'LFM', 'IM', 'IFM', 'DIFM',
        'FNN', 'PNN', 'PIN', 'ONN', 'AFN',
        'NFM', 'WDL', 'DeepFM', 'DeepFEFM', 'DeepIM', 'FLEN',
        'CCPM', 'FGCNN', 'XDeepFM', 'FiBiNet', 'AutoInt',
        'GRU4Rec', 'Caser', 'SASRec', 'BERT4Rec',
        'BST', 'DIN', 'DIEN', 'DSIN',
    ]
    
    def __init__(
        self,
        task: str = 'binary',
        metric: str = 'auc',
        time_budget: int = 600,
        n_trials: int = 50,
        include_models: List[str] = None,
        exclude_models: List[str] = None,
        search_ml: bool = True,
        search_dl: bool = True,
        dl_backend: str = 'deepctr',  # 'deepctr' or 'mlgb'
        device: str = 'auto',
        seed: int = 42,
        verbose: int = 1,
    ):
        """
        Initialize AutoCTR.
        
        Args:
            task: Task type, 'binary' or 'regression'
            metric: Optimization metric, 'auc', 'logloss', 'rmse'
            time_budget: Total search time in seconds
            n_trials: Number of Optuna trials
            include_models: List of models to include (None = all)
            exclude_models: List of models to exclude
            search_ml: Whether to search ML models
            search_dl: Whether to search DL models
            dl_backend: Deep learning backend, 'deepctr' or 'mlgb'
            device: Device for DL models, 'cpu', 'cuda', or 'auto'
            seed: Random seed
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.task = task
        self.metric = metric
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.search_ml = search_ml
        self.search_dl = search_dl
        self.dl_backend = dl_backend
        self.seed = seed
        self.verbose = verbose
        
        # Device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Model pool
        self._setup_model_pool(include_models, exclude_models)
        
        # Results
        self._best_model = None
        self._best_model_name = None
        self._best_score = None
        self._best_params = None
        self._study = None
        self._all_results = []
        
        # Data processor
        self.data_processor = DataProcessor()
    
    def _setup_model_pool(self, include_models, exclude_models):
        """Setup model pool based on include/exclude lists."""
        self.ml_models = self.ML_MODELS.copy() if self.search_ml else []
        
        if self.dl_backend == 'deepctr':
            self.dl_models = self.DL_MODELS_DEEPCTR.copy() if self.search_dl else []
        else:
            self.dl_models = self.DL_MODELS_MLGB.copy() if self.search_dl else []
        
        all_models = self.ml_models + self.dl_models
        
        if include_models:
            all_models = [m for m in all_models if m in include_models]
        
        if exclude_models:
            all_models = [m for m in all_models if m not in exclude_models]
        
        self.ml_models = [m for m in self.ml_models if m in all_models]
        self.dl_models = [m for m in self.dl_models if m in all_models]
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame = None,
        y_val: np.ndarray = None,
        sparse_features: List[str] = None,
        dense_features: List[str] = None,
        cross_features: List[Tuple[str, str]] = None,
        high_cardinality_threshold: int = 100,
    ) -> 'AutoCTR':
        """
        Fit AutoCTR: search for best model and hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (if None, use train-val split)
            y_val: Validation labels
            sparse_features: List of categorical feature names
            dense_features: List of numerical feature names
            cross_features: List of feature pairs for crossing
            high_cardinality_threshold: Threshold for high-cardinality handling
        
        Returns:
            self
        """
        start_time = datetime.now()
        
        if self.verbose >= 1:
            print("="*60)
            print("CTR Auto HyperOpt - Model Search")
            print("="*60)
            print(f"Start Time: {start_time}")
            print(f"Device: {self.device}")
            print(f"ML Models: {self.ml_models}")
            print(f"DL Models: {self.dl_models}")
            print(f"Time Budget: {self.time_budget}s")
        
        # Preprocess data
        if self.verbose >= 1:
            print("\n[1/3] Preprocessing data...")
        
        processed_data = self.data_processor.fit_transform(
            X_train, y_train, X_val, y_val,
            sparse_features=sparse_features,
            dense_features=dense_features,
            cross_features=cross_features,
            high_cardinality_threshold=high_cardinality_threshold,
        )
        
        if self.verbose >= 1:
            print(f"  Train: {len(processed_data['y_train']):,}")
            print(f"  Val: {len(processed_data['y_val']):,}")
            print(f"  Features: {len(processed_data['feature_names'])}")
        
        # Split time budget
        ml_budget = int(self.time_budget * 0.3) if self.search_ml and self.search_dl else self.time_budget
        dl_budget = self.time_budget - ml_budget if self.search_ml else self.time_budget
        
        # Search ML models
        if self.ml_models:
            if self.verbose >= 1:
                print(f"\n[2/3] Searching ML models ({ml_budget}s)...")
            
            ml_search = MLSearch(
                models=self.ml_models,
                metric=self.metric,
                time_budget=ml_budget,
                seed=self.seed,
                verbose=self.verbose,
            )
            ml_results = ml_search.search(processed_data)
            self._all_results.extend(ml_results)
        
        # Search DL models
        if self.dl_models:
            if self.verbose >= 1:
                print(f"\n[3/3] Searching DL models ({dl_budget}s)...")
            
            dl_search = DLSearch(
                models=self.dl_models,
                backend=self.dl_backend,
                metric=self.metric,
                time_budget=dl_budget,
                n_trials=self.n_trials,
                device=self.device,
                seed=self.seed,
                verbose=self.verbose,
            )
            dl_results, study = dl_search.search(processed_data)
            self._all_results.extend(dl_results)
            self._study = study
        
        # Find best model
        self._find_best_model()
        
        # Summary
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("Search Complete!")
            print("="*60)
            print(f"Best Model: {self._best_model_name}")
            print(f"Best {self.metric.upper()}: {self._best_score:.6f}")
            print(f"Best Params: {self._best_params}")
            print(f"Total Time: {datetime.now() - start_time}")
        
        return self
    
    def _find_best_model(self):
        """Find best model from all results."""
        if not self._all_results:
            raise ValueError("No search results available")
        
        # Sort by score (higher is better for AUC, lower for loss)
        reverse = self.metric in ['auc']
        sorted_results = sorted(self._all_results, key=lambda x: x['score'], reverse=reverse)
        
        best = sorted_results[0]
        self._best_model = best['model']
        self._best_model_name = best['model_name']
        self._best_score = best['score']
        self._best_params = best['params']
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict labels for X."""
        if self._best_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_processed = self.data_processor.transform(X)
        
        if hasattr(self._best_model, 'predict'):
            return self._best_model.predict(X_processed)
        else:
            raise NotImplementedError("Model does not support predict()")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for X."""
        if self._best_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_processed = self.data_processor.transform(X)
        
        if hasattr(self._best_model, 'predict_proba'):
            proba = self._best_model.predict_proba(X_processed)
            if proba.ndim == 2:
                return proba[:, 1]
            return proba
        elif hasattr(self._best_model, 'predict'):
            return self._best_model.predict(X_processed)
        else:
            raise NotImplementedError("Model does not support prediction")
    
    @property
    def best_model(self):
        """Get best model."""
        return self._best_model
    
    @property
    def best_model_name(self) -> str:
        """Get best model name."""
        return self._best_model_name
    
    @property
    def best_score(self) -> float:
        """Get best score."""
        return self._best_score
    
    @property
    def best_params(self) -> dict:
        """Get best parameters."""
        return self._best_params
    
    @property
    def all_results(self) -> List[dict]:
        """Get all search results."""
        return self._all_results
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get leaderboard of all models."""
        if not self._all_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self._all_results)
        df = df[['model_name', 'score', 'params']].sort_values(
            'score', ascending=(self.metric not in ['auc'])
        )
        return df.reset_index(drop=True)
