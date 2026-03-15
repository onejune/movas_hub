"""
Machine Learning Model Search using FLAML
"""

from typing import Dict, List
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

from flaml import AutoML


class MLSearch:
    """Search ML models (LightGBM, XGBoost, CatBoost, etc.) using FLAML."""
    
    def __init__(
        self,
        models: List[str],
        metric: str = 'auc',
        time_budget: int = 300,
        seed: int = 42,
        verbose: int = 1,
    ):
        self.models = models
        self.metric = metric
        self.time_budget = time_budget
        self.seed = seed
        self.verbose = verbose
    
    def search(self, data: dict) -> List[dict]:
        """
        Search ML models.
        
        Args:
            data: Dict with 'X_train', 'y_train', 'X_val', 'y_val', etc.
        
        Returns:
            List of result dicts with 'model_name', 'score', 'params', 'model'
        """
        results = []
        
        # Per-model time budget
        per_model_budget = max(60, self.time_budget // len(self.models))
        
        for model_name in self.models:
            if self.verbose >= 1:
                print(f"  Training {model_name}...")
            
            try:
                automl = AutoML()
                automl.fit(
                    X_train=data['X_train'],
                    y_train=data['y_train'],
                    task='classification',
                    metric='roc_auc' if self.metric == 'auc' else self.metric,
                    time_budget=per_model_budget,
                    estimator_list=[model_name],
                    seed=self.seed,
                    verbose=0,
                )
                
                # Evaluate
                y_pred = automl.predict_proba(data['X_val'])[:, 1]
                
                if self.metric == 'auc':
                    score = roc_auc_score(data['y_val'], y_pred)
                elif self.metric == 'logloss':
                    score = log_loss(data['y_val'], y_pred)
                else:
                    score = roc_auc_score(data['y_val'], y_pred)
                
                results.append({
                    'model_name': model_name,
                    'score': score,
                    'params': automl.best_config,
                    'model': automl,
                })
                
                if self.verbose >= 1:
                    print(f"    {self.metric.upper()}: {score:.6f}")
            
            except Exception as e:
                if self.verbose >= 1:
                    print(f"    Failed: {e}")
                results.append({
                    'model_name': model_name,
                    'score': 0.5 if self.metric == 'auc' else float('inf'),
                    'params': {},
                    'model': None,
                })
        
        return results
