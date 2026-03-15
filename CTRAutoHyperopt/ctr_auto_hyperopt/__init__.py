"""
CTR Auto HyperOpt
=================

Automated Model Selection & Hyperparameter Optimization for CTR Prediction.

Features:
- 50+ CTR models (DeepFM, xDeepFM, DCN, AutoInt, etc.)
- ML models (XGBoost, LightGBM, CatBoost)
- Optuna-based Bayesian optimization
- Auto feature engineering

Usage:
    from ctr_auto_hyperopt import AutoCTR
    
    auto_ctr = AutoCTR(metric='auc', time_budget=600)
    auto_ctr.fit(X_train, y_train, X_val, y_val)
    y_pred = auto_ctr.predict(X_test)
"""

from .auto_ctr import AutoCTR
from .search import MLSearch, DLSearch
from .data import DataProcessor

__version__ = '0.1.0'
__author__ = 'Walter Wan'
__all__ = ['AutoCTR', 'MLSearch', 'DLSearch', 'DataProcessor']
