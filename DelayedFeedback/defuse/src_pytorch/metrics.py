#!/usr/bin/env python3
"""
DEFUSE Evaluation Metrics
"""

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score


def compute_metrics(labels, preds):
    """
    Compute evaluation metrics
    
    Args:
        labels: Ground truth labels (numpy array)
        preds: Predicted probabilities (numpy array)
        
    Returns:
        Dict with auc, prauc, logloss
    """
    # Clip predictions for numerical stability
    preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)
    
    auc = roc_auc_score(labels, preds)
    prauc = average_precision_score(labels, preds)
    logloss = log_loss(labels, preds_clipped)
    
    return {
        'auc': auc,
        'prauc': prauc,
        'logloss': logloss,
    }


def print_metrics(metrics, prefix=""):
    """Print metrics in a formatted way"""
    if prefix:
        prefix = f"{prefix} "
    print(f"{prefix}AUC={metrics['auc']:.4f}, PR-AUC={metrics['prauc']:.4f}, LogLoss={metrics['logloss']:.4f}")
