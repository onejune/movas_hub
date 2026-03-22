#!/usr/bin/env python3
"""
DEFUSE Criteo Benchmark - Metrics

Evaluation metrics for delayed feedback methods.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from typing import Dict, Optional


def cal_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Calculate AUC score."""
    try:
        return roc_auc_score(labels, probs)
    except ValueError:
        return 0.5


def cal_prauc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Calculate PR-AUC score."""
    try:
        return average_precision_score(labels, probs)
    except ValueError:
        return 0.0


def cal_logloss_with_prob(labels: np.ndarray, probs: np.ndarray, eps: float = 1e-7) -> float:
    """Calculate log loss from probabilities."""
    probs = np.clip(probs, eps, 1 - eps)
    return log_loss(labels, probs)


def cal_logloss_with_logits(labels: np.ndarray, logits: np.ndarray) -> float:
    """Calculate log loss from logits."""
    probs = 1 / (1 + np.exp(-logits))
    return cal_logloss_with_prob(labels, probs)


class ScalarMovingAverage:
    """
    Moving average calculator for streaming evaluation.
    Aligned with TF implementation.
    """
    
    def __init__(self):
        self.sum = 0.0
        self.count = 0
    
    def add(self, value: float, count: int = 1):
        """Add weighted value."""
        self.sum += value
        self.count += count
    
    def get(self) -> float:
        """Get current average."""
        if self.count == 0:
            return 0.0
        return self.sum / self.count
    
    def reset(self):
        """Reset accumulator."""
        self.sum = 0.0
        self.count = 0


def evaluate(labels: np.ndarray, 
             probs: np.ndarray, 
             method: str = "default") -> Dict[str, float]:
    """
    Evaluate predictions.
    
    Args:
        labels: Ground truth labels
        probs: Predicted probabilities
        method: Method name (for FNC calibration)
    
    Returns:
        Dictionary with AUC, PR-AUC, LogLoss
    """
    # FNC calibration
    if method.upper() == "FNC":
        probs = probs / (1 - probs + 1e-8)
        probs = np.clip(probs, 0, 1)
    
    return {
        "auc": cal_auc(labels, probs),
        "pr_auc": cal_prauc(labels, probs),
        "logloss": cal_logloss_with_prob(labels, probs)
    }
