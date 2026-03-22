#!/usr/bin/env python3
"""
DEFUSE Criteo Benchmark - Loss Functions

Implements all loss functions for delayed feedback methods.
Aligned with TensorFlow implementation in src_tf/loss.py.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Optional


def stable_log1pex(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable log(1 + exp(-x)) = softplus(-x) = -log(sigmoid(x))
    """
    return F.softplus(-x)


# =============================================================================
# Baseline Losses
# =============================================================================

def vanilla_loss(outputs: Dict[str, torch.Tensor], 
                 targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Vanilla cross-entropy loss.
    
    Args:
        outputs: {"logits": (batch, 1)}
        targets: {"label": (batch,)}
    """
    logits = outputs["logits"].squeeze(-1)
    labels = targets["label"].float()
    return F.binary_cross_entropy_with_logits(logits, labels)


def fnw_loss(outputs: Dict[str, torch.Tensor],
             targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Fake Negative Weighted (FNW) loss.
    
    Reweights samples based on predicted conversion probability.
    
    Formula:
        pos_loss = (1 + p) * -log(σ(x))
        neg_loss = -(1 - p) * (1 + p) * (-x - log(1 + exp(-x)))
    
    where p = σ(x) with stop gradient.
    """
    x = outputs["logits"].squeeze(-1)
    z = targets["label"].float()
    
    p_no_grad = torch.sigmoid(x.detach())
    
    pos_loss = (1 + p_no_grad) * stable_log1pex(x)
    neg_loss = -(1 - p_no_grad) * (1 + p_no_grad) * (-x - stable_log1pex(x))
    
    loss = pos_loss * z + neg_loss * (1 - z)
    return loss.mean()


def fnc_loss(outputs: Dict[str, torch.Tensor],
             targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Fake Negative Calibration (FNC) loss.
    
    Same as vanilla BCE, but prediction is calibrated during inference:
        calibrated_prob = prob / (1 - prob + eps)
    """
    return vanilla_loss(outputs, targets)


def fnc_calibrate(prob: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Calibrate FNC predictions."""
    return prob / (1 - prob + eps)


# =============================================================================
# DFM Loss (Exponential Delay)
# =============================================================================

def dfm_loss(outputs: Dict[str, torch.Tensor],
             targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Delayed Feedback Model (DFM) loss with exponential delay.
    
    Models both conversion probability and delay distribution.
    
    Args:
        outputs: {"logits": (batch, 1), "log_lamb": (batch, 1)}
        targets: {"label": (batch,), "delay": (batch,), "elapsed": (batch,)}
            - label: 1 if converted, 0 otherwise
            - delay: time from click to conversion (for positives)
            - elapsed: observation time (for negatives)
    """
    x = outputs["logits"].squeeze(-1)
    log_lamb_raw = outputs["log_lamb"].squeeze(-1)
    
    z = targets["label"].float()
    d = targets.get("delay", targets["label"]).float()  # delay for positives
    e = targets.get("elapsed", d).float()  # elapsed time
    
    # λ = softplus(log_lamb) to ensure positivity
    lamb = F.softplus(log_lamb_raw)
    log_lamb = torch.log(lamb + 1e-8)
    
    p = torch.sigmoid(x)
    
    # Positive loss: -log(p * λ * exp(-λd)) = -log(p) - log(λ) + λd
    pos_loss = -(-stable_log1pex(x) + log_lamb - lamb * d)
    
    # Negative loss: -log(1 - p + p * exp(-λe))
    neg_loss = -torch.log(1 - p + p * torch.exp(-lamb * e) + 1e-8)
    
    loss = pos_loss * z + neg_loss * (1 - z)
    return loss.mean()


# =============================================================================
# Pretrain Losses
# =============================================================================

def pretrain_tn_dp_loss(outputs: Dict[str, torch.Tensor],
                        targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Pretraining loss for tn/dp classifiers (ES-DFM, DEFUSE).
    
    Args:
        outputs: {"tn_logits": (batch, 1), "dp_logits": (batch, 1)}
        targets: {"tn_label": (batch,), "dp_label": (batch,), "pos_label": (batch,)}
    """
    tn_logits = outputs["tn_logits"].squeeze(-1)
    dp_logits = outputs["dp_logits"].squeeze(-1)
    
    tn_label = targets["tn_label"].float()
    dp_label = targets["dp_label"].float()
    pos_label = targets.get("pos_label", targets["dp_label"]).float()
    
    # tn_mask: non-positive + delayed positive samples
    tn_mask = (1 - pos_label) + dp_label
    
    # Masked tn loss
    tn_loss_raw = F.binary_cross_entropy_with_logits(tn_logits, tn_label, reduction='none')
    tn_loss = (tn_loss_raw * tn_mask).sum() / (tn_mask.sum() + 1e-8)
    
    # dp loss
    dp_loss = F.binary_cross_entropy_with_logits(dp_logits, dp_label)
    
    return tn_loss + dp_loss


def pretrain_dp_loss(outputs: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Pretraining loss for dp classifier only (DEFER).
    
    Args:
        outputs: {"logits": (batch, 1)}
        targets: {"dp_label": (batch,)}
    """
    logits = outputs["logits"].squeeze(-1)
    dp_label = targets["dp_label"].float()
    return F.binary_cross_entropy_with_logits(logits, dp_label)


# =============================================================================
# DEFER Loss
# =============================================================================

def defer_loss(outputs: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    DEFER loss with importance weighting.
    
    Uses pretrained dp model to estimate P(delay > C | y=1, x).
    
    Formula:
        pos_weight = 2 * (2 - dp_prob)
        neg_weight = (1 - p) / (1 - p + 0.5 * dp_prob)
    """
    x = outputs["logits"].squeeze(-1)
    dp_logits = outputs["dp_logits"].squeeze(-1)
    z = targets["label"].float()
    
    p = torch.sigmoid(x)
    dp_prob = torch.sigmoid(dp_logits.detach())
    
    # Importance weights
    pos_weight = 2 * (2 - dp_prob)
    neg_weight = (1 - p.detach()) / (1 - p.detach() + 0.5 * dp_prob + 1e-8)
    
    # Loss
    pos_loss = stable_log1pex(x)  # -log(σ(x))
    neg_loss = x + stable_log1pex(x)  # -log(1-σ(x))
    
    loss = pos_loss * pos_weight * z + neg_loss * neg_weight * (1 - z)
    return loss.mean()


def unbiased_defer_loss(outputs: Dict[str, torch.Tensor],
                        targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Unbiased DEFER loss with 4-component formulation.
    
    Similar to DEFUSE but uses DEFER's duplicate samples strategy.
    """
    x = outputs["logits"].squeeze(-1)
    dp_logits = outputs["dp_logits"].squeeze(-1)
    tn_logits = outputs["tn_logits"].squeeze(-1)
    z = targets["label"].float()
    
    dp_prob = torch.sigmoid(dp_logits.detach())
    tn_prob = torch.sigmoid(tn_logits.detach())
    
    # zi = P(fake negative) = 1 - P(true negative)
    zi = 1 - tn_prob
    
    one = torch.ones_like(z)
    
    # Loss weights
    loss1_weight = one + dp_prob  # IP
    loss2_weight = dp_prob  # FN
    loss3_weight = one + dp_prob  # RN
    loss4_weight = one  # DP
    
    # Loss components
    loss1 = stable_log1pex(x)  # -log(σ(x)) for IP
    loss2 = stable_log1pex(x)  # -log(σ(x)) for FN
    loss3 = x + stable_log1pex(x)  # -log(1-σ(x)) for RN
    loss4 = stable_log1pex(x)  # -log(σ(x)) for DP
    
    # Weighted losses
    loss1 = loss1 * loss1_weight
    loss2 = zi * loss2 * loss2_weight
    loss3 = (1 - zi) * loss3 * loss3_weight
    loss4 = loss4 * loss4_weight
    
    loss = z * (loss1 + loss4) + (1 - z) * (loss2 + loss3)
    return loss.mean()


# =============================================================================
# ES-DFM Loss
# =============================================================================

def esdfm_loss(outputs: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor],
               eps: float = 1e-7) -> torch.Tensor:
    """
    ES-DFM loss with importance weighting.
    
    Formula:
        pos_weight = 1 + dp_prob
        neg_weight = (1 + dp_prob) * tn_prob
    """
    x = outputs["logits"].squeeze(-1)
    tn_logits = outputs["tn_logits"].squeeze(-1)
    dp_logits = outputs["dp_logits"].squeeze(-1)
    z = targets["label"].float()
    
    tn_prob = torch.sigmoid(tn_logits.detach()).clamp(eps, 1 - eps)
    dp_prob = torch.sigmoid(dp_logits.detach()).clamp(eps, 1 - eps)
    
    # Importance weights
    pos_weight = 1 + dp_prob
    neg_weight = (1 + dp_prob) * tn_prob
    
    # Loss
    pos_loss = stable_log1pex(x)  # -log(σ(x))
    neg_loss = x + stable_log1pex(x)  # -log(1-σ(x))
    
    loss = pos_loss * pos_weight * z + neg_loss * neg_weight * (1 - z)
    return loss.mean()


# =============================================================================
# DEFUSE Loss
# =============================================================================

def defuse_loss(outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                eps: float = 1e-7) -> torch.Tensor:
    """
    DEFUSE loss with 4-component label correction.
    
    Sample types:
    - IP (In-window Positive): immediate conversion, label=1
    - FN (Fake Negative): delayed conversion appearing as negative, label=0
    - RN (Real Negative): true negative, label=0
    - DP (Delayed Positive): delayed conversion (duplicated sample), label=1
    
    Loss formula:
    - loss1 (IP): -log(σ(x)) * (1 + dp_prob)
    - loss2 (FN): zi * -log(σ(x)) * dp_prob
    - loss3 (RN): (1-zi) * -log(1-σ(x)) * (1 + dp_prob)
    - loss4 (DP): -log(σ(x)) * 1
    
    where zi = 1 - tn_prob = P(fake negative)
    """
    x = outputs["logits"].squeeze(-1)
    tn_logits = outputs["tn_logits"].squeeze(-1)
    dp_logits = outputs["dp_logits"].squeeze(-1)
    z = targets["label"].float()
    
    tn_prob = torch.sigmoid(tn_logits.detach()).clamp(eps, 1 - eps)
    dp_prob = torch.sigmoid(dp_logits.detach()).clamp(eps, 1 - eps)
    
    # zi = P(fake negative) = 1 - P(true negative)
    zi = (1 - tn_prob).clamp(eps, 1 - eps)
    
    one = torch.ones_like(z)
    
    # Loss weights
    loss1_weight = one + dp_prob  # IP
    loss2_weight = dp_prob  # FN
    loss3_weight = one + dp_prob  # RN
    loss4_weight = one  # DP
    
    # Loss components
    neg_log_sigmoid = stable_log1pex(x)  # -log(σ(x))
    neg_log_1_minus_sigmoid = x + stable_log1pex(x)  # -log(1-σ(x))
    
    loss1 = neg_log_sigmoid * loss1_weight  # IP
    loss2 = zi * neg_log_sigmoid * loss2_weight  # FN
    loss3 = (1 - zi) * neg_log_1_minus_sigmoid * loss3_weight  # RN
    loss4 = neg_log_sigmoid * loss4_weight  # DP
    
    loss = z * (loss1 + loss4) + (1 - z) * (loss2 + loss3)
    return loss.mean()


# =============================================================================
# Bi-DEFUSE Loss
# =============================================================================

def bidefuse_loss(outputs: Dict[str, torch.Tensor],
                  targets: Dict[str, torch.Tensor],
                  eps: float = 1e-7) -> torch.Tensor:
    """
    Bi-DEFUSE loss with separate in-window and out-window heads.
    
    Aligned with TF unbiased_bidefuse_loss:
    - inw_head: BCE on cvr_label, masked by inw_mask (original samples only)
    - outw_head: DEFUSE-style reweighted loss on outw_label (delay_label)
    
    Args:
        outputs: {"logits_inw": (batch, 1), "logits_outw": (batch, 1)}
        targets: {"label": (batch,), "inw_label": (batch,), "delay_label": (batch,)}
    """
    inw_logits = outputs["logits_inw"].squeeze(-1)
    outw_logits = outputs["logits_outw"].squeeze(-1)
    
    cvr_label = targets["label"].float()
    # outw_label = delay_label (1 for duplicated DP samples)
    outw_label = targets.get("delay_label", torch.zeros_like(targets["label"])).float()
    # inw_mask = inw_label (1 for original samples, 0 for duplicated)
    inw_mask = targets.get("inw_label", torch.ones_like(targets["label"])).float()
    
    # In-window loss: BCE on cvr_label, masked by inw_mask
    inw_pos = stable_log1pex(inw_logits)
    inw_neg = inw_logits + stable_log1pex(inw_logits)
    inw_loss_raw = cvr_label * inw_pos + (1 - cvr_label) * inw_neg
    inw_loss = (inw_loss_raw * inw_mask).sum() / (inw_mask.sum() + eps)
    
    # Out-window loss: DEFUSE-style reweighting
    one = torch.ones_like(outw_label)
    p_no_grad = torch.sigmoid(outw_logits.detach())
    wi = p_no_grad  # FN probability estimate
    
    loss1 = stable_log1pex(outw_logits)  # DP: -log(sigmoid(x))
    loss2 = outw_logits + stable_log1pex(outw_logits)  # RN: -log(1-sigmoid(x))
    loss3 = stable_log1pex(outw_logits)  # FN: -log(sigmoid(x))
    
    loss1_weight = one
    loss2_weight = one + p_no_grad
    loss3_weight = p_no_grad
    
    loss1 = loss1_weight * loss1
    loss2 = loss2_weight * loss2 * (one - wi)
    loss3 = loss3_weight * loss3 * wi
    
    outw_loss = (outw_label * loss1 + (one - outw_label) * (loss2 + loss3)).mean()
    
    return inw_loss + outw_loss


# =============================================================================
# Loss Registry
# =============================================================================

def get_loss_fn(name: str):
    """Get loss function by name."""
    losses = {
        # Baselines
        "vanilla": vanilla_loss,
        "cross_entropy": vanilla_loss,
        "fnw": fnw_loss,
        "fake_negative_weighted": fnw_loss,
        "fnc": fnc_loss,
        
        # DFM
        "dfm": dfm_loss,
        "exp_delay": dfm_loss,
        
        # Pretrain
        "pretrain_tn_dp": pretrain_tn_dp_loss,
        "pretrain_dp": pretrain_dp_loss,
        
        # DEFER
        "defer": defer_loss,
        "unbiased_defer": unbiased_defer_loss,
        
        # ES-DFM
        "esdfm": esdfm_loss,
        
        # DEFUSE
        "defuse": defuse_loss,
        
        # Bi-DEFUSE
        "bidefuse": bidefuse_loss,
    }
    
    if name.lower() not in losses:
        raise ValueError(f"Unknown loss: {name}. Available: {list(losses.keys())}")
    
    return losses[name.lower()]
