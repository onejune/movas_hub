#!/usr/bin/env python3
"""
DEFUSE Loss Functions
"""

import torch
import torch.nn.functional as F


def stable_log1pex(x):
    """Numerically stable log(1 + exp(x))"""
    return torch.clamp(x, min=0) + torch.log1p(torch.exp(-torch.abs(x)))


def pretrain_loss(outputs, targets):
    """
    Pretraining loss for tn/dp classifiers
    Same as ES-DFM pretraining
    """
    tn_logits = outputs['tn_logits'].squeeze(-1)
    dp_logits = outputs['dp_logits'].squeeze(-1)
    
    tn_label = targets['tn_label'].float()
    dp_label = targets['dp_label'].float()
    pos_label = targets['pos_label'].float()
    
    # tn_mask: non-positive + delayed positive samples
    tn_mask = (1 - pos_label) + dp_label
    
    # tn loss (masked)
    tn_loss_raw = F.binary_cross_entropy_with_logits(tn_logits, tn_label, reduction='none')
    tn_loss = (tn_loss_raw * tn_mask).sum() / (tn_mask.sum() + 1e-8)
    
    # dp loss
    dp_loss = F.binary_cross_entropy_with_logits(dp_logits, dp_label, reduction='mean')
    
    return tn_loss + dp_loss


def defuse_loss(outputs, targets):
    """
    DEFUSE loss function
    Asymptotically unbiased estimation via label correction
    
    Samples are divided into 4 categories:
    - IP (In-window Positive): z=1, converted within window
    - DP (Delayed Positive): z=1, converted after window  
    - RN (Real Negative): z=0, truly negative
    - FN (Fake Negative): z=0, will convert but not observed yet
    """
    cv_logits = outputs['cv_logits'].squeeze(-1)
    tn_logits = outputs['tn_logits'].squeeze(-1)
    dp_logits = outputs['dp_logits'].squeeze(-1)
    
    z = targets['label'].float()  # observed label
    
    # Stop gradient for auxiliary predictions
    tn_prob = torch.sigmoid(tn_logits).detach()
    dp_prob = torch.sigmoid(dp_logits).detach()
    
    # zi: probability of being fake negative (1 - tn_prob)
    zi = 1 - tn_prob
    
    # Loss components (from paper)
    loss1 = stable_log1pex(cv_logits)  # IP: -log(sigmoid(x))
    loss2 = stable_log1pex(cv_logits)  # FN: -log(sigmoid(x))
    loss3 = cv_logits + stable_log1pex(cv_logits)  # RN: -log(1-sigmoid(x))
    loss4 = stable_log1pex(cv_logits)  # DP: -log(sigmoid(x))
    
    # Weights (from paper)
    loss1_weight = (1 + dp_prob).detach()
    loss2_weight = dp_prob.detach()
    loss3_weight = (1 + dp_prob).detach()
    loss4_weight = torch.ones_like(dp_prob)
    
    # Weighted losses
    loss1 = loss1 * loss1_weight
    loss2 = zi * loss2 * loss2_weight
    loss3 = (1 - zi) * loss3 * loss3_weight
    loss4 = loss4 * loss4_weight
    
    # Final loss: z * (IP + DP) + (1-z) * (FN + RN)
    loss = torch.mean(z * (loss1 + loss4) + (1 - z) * (loss2 + loss3))
    
    return loss


def bidefuse_loss(outputs, targets):
    """
    Bi-DEFUSE loss function
    Dual-head model: in-window + out-window predictions
    """
    inw_logits = outputs['logits_inw'].squeeze(-1)
    outw_logits = outputs['logits_outw'].squeeze(-1)
    
    cvr_label = targets['label'].float()
    outw_label = targets.get('outw_label', targets.get('dp_label', torch.zeros_like(cvr_label))).float()
    inw_mask = targets.get('inw_mask', torch.ones_like(cvr_label)).float()
    
    # In-window loss (masked)
    inw_pos = stable_log1pex(inw_logits)
    inw_neg = inw_logits + stable_log1pex(inw_logits)
    inw_loss = ((cvr_label * inw_pos + (1 - cvr_label) * inw_neg) * inw_mask).sum() / (inw_mask.sum() + 1e-8)
    
    # Out-window loss with DEFUSE correction
    p_no_grad = torch.sigmoid(outw_logits).detach()
    wi = p_no_grad
    
    loss1 = stable_log1pex(outw_logits)
    loss2 = outw_logits + stable_log1pex(outw_logits)
    loss3 = stable_log1pex(outw_logits)
    
    loss1_weight = torch.ones_like(p_no_grad)
    loss2_weight = 1 + p_no_grad
    loss3_weight = p_no_grad
    
    loss1 = loss1_weight * loss1
    loss2 = loss2_weight * loss2 * (1 - wi)
    loss3 = loss3_weight * loss3 * wi
    
    outw_loss = torch.mean(outw_label * loss1 + (1 - outw_label) * (loss2 + loss3))
    
    return inw_loss + outw_loss


def cross_entropy_loss(outputs, targets):
    """Standard cross entropy loss (vanilla baseline)"""
    cv_logits = outputs['cv_logits'].squeeze(-1)
    label = targets['label'].float()
    return F.binary_cross_entropy_with_logits(cv_logits, label)


def get_loss_fn(name):
    """Get loss function by name"""
    loss_fns = {
        'pretrain': pretrain_loss,
        'defuse': defuse_loss,
        'bidefuse': bidefuse_loss,
        'cross_entropy': cross_entropy_loss,
    }
    if name not in loss_fns:
        raise ValueError(f"Unknown loss function: {name}. Available: {list(loss_fns.keys())}")
    return loss_fns[name]
