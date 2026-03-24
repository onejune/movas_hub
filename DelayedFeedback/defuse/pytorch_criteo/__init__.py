#!/usr/bin/env python3
"""
DEFUSE Criteo Benchmark

PyTorch implementation of delayed feedback methods:
- Vanilla: Standard BCE loss
- FNW: Fake Negative Weighted
- FNC: Fake Negative Calibration
- DFM: Delayed Feedback Model (exponential delay)
- DEFER: Importance weighting with dp model
- ES-DFM: Importance weighting with tn+dp model
- DEFUSE: 4-class label correction
- Bi-DEFUSE: Dual-head variant with attention
"""

__version__ = "1.0.0"
