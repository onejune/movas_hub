#!/usr/bin/env python3
"""
DEFUSE Criteo Benchmark - Main Entry

Run all delayed feedback methods on Criteo Conversion Logs dataset.

Usage:
    python run_criteo.py [--methods METHOD1,METHOD2,...] [--device cuda/cpu]
"""
import argparse
import os
import sys
import time

import torch

from config import Config
from data import load_criteo_data, DataDF
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="DEFUSE Criteo Benchmark")
    
    parser.add_argument("--data_path", type=str,
                        default="/mnt/workspace/walter.wan/open_research/criteo_dataset/data.txt",
                        help="Path to Criteo data.txt")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/workspace/walter.wan/open_research/DEFUSE/outputs_criteo",
                        help="Output directory")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated list of methods to run (default: all)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create config
    config = Config()
    config.data.data_path = args.data_path
    config.data.output_dir = args.output_dir
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.lr
    
    if args.methods:
        config.methods = args.methods.split(",")
    
    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    print("="*60)
    print("DEFUSE Criteo Benchmark")
    print("="*60)
    print(f"Data path: {config.data.data_path}")
    print(f"Output dir: {config.data.output_dir}")
    print(f"Device: {device}")
    print(f"Methods: {config.methods}")
    print(f"Batch size: {config.train.batch_size}")
    print(f"Learning rate: {config.train.learning_rate}")
    print("="*60)
    
    # Load data
    print("\nLoading Criteo dataset...")
    start_time = time.time()
    features, click_ts, pay_ts = load_criteo_data(config.data.data_path)
    data = DataDF.from_raw(features, click_ts, pay_ts, config.data.attribution_window)
    print(f"Loaded {len(data)} samples in {time.time() - start_time:.1f}s")
    
    # Print data statistics
    pos_count = (data.labels > 0).sum()
    print(f"Positive samples: {pos_count} ({100*pos_count/len(data):.2f}%)")
    
    # Create trainer and run
    trainer = Trainer(config, device)
    results = trainer.run_all_methods(data)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
