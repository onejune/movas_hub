#!/usr/bin/env python

import sys
import os

def ensure_metaspore_path():
    """确保 MetaSpore 路径在 PYTHONPATH 中"""
    metaspore_dir = "/mnt/workspace/walter.wan/git_project/github_onejune/movas_hub/DeepForgeX/MetaSpore/python"
    if metaspore_dir not in sys.path:
        sys.path.insert(0, metaspore_dir)

ensure_metaspore_path()

def main():
    from metaspore.trainflows import DNNTrainFlow
    
    args = DNNTrainFlow.parse_args()
    trainer = DNNTrainFlow(config_path=args.conf)
    trainer.run_complete_flow(args)

if __name__ == "__main__":
    main()
