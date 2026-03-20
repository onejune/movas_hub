#!/usr/bin/env python
"""
DEFER v2 训练入口

Usage:
    python ms_defer.py --conf conf/config.yaml --eval_keys business_type
"""
import sys
import os
import argparse

# Python 版本 (metaspore 编译版本)
os.environ['PYSPARK_PYTHON'] = '/root/anaconda3/envs/spore/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/anaconda3/envs/spore/bin/python'

# 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wd_v5'))

from defer_trainFlow import DeferTrainFlow
from movas_logger import MovasLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default='conf/config.yaml')
    parser.add_argument('--eval_keys', default='business_type')
    args = parser.parse_args()
    
    print(f"DEFER v2 Training | Config: {args.conf}")
    
    trainer = DeferTrainFlow(config_path=args.conf)
    try:
        trainer.run_complete_flow(args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
    finally:
        MovasLogger.save_to_local()


if __name__ == "__main__":
    main()
