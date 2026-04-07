#!/usr/bin/env python3
"""只运行评估，生成预测结果"""
import sys
sys.path.insert(0, './src')
from base_trainFlow import BaseTrainFlow

flow = BaseTrainFlow('./conf/widedeep.yaml')
flow._run_evaluation_manual(model_date="2026-03-02", sample_date="2026-03-03")
