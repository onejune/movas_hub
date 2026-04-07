#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metaspore.trainflows - 训练流程模块

提供 MetaSpore 模型训练的完整流程封装。

使用方式:
    from metaspore.trainflows import DNNTrainFlow
    
    # 方式一: 命令行入口
    DNNTrainFlow.main()
    
    # 方式二: 编程调用
    flow = DNNTrainFlow("./conf/widedeep.yaml")
    flow.run(name="exp_v1", eval_keys="business_type")
"""

from .base import BaseTrainFlow
from .dnn import DNNTrainFlow
from .mtl import MTLTrainFlow

__all__ = [
    "BaseTrainFlow",
    "DNNTrainFlow", 
    "MTLTrainFlow",
]
