#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WD_V5 CTR 训练项目 - 使用 MetaSpore 包结构

项目名称: wd_v5
训练流程: DNNTrainFlow
模型类型: WideDeep
"""

import sys
import os

def main():
    """主训练函数"""
    from metaspore.trainflows import DNNTrainFlow
    
    # 解析命令行参数
    args = DNNTrainFlow.parse_args()
    
    # 创建训练器实例
    trainer = DNNTrainFlow(config_path=args.conf)
    
    # 执行完整训练流程
    trainer.run_complete_flow(args)

if __name__ == "__main__":
    main()