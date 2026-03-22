#!/usr/bin/env python3
"""
DEFUSE vs ES-DFM Full Data 对比实验

使用天级增量训练, 避免 OOM
"""

import sys
sys.path.insert(0, '/mnt/workspace/walter.wan/open_research/DEFUSE')

from src_pytorch import (
    Config, IncrementalDataLoader, Trainer,
    list_available_dates
)


def main():
    # 配置
    config = Config()
    
    # 获取可用日期
    raw_data_dir = config.data.raw_data_dir
    available_dates = list_available_dates(raw_data_dir)
    print(f"Available dates: {available_dates[0]} to {available_dates[-1]} ({len(available_dates)} days)")
    
    # 划分训练/测试集
    train_end = config.data.train_end_date
    test_start = config.data.test_start_date
    
    train_dates = [d for d in available_dates if d <= train_end]
    test_dates = [d for d in available_dates if d >= test_start]
    
    print(f"Train dates: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
    print(f"Test dates: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
    
    # 创建数据加载器
    data_loader = IncrementalDataLoader(
        data_dir=raw_data_dir,
        train_dates=train_dates,
        test_dates=test_dates,
        exclude_cols=config.data.exclude_cols,
        days_per_batch=config.train.days_per_batch
    )
    
    # 拟合编码器
    data_loader.fit_encoder()
    
    # 保存编码器
    encoder_path = f"{config.train.output_dir}/encoder.json"
    data_loader.save_encoder(encoder_path)
    print(f"Encoder saved to {encoder_path}")
    
    # 创建训练器
    trainer = Trainer(
        data_loader=data_loader,
        output_dir=config.train.output_dir,
        batch_size=config.train.batch_size,
        learning_rate=config.train.learning_rate,
        num_workers=config.train.num_workers,
        device='cpu'  # 无 GPU
    )
    
    # 对比训练
    results = trainer.compare_methods(
        methods=config.train.methods,
        pretrain_epochs=config.train.pretrain_epochs,
        finetune_epochs=config.train.finetune_epochs
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for method, metrics in results.items():
        print(f"{method.upper()}: AUC={metrics['auc']:.4f}, "
              f"PR-AUC={metrics['pr_auc']:.4f}, LogLoss={metrics['logloss']:.4f}")


if __name__ == '__main__':
    main()
