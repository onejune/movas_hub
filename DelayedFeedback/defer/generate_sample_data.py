#!/usr/bin/env python3
"""
生成模拟的 Criteo 延迟反馈数据集（用于测试代码流程）

数据格式:
- 列 0: click_ts (点击时间戳)
- 列 1: pay_ts (转化时间戳，未转化为空)
- 列 2-9: 8 个数值特征 (归一化到 0-1)
- 列 10-18: 9 个类别特征

模拟 60 天数据，约 10 万样本
"""

import numpy as np
import pandas as pd
import os

# 配置
NUM_SAMPLES = 100000  # 样本数
NUM_DAYS = 60  # 天数
CVR = 0.23  # 转化率
SECONDS_PER_DAY = 86400

# 输出路径
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'criteo_data.txt')

def generate_delay_time(n):
    """
    生成转化延迟时间分布
    参考论文中的分布:
    - ~35% 在 15 分钟内转化
    - ~50% 在 1 小时内转化
    - ~70% 在 1 天内转化
    """
    # 使用混合分布
    delays = []
    for _ in range(n):
        r = np.random.random()
        if r < 0.35:
            # 15 分钟内
            delay = np.random.exponential(300)  # 5 分钟均值
        elif r < 0.50:
            # 15 分钟 - 1 小时
            delay = 900 + np.random.exponential(900)  # 15 分钟 + 15 分钟均值
        elif r < 0.70:
            # 1 小时 - 1 天
            delay = 3600 + np.random.exponential(7200)  # 1 小时 + 2 小时均值
        else:
            # 1 天以上
            delay = 86400 + np.random.exponential(86400)  # 1 天 + 1 天均值
        delays.append(delay)
    return np.array(delays)

def main():
    print("=" * 60)
    print("生成模拟 Criteo 延迟反馈数据集")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 生成点击时间戳 (60 天内均匀分布)
    # 注意：Criteo 数据集使用相对时间戳（从 0 开始），不是绝对时间戳
    start_ts = 0  # 从 0 开始，与原始 Criteo 数据集格式一致
    click_ts = start_ts + np.random.uniform(0, NUM_DAYS * SECONDS_PER_DAY, NUM_SAMPLES)
    click_ts = np.sort(click_ts).astype(int)
    
    # 生成转化标签
    is_converted = np.random.random(NUM_SAMPLES) < CVR
    num_converted = np.sum(is_converted)
    
    # 生成转化时间
    pay_ts = np.full(NUM_SAMPLES, np.nan)
    delays = generate_delay_time(num_converted)
    pay_ts[is_converted] = click_ts[is_converted] + delays
    
    # 生成 8 个数值特征 (归一化到 0-1)
    num_features = np.random.random((NUM_SAMPLES, 8))
    
    # 生成 9 个类别特征
    cat_features = []
    vocab_sizes = [100, 50, 200, 30, 150, 80, 60, 40, 120]  # 每个类别特征的词汇量
    for vocab_size in vocab_sizes:
        cat_features.append(np.random.randint(0, vocab_size, NUM_SAMPLES).astype(str))
    cat_features = np.column_stack(cat_features)
    
    # 组合数据
    data = np.column_stack([
        click_ts,
        pay_ts,
        num_features,
        cat_features
    ])
    
    # 创建 DataFrame
    columns = ['click_ts', 'pay_ts'] + \
              [f'num_{i}' for i in range(8)] + \
              [f'cat_{i}' for i in range(9)]
    df = pd.DataFrame(data, columns=columns)
    
    # 保存为 TSV 格式 (无表头)
    # 注意: pay_ts 为空时应该保持为空字符串，让 pandas 读取时识别为 NaN
    df.to_csv(OUTPUT_FILE, sep='\t', index=False, header=False, na_rep='')
    
    # 打印统计信息
    print(f"\n生成数据统计:")
    print(f"  - 样本数: {NUM_SAMPLES:,}")
    print(f"  - 转化数: {num_converted:,}")
    print(f"  - 转化率: {num_converted/NUM_SAMPLES:.4f}")
    print(f"  - 天数: {NUM_DAYS}")
    print(f"  - 文件路径: {OUTPUT_FILE}")
    print(f"  - 文件大小: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.2f} MB")
    
    # 打印延迟分布
    converted_delays = delays
    print(f"\n转化延迟分布:")
    print(f"  - 15 分钟内: {np.mean(converted_delays < 900):.2%}")
    print(f"  - 30 分钟内: {np.mean(converted_delays < 1800):.2%}")
    print(f"  - 1 小时内: {np.mean(converted_delays < 3600):.2%}")
    print(f"  - 1 天内: {np.mean(converted_delays < 86400):.2%}")
    print(f"  - 均值: {np.mean(converted_delays)/3600:.2f} 小时")
    print(f"  - 中位数: {np.median(converted_delays)/3600:.2f} 小时")
    
    print("\n" + "=" * 60)
    print("数据生成完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()
