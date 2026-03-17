# -*- coding: utf-8 -*-
"""
运行所有 Benchmark
==================
分别运行 ML / DeepCTR / MLGB 三个 benchmark
"""

import os
import sys
import json
from datetime import datetime

sys.path.append('/mnt/workspace/walter.wan/utils')
from feishu_notifier import FeishuNotifier

# 导入各个 benchmark
from benchmark_ml import run as run_ml
from benchmark_deepctr import run as run_deepctr
from benchmark_mlgb import run as run_mlgb


RESULT_DIR = '/tmp/ctr_benchmark'


def main():
    start = datetime.now()
    all_results = []
    
    print("=" * 60)
    print("CTR 模型全量 Benchmark")
    print("=" * 60)
    print(f"开始时间: {start}\n")
    
    # 1. ML 模型
    print("\n" + "=" * 60)
    print("Phase 1: ML 模型 (FLAML)")
    print("=" * 60)
    try:
        ml_result = run_ml()
        all_results.append(ml_result)
    except Exception as e:
        print(f"ML benchmark 失败: {e}")
    
    # 2. DeepCTR 模型
    print("\n" + "=" * 60)
    print("Phase 2: DeepCTR 模型")
    print("=" * 60)
    try:
        deepctr_results = run_deepctr()
        all_results.extend(deepctr_results)
    except Exception as e:
        print(f"DeepCTR benchmark 失败: {e}")
    
    # 3. MLGB 模型
    print("\n" + "=" * 60)
    print("Phase 3: MLGB 模型")
    print("=" * 60)
    try:
        mlgb_results = run_mlgb()
        all_results.extend(mlgb_results)
    except Exception as e:
        print(f"MLGB benchmark 失败: {e}")
    
    # 汇总
    elapsed = (datetime.now() - start).total_seconds()
    
    # 过滤成功的结果并排序
    success_results = [r for r in all_results if r.get('status', 'success') == 'success' or 'auc' in r]
    success_results = sorted(success_results, key=lambda x: -x.get('auc', 0))
    
    print("\n" + "=" * 60)
    print("全量结果 (按 AUC 排序)")
    print("=" * 60)
    print(f"{'模型':<25} {'AUC':>10} {'LogLoss':>10} {'PCOC':>8}")
    print("-" * 60)
    for r in success_results:
        model = r.get('model', 'Unknown')
        auc = r.get('auc', 0)
        logloss = r.get('logloss', 0)
        pcoc = r.get('pcoc', 0)
        print(f"{model:<25} {auc:>10.4f} {logloss:>10.4f} {pcoc:>8.3f}")
    print("-" * 60)
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    
    # 保存汇总结果
    summary_file = os.path.join(RESULT_DIR, f"all_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'results': success_results,
            'total_time_sec': elapsed,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=str)
    print(f"\n汇总结果: {summary_file}")
    
    # 飞书通知
    try:
        msg = f"""⚔️ CTR 全量 Benchmark 完成

模型数: {len(success_results)}
总耗时: {elapsed/60:.1f} 分钟

Top 5 模型:
"""
        for r in success_results[:5]:
            msg += f"  {r.get('model', '?')}: AUC={r.get('auc', 0):.4f}, PCOC={r.get('pcoc', 0):.3f}\n"
        
        FeishuNotifier.notify(msg)
        print("✅ 飞书通知已发送")
    except Exception as e:
        print(f"飞书通知失败: {e}")
    
    return success_results


if __name__ == '__main__':
    main()
