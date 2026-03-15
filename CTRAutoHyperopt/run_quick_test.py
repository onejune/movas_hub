"""
快速测试脚本 - 用小数据量验证流程
"""

import os
import sys

# 修改配置为快速测试模式
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
os.environ['OMP_NUM_THREADS'] = '4'

# 导入主模块并修改配置
import ctr_autopytorch as ctr

# 快速测试配置
ctr.TRAIN_DAYS = 2          # 只用 2 天训练
ctr.TEST_DAYS = 1           # 只用 1 天测试
ctr.SAMPLE_FRAC = 0.1       # 采样 10%
ctr.TOTAL_WALLTIME_LIMIT = 300   # 5 分钟搜索
ctr.FUNC_EVAL_TIME_LIMIT = 60    # 1 分钟单次评估
ctr.HIGH_CARDINALITY_THRESHOLD = 50  # 降低阈值

if __name__ == '__main__':
    print("="*60)
    print("快速测试模式")
    print("="*60)
    print(f"训练天数: {ctr.TRAIN_DAYS}")
    print(f"测试天数: {ctr.TEST_DAYS}")
    print(f"采样比例: {ctr.SAMPLE_FRAC}")
    print(f"搜索时间: {ctr.TOTAL_WALLTIME_LIMIT}s")
    print("="*60 + "\n")
    
    api, results = ctr.main()
