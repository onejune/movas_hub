# Permutation Importance 工具

真正的 Permutation Importance 分析工具，通过打乱特征值并重新推理来评估特征重要性。

## 原理

1. 计算基准 AUC（原始数据）
2. 对每个特征：打乱其值 → 重新推理 → 计算新 AUC
3. 重要性 = 基准 AUC - 打乱后 AUC
4. 下降越多，特征越重要

## 使用方法

```bash
# 基本用法
python permutation_importance.py \
    --project_dir /path/to/dnn_ivr16_v1 \
    --model_date 2026-03-02 \
    --sample_date 2026-03-03

# 指定特征子集（加速测试）
python permutation_importance.py \
    --project_dir /path/to/dnn_ivr16_v1 \
    --model_date 2026-03-02 \
    --sample_date 2026-03-03 \
    --features country,adx,business_type

# 并行执行（多个特征同时分析）
python permutation_importance.py \
    --project_dir /path/to/dnn_ivr16_v1 \
    --model_date 2026-03-02 \
    --sample_date 2026-03-03 \
    --parallel 3
```

## 输出

- `output/permutation_importance/importance.csv` - 完整结果
- `output/permutation_importance/summary.txt` - 摘要报告
- `output/permutation_importance/to_remove.txt` - 建议删除的特征

## 注意事项

- 每个特征需要一次完整推理，253 个特征约需 4-5 小时
- 建议先用 `--features` 测试少量特征
- 支持断点续跑（已完成的特征会跳过）

## 版本历史

- v1.0.0 (2026-04-05): 初始版本
