# Permutation Importance 工具

真正的 Permutation Importance 分析工具，通过打乱特征值并重新推理来评估特征重要性。

## 原理

1. 计算基准 AUC（原始数据）
2. 对每个特征：打乱其值 → 重新推理 → 计算新 AUC
3. 重要性 = 基准 AUC - 打乱后 AUC
4. 下降越多，特征越重要

## 前置条件

1. 项目已有训练好的模型 (`output/model_YYYY-MM-DD/`)
2. `base_trainFlow.py` 已支持 `--shuffle_feature` 参数（v1.0.0+ 已内置）
3. 项目有 `validation.sh` 或 `validation_perm.sh`

## 使用方法

### 方法 1：使用 Python 脚本（推荐）

```bash
# 进入工具目录
cd /mnt/workspace/walter.wan/git_project/github_onejune/movas_hub/DeepForgeX/MetaSpore/workshop/tools/permutation_importance

# 分析所有特征（耗时较长，约 3-5 分钟/特征）
python permutation_importance.py \
    --project_dir ../ctr/dnn_ivr16_v2 \
    --model_date 2026-03-02 \
    --sample_date 2026-03-03

# 只分析指定特征（快速测试）
python permutation_importance.py \
    --project_dir ../ctr/dnn_ivr16_v2 \
    --model_date 2026-03-02 \
    --sample_date 2026-03-03 \
    --features country,adx,city,business_type
```

### 方法 2：手动运行单个特征

```bash
cd /path/to/your/project

# 基准评估
bash validation_perm.sh 2026-03-02 2026-03-03 "" business_type

# 打乱 country 特征
bash validation_perm.sh 2026-03-02 2026-03-03 country business_type
```

## 输出文件

分析完成后，结果保存在 `output/permutation_importance/`：

- `results.json` - 完整结果（支持断点续跑）
- `importance.csv` - 特征重要性排名
- `to_remove.txt` - 建议删除的低重要性特征
- `summary.txt` - 分析摘要
- `logs/` - 每个特征的验证日志

## 示例输出

```
Top 10 重要特征:
   feature  importance  importance_pct  shuffled_auc  level  rank
   country      0.0063          0.75%        0.8314   HIGH     1
       adx      0.0045          0.54%        0.8332   HIGH     2
      city      0.0038          0.45%        0.8339   HIGH     3
    ...
```

## 注意事项

1. **耗时长**：每个特征需要一次完整推理，253 个特征约需 12-15 小时
2. **支持断点续跑**：已分析的特征会缓存，中断后可继续
3. **资源消耗**：每次推理需要 Spark 资源，建议在空闲时段运行
4. **建议先测试**：用 `--features` 参数先测试少量特征

## 版本历史

- v1.0.0 (2026-04-05): 初始版本
  - 支持特征打乱 + 重新推理
  - 断点续跑
  - 自动报告生成
