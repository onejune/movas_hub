# MetaSpore Tools

## feature_importance.py

Permutation Importance 特征重要性分析。

### 用法

```bash
cd /mnt/workspace/walter.wan/git_project/github_onejune/movas_hub/DeepForgeX/MetaSpore

# 分析指定特征
python tools/feature_importance.py \
    --conf ./workshop/ctr/dnn_ivr16_v2/conf/widedeep.yaml \
    --model_date 2026-03-02 \
    --sample_date 2026-03-03 \
    --features country,adx,city,business_type

# 分析全部特征（从 schema 读取）
python tools/feature_importance.py \
    --conf ./workshop/ctr/dnn_ivr16_v2/conf/widedeep.yaml \
    --model_date 2026-03-02 \
    --sample_date 2026-03-03 \
    --schema ./workshop/ctr/dnn_ivr16_v2/conf/combine_schema

# 分组评估（同时打乱组内所有特征，评估整体贡献）
python tools/feature_importance.py \
    --conf ./workshop/ctr/dnn_ivr16_v2/conf/widedeep.yaml \
    --model_date 2026-03-02 \
    --sample_date 2026-03-03 \
    --schema ./workshop/ctr/dnn_ivr16_v2/conf/combine_schema \
    --group "realtime:huf_*,ruf_*,ruf2_*" \
    --group "inner:duf_inner_*" \
    --group "outer:duf_outer_*" \
    --group "base:country,city,adx,bundle,tagid,os"

# 指定输出目录
python tools/feature_importance.py \
    --conf ... \
    --output ./my_output

# 清除缓存重新分析
python tools/feature_importance.py \
    --conf ... \
    --clear
```

### 参数

| 参数 | 必填 | 说明 |
|------|:----:|------|
| `--conf, -c` | ✓ | 配置文件 yaml |
| `--model_date, -m` | ✓ | 模型日期 |
| `--sample_date, -s` | ✓ | 样本日期 |
| `--features, -f` | * | 特征列表，逗号分隔 |
| `--schema` | * | combine_schema 文件 |
| `--group, -g` | | 分组评估，可多次指定 |
| `--eval_keys, -e` | | 评估维度，默认 business_type |
| `--output, -o` | | 输出目录，默认 ./pi_output |
| `--workdir, -w` | | 工作目录 |
| `--clear` | | 清除缓存重新开始 |

`*` features 和 schema 至少指定一个

### 分组评估

分组格式：`name:pattern1,pattern2,...`

支持通配符 `*`：
- `huf_*` 匹配所有以 `huf_` 开头的特征
- `*_24h` 匹配所有以 `_24h` 结尾的特征

分组评估会**同时打乱组内所有特征**，测量整个特征组的贡献。适用于：
- 评估交叉特征组（如所有 `#demand_pkgname` 相关特征）
- 评估时间窗口特征组（如所有实时特征 vs 离线特征）
- 评估特征类别（如用户特征 vs 物品特征）

### 输出

```
pi_output/
├── checkpoint.json    # 断点续跑
├── importance.csv     # 结果排名
├── group_details.json # 分组详情（分组模式）
└── logs/              # 验证日志
    ├── baseline.log
    ├── country.log
    └── ...
```
