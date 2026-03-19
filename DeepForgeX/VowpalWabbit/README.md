# VowpalWabbit

基于 Vowpal Wabbit 的 FTRL-LR 实现，用于与 Java FTRL-Proximal-LR 对比实验。

## 目录结构

```
VowpalWabbit/
├── tools/                    # 工具脚本
│   ├── csv2vw.c              # CSV → VW 格式转换 (C, 单线程)
│   ├── csv2vw_parallel.c     # CSV → VW 格式转换 (C, 多线程)
│   ├── csv2vw.py             # CSV → VW 格式转换 (Python)
│   ├── parquet2vw.py         # Parquet → VW 格式转换
│   ├── evaluate.py           # 评估脚本 (AUC/PCOC/LogLoss by business_type)
│   └── score_kdd.py          # KDD 格式评分
├── workshop/
│   └── ivr16_aert_vw/        # 实验目录
│       ├── conf/
│       │   └── combine_schema    # 特征配置 (与 Java 版相同)
│       ├── train.sh              # VW 训练脚本
│       ├── run_benchmark.sh      # 批量实验脚本
│       ├── run_java_ftrl.sh      # Java FTRL 对比脚本
│       └── send_report.py        # 飞书通知
└── README.md
```

## 数据转换

### CSV → VW 格式

使用 C 版本转换器（推荐，速度快 5x）:

```bash
# 编译
cd tools
gcc -O3 -o csv2vw csv2vw.c

# 转换 (从 stdin 读取，输出到 stdout)
cat /path/to/data.csv | ./csv2vw /path/to/combine_schema > train.vw
```

多线程版本:
```bash
gcc -O3 -pthread -o csv2vw_parallel csv2vw_parallel.c
cat /path/to/data.csv | ./csv2vw_parallel /path/to/combine_schema 16 > train.vw
```

### 数据格式

VW 格式示例:
```
1 'shopee_cps |s f1:v1 f2:v2 |c f1_X_f2:v1_X_v2
-1 'lazada_rta |s f1:v1 f2:v2 |c f1_X_f2:v1_X_v2
```

- **Label**: 1 (正样本) 或 -1 (负样本)
- **Tag**: `'business_type` 用于分组评估
- **Namespace `|s`**: 单特征 (41 个)
- **Namespace `|c`**: 交叉特征 (346 个)

## 训练

```bash
cd workshop/ivr16_aert_vw

# 单日训练
vw --ftrl \
    --ftrl_alpha 0.1 \
    --ftrl_beta 1 \
    --l1 1 \
    --l2 300 \
    -b 24 \
    --loss_function logistic \
    --link logistic \
    -d train.vw \
    -f model.vw

# 预测
vw -t -i model.vw -p pred.txt test.vw

# 评估
python ../../tools/evaluate.py -d test.vw -p pred.txt
```

## VW 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--ftrl` | 启用 FTRL-Proximal 优化 | - |
| `--ftrl_alpha` | 学习率参数 α | 0.1 |
| `--ftrl_beta` | 学习率参数 β | 1.0 |
| `--l1` | L1 正则化 | 1 |
| `--l2` | L2 正则化 | 300 |
| `-b` | 特征哈希位数 (2^b 桶) | 24 |
| `--loss_function logistic` | 二分类损失 | - |
| `--link logistic` | 输出概率 (sigmoid) | - |

## 实验结果 (2026-03-18)

### 1天训练 (02-18 train, 02-19 test, filtered 6 business_types)

| Model | Overall AUC | aecps | lazada_rta | lazada_cps | PCOC |
|-------|-------------|-------|------------|------------|------|
| **Java FTRL** | 0.8415 | **0.7977** | **0.7757** | **0.7771** | 1.12 |
| VW FTRL | 0.8679 | 0.7719 | 0.7525 | 0.7582 | 1.19 |

**结论**: VW Overall AUC 较高是因为测试集不同（VW 用 filtered 数据）。分业务线 AUC，Java FTRL 领先 VW 约 2 个百分点。

### 参数调优

| 配置 | Overall AUC | 备注 |
|------|-------------|------|
| baseline (b24, l1=1, l2=300) | 0.8671 | - |
| b=26 | 0.8671 | 无提升 |
| b=28 | 0.8671 | 无提升 |
| l1=0.1, l2=30 | **0.8679** | 最佳 |
| l1=0, l2=1 | 0.8648 | 过拟合 |

## 对比 Java FTRL

| 特性 | Java FTRL | VW FTRL |
|------|-----------|---------|
| 特征哈希 | BKDR Hash | MurmurHash3 |
| 交叉特征 | combine_schema 配置 | combine_schema 配置 |
| 训练方式 | 批量 | 在线/流式 |
| 内存占用 | 较高 | 极低 |
| 分业务线 AUC | **更高 (+2%)** | 较低 |

## 数据路径

- **源数据 (CSV)**: `/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v7/csv/part=YYYY-MM-DD`
- **转换后 (VW)**: `/mnt/workspace/walter.wan/vw_data/ivr16_aert_vw/`
- **特征配置**: `workshop/ivr16_aert_vw/conf/combine_schema`
