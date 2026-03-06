# AlphaFM

基于 FTRL-Proximal 优化算法的 Factorization Machine (FM) 单机多线程实现，专为大规模 CTR 预估场景设计。

> 命名灵感来自 AlphaGo 🎮

## ✨ 特性

- **流式训练** — 支持从 HDFS/管道流式读取数据，边下载边计算
- **单遍收敛** — FTRL 优化器只需过一遍数据即可收敛，无需多次迭代
- **增量训练** — 支持加载已有模型继续训练
- **稀疏解** — L1 正则化 + `force_v_sparse` 选项获得高度稀疏模型
- **内存优化** — 支持 10 亿级特征维度（128G 内存）
- **灵活退化** — 设置 `-dim 1,1,0` 即退化为标准 LR

## 📦 项目结构

```
AlphaFM-master/
├── fm_train.cpp           # 训练入口
├── fm_predict.cpp         # 预测入口
├── model_bin_tool.cpp     # 模型格式转换工具
├── Makefile               # 编译脚本
├── src/
│   ├── FTRL/
│   │   ├── ftrl_model.h       # FM 模型定义
│   │   ├── ftrl_trainer.h     # 训练器
│   │   ├── ftrl_predictor.h   # 预测器
│   │   ├── predict_model.h    # 预测模型
│   │   └── model_bin_file.h   # 二进制模型读写
│   ├── Frame/
│   │   └── pc_frame.h         # 多线程框架
│   ├── Sample/
│   │   └── fm_sample.h        # 样本解析
│   ├── Utils/
│   │   └── utils.h            # 工具函数
│   ├── Mem/
│   │   └── mem_pool.h         # 内存池
│   └── Lock/
│       └── lock_pool.h        # 锁池
├── bin/                   # 编译输出目录
└── workshop/              # 实验配置
    └── fm_exp_v1/         # 示例实验
```

## 🚀 快速开始

### 环境要求

- Linux x86_64
- g++ 8.0+ (支持 C++11)
- pthread

### 编译

```bash
make
```

生成可执行文件：
- `bin/fm_train` — 训练工具
- `bin/fm_predict` — 预测工具
- `bin/model_bin_tool` — 模型格式转换工具

---

## 📖 使用指南

### 训练

```bash
# 从 HDFS 流式训练
hadoop fs -cat /path/to/train_data | ./bin/fm_train [options] -m model.txt

# 本地文件训练
cat train_data.txt | ./bin/fm_train [options] -m model.txt
```

**训练参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m <path>` | 模型输出路径 | 必填 |
| `-mf <format>` | 模型格式：`txt` 或 `bin` | txt |
| `-dim <k0,k1,k2>` | k0=偏置, k1=一阶, k2=二阶向量维度 | 1,1,8 |
| `-init_stdev <σ>` | 二阶因子初始化标准差 | 0.1 |
| `-w_alpha <α>` | w 的 FTRL 学习率参数 α | 0.05 |
| `-w_beta <β>` | w 的 FTRL 学习率参数 β | 1.0 |
| `-w_l1 <λ1>` | w 的 L1 正则化 | 0.1 |
| `-w_l2 <λ2>` | w 的 L2 正则化 | 5.0 |
| `-v_alpha <α>` | v 的 FTRL 学习率参数 α | 0.05 |
| `-v_beta <β>` | v 的 FTRL 学习率参数 β | 1.0 |
| `-v_l1 <λ1>` | v 的 L1 正则化 | 0.1 |
| `-v_l2 <λ2>` | v 的 L2 正则化 | 5.0 |
| `-core <n>` | 线程数 | 1 |
| `-im <path>` | 初始模型路径（增量训练） | - |
| `-imf <format>` | 初始模型格式 | txt |
| `-fvs <0/1>` | 强制稀疏：wi=0 时令 vi=0 | 0 |
| `-mnt <type>` | 模型数值类型：`double` 或 `float` | double |

### 预测

```bash
cat test_data.txt | ./bin/fm_predict [options] -m model.txt -out predictions.txt
```

**预测参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m <path>` | 模型文件路径 | 必填 |
| `-mf <format>` | 模型格式 | txt |
| `-dim <k2>` | 二阶向量维度 | 8 |
| `-core <n>` | 线程数 | 1 |
| `-out <path>` | 预测结果输出路径 | 必填 |
| `-mnt <type>` | 数值类型 | double |

### 模型格式转换

```bash
# 查看二进制模型信息
./bin/model_bin_tool -task 1 -im model.bin

# 二进制 → 文本
./bin/model_bin_tool -task 2 -im model.bin -om model.txt

# 二进制 → 文本（仅非零特征）
./bin/model_bin_tool -task 3 -im model.bin -om model.txt

# 文本 → 二进制
./bin/model_bin_tool -task 4 -im model.txt -om model.bin -dim 8
```

---

## 📊 数据格式

### 标准 LibSVM 格式

```
label feature1:value1 feature2:value2 ...
```

示例：
```
1 sex:1 age:0.3 city:beijing country:cn
0 sex:0 age:0.7 city:shanghai country:cn
-1 sex:1 age:0.5 city:guangzhou country:cn
```

- **label**: 1/0 或 1/-1
- **feature**: 字符串（不限于整数）
- **value**: 整数或浮点数（建议归一化）
- value=0 的特征可省略

### CSV 格式（扩展支持）

通过 `column_name` 和 `combine_schema` 配置文件，支持直接读取 CSV 格式数据并自动生成特征交叉。

---

## 📝 模型文件格式

### 文本格式

**第一行（偏置）：**
```
bias w w_n w_z
```

**其他行（特征）：**
```
feature_name w v1 v2 ... vk w_n w_z v_n1 v_n2 ... v_nk v_z1 v_z2 ... v_zk
```

其中：
- `w` — 一阶权重
- `v1...vk` — 二阶因子向量
- `w_n, w_z` — FTRL 中间变量
- `v_n1...v_nk, v_z1...v_zk` — FTRL 中间变量

### 预测结果格式

```
label score
```

- `label`: 1 或 -1
- `score`: 预测为正样本的概率

---

## 🗂️ Workshop 使用示例

### 目录结构

```
workshop/fm_exp_v1/
├── conf/
│   ├── column_name       # 特征列映射
│   └── combine_schema    # 特征交叉配置
├── run.sh                # 一键训练脚本
├── train_fm.sh           # 单日训练脚本
├── val_by_model.sh       # 模型验证脚本
├── figure_auc.py         # AUC 计算脚本
├── score_kdd.py          # KDD 评估指标
└── output/               # 模型输出目录
```

### 训练脚本示例 (train_fm.sh)

```bash
#!/bin/bash

# FTRL 参数配置
para="-init_stdev 0.01 -w_l2 5.0 -w_l1 5 -v_l1 0.1 \
      -w_alpha 0.1 -v_l2 5.0 -v_alpha 0.01 \
      -w_beta 1.0 -v_beta 1.0 -core 16 -dim 1,1,8"

model_out="./output/base"
train_date=$1
train_data="../sample/sample_${train_date}"

# 首次训练
if [[ ! -f "${model_out}" ]]; then
    cat ${train_data} | ./bin/fm_train $para -m ${model_out}
# 增量训练
else
    cat ${train_data} | ./bin/fm_train $para -m ${model_out} -im ${model_out}
fi

# 保存当天模型快照
cp ${model_out} ${model_out}.${train_date}
echo "Feature count: $(wc -l ${model_out})"
```

### 验证脚本示例 (val_by_model.sh)

```bash
#!/bin/bash

model_date=$1
sample_date=$2

model_path="./output/base.${model_date}"
sample_path="../sample/sample_${sample_date}"

# 预测
cat ${sample_path} | ./bin/fm_predict -m ${model_path} -out ${model_path}.predict -core 8 -dim 8

# 计算 AUC
cat ${model_path}.predict | python figure_auc.py
```

### 批量训练 (run.sh)

```bash
#!/bin/bash

start_date="20250101"
end_date="20250115"

current_date=$start_date
while [ "$current_date" -le "$end_date" ]; do
    echo "========== Training: $current_date =========="
    train_date=$(date -d "$current_date" +"%Y-%m-%d")
    sh train_fm.sh $train_date
    current_date=$(date -d "$current_date + 1 day" +"%Y%m%d")
done

# 验证最后一天
sh val_by_model.sh 2025-01-14 2025-01-15
```

---

## ⚙️ 高级用法

### 退化为 LR

设置 `-dim 1,1,0` 禁用二阶交互，FM 退化为标准 LR：

```bash
cat train.txt | ./bin/fm_train -dim 1,1,0 -m lr_model.txt
```

### 强制稀疏

启用 `-fvs 1` 后，当 wi=0 时会强制将对应的 vi 向量置零，获得更稀疏的模型：

```bash
cat train.txt | ./bin/fm_train -fvs 1 -w_l1 1.0 -m sparse_model.txt
```

### 二进制模型

使用二进制格式可提升 10 倍模型加载速度：

```bash
# 训练时直接输出二进制
cat train.txt | ./bin/fm_train -mf bin -m model.bin

# 预测时加载二进制
cat test.txt | ./bin/fm_predict -mf bin -m model.bin -out pred.txt
```

### 内存优化

使用 `-mnt float` 将模型参数类型改为 float，内存占用减半：

```bash
cat train.txt | ./bin/fm_train -mnt float -m model.txt
```

---

## 💡 调参建议

| 场景 | 建议配置 |
|------|----------|
| 特征稀疏、需要特征选择 | 增大 `-w_l1` |
| 二阶交互过拟合 | 增大 `-v_l2`，减小 `-init_stdev` |
| 训练速度优先 | 增大 `-core`，使用 `-mnt float` |
| 模型过大 | 启用 `-fvs 1`，增大 `-w_l1` |
| 加载速度优先 | 使用 `-mf bin` 二进制格式 |

### 参考参数组合

```bash
# 基础配置
-dim 1,1,8 -init_stdev 0.1 -core 16

# 稀疏配置
-w_l1 5.0 -w_l2 5.0 -v_l1 0.1 -v_l2 5.0

# 学习率配置
-w_alpha 0.1 -w_beta 1.0 -v_alpha 0.05 -v_beta 1.0
```

---

## 📈 性能参考

**测试环境：**
- 样本数：1000 万
- 特征维度：200 万
- CPU：2.10GHz
- 线程数：10

**参数配置：**
```bash
-dim 1,1,2 -w_l1 0.05 -v_l1 0.05 -init_stdev 0.001 -w_alpha 0.01 -v_alpha 0.01 -core 10
```

**训练时间：约 10 分钟**

---

## 📚 参考资料

- [FM 原理 + FTRL 优化详解](http://castellanzhang.github.io/2016/10/16/fm_ftrl_softmax/)
- [AlphaFM 内存优化](http://castellanzhang.github.io/2018/09/01/memory_optimization_for_alphafm/)
- [Ad Click Prediction: a View from the Trenches (Google 2013)](https://research.google/pubs/pub41159/)
- [Factorization Machines (Rendle 2010)](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
