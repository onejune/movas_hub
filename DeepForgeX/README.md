# DeepForgeX

CTR/CVR/LTV 预估模型工具集，涵盖从传统机器学习到深度学习的完整解决方案。

## 🏗️ 项目概览

```
DeepForgeX/
├── FTRL-Proximal-LR/     # ⭐ Java FTRL LR — 主力生产模型
├── AlphaFM-master/       # C++ FM — 高性能因子分解机
├── AlphaPLM-master/      # C++ PLM — 分片线性模型
├── MetaSpore/            # PyTorch DNN — 深度学习框架
└── utils/                # 公共工具库（Shell + Python）
```

## 📦 子项目说明

| 子项目 | 语言 | 模型 | 特点 | 详细文档 |
|--------|------|------|------|----------|
| **FTRL-Proximal-LR** | Java | LR | 生产主力，支持 6 种损失函数 | [README](FTRL-Proximal-LR/README.md) |
| **AlphaFM-master** | C++ | FM | 流式训练，10 亿级特征 | [README](AlphaFM-master/README.md) |
| **AlphaPLM-master** | C++ | PLM | 分片线性，非线性建模 | [README](AlphaPLM-master/README.md) |
| **MetaSpore** | PyTorch | DNN | WideDeep/DeepFM/DCN 等 | [README](MetaSpore/README.md) |

---

## 🚀 快速开始

### FTRL-Proximal-LR (Java LR)

```bash
cd FTRL-Proximal-LR
mvn clean package -DskipTests

java -jar target/ftrl_maven_walter-0.0.1-SNAPSHOT.jar \
    -i /path/to/train_data \
    -c /path/to/conf/ \
    -n exp_name \
    -f f
```

### AlphaFM-master (C++ FM)

```bash
cd AlphaFM-master
make

cat train.txt | ./bin/fm_train -dim 1,1,8 -core 16 -m model.txt
```

### MetaSpore (PyTorch DNN)

```bash
cd MetaSpore/workshop/wd_v1
bash run_train.sh
```

---

## 🎯 选型指南

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| **日常 CTR 预估** | FTRL-Proximal-LR | 稳定、易部署、支持增量训练 |
| **高维稀疏 + 交叉** | AlphaFM-master | 单遍收敛、内存优化、10 亿级特征 |
| **复杂特征交互** | MetaSpore (DeepFM/DCN) | 自动学习高阶交叉 |
| **LTV 预估** | MetaSpore (ZILN/WCE) | 专用损失函数 |
| **超大规模** | AlphaFM + 流式训练 | 边下载边计算 |

---

## 📊 支持的模型

### 传统机器学习

| 模型 | 实现 | 说明 |
|------|------|------|
| LR (Logistic Regression) | FTRL-Proximal-LR | FTRL 优化，支持特征交叉 |
| FM (Factorization Machine) | AlphaFM-master | 二阶特征交互，可退化为 LR |
| PLM (Piece-wise Linear Model) | AlphaPLM-master | 分片线性，非线性建模 |

### 深度学习

| 模型 | 实现 | 说明 |
|------|------|------|
| Wide & Deep | MetaSpore | Wide (LR) + Deep (DNN) |
| DeepFM | MetaSpore | FM + DNN |
| DCN | MetaSpore | Deep & Cross Network |
| FFM | MetaSpore | Field-aware FM |
| xDeepFM | MetaSpore | Compressed Interaction Network |

---

## 🛠️ 公共工具库 (utils/)

`utils/` 目录包含所有子项目共用的脚本和工具。

### Shell 脚本

| 脚本 | 功能 |
|------|------|
| `lib_common.sh` | FTRL 训练公共函数库（数据准备、训练执行、模型保存、验证、清理） |
| `dnn_lib_common.sh` | DNN 训练公共函数库（环境检查、模型训练、验证） |
| `kill_trainer.sh` | 终止训练进程 |
| `rsyn_git.sh` | Git 同步脚本 |

### Python 工具

| 脚本 | 功能 |
|------|------|
| `dnn_trainFlow.py` | DNN 训练主流程（支持 WideDeep/DeepFM/DCN/PEPNet 等） |
| `ltv_trainFlow.py` | LTV 预估训练流程 |
| `winrate_trainFlow.py` | Winrate 预估训练流程 |
| `MTL_trainFlow.py` | 多任务学习训练流程 |
| `MDL_trainFlow.py` | 多域学习训练流程 |

### 评估工具

| 脚本 | 功能 |
|------|------|
| `metrics_eval.py` | 评估指标计算（AUC、PCOC、LogLoss） |
| `score_kdd.py` | KDD 评估指标（NWMAE、WRMSE、分桶校准） |
| `figure_auc_regression.py` | AUC/校准度可视化 |

### 辅助工具

| 脚本 | 功能 |
|------|------|
| `feishu_notifier.py` | 飞书消息通知（训练完成/验证结果） |
| `movas_logger.py` | 统一日志工具（支持 Spark 环境） |
| `feature_stat.py` | 特征统计分析 |
| `filter_old_feature.py` | 过滤过期特征 |
| `find_missed_feature.py` | 查找缺失特征 |

### 使用方式

在实验目录中引入公共库：

```bash
# FTRL 训练脚本
source /path/to/utils/lib_common.sh
init_env
model_train

# DNN 训练脚本
source /path/to/utils/dnn_lib_common.sh
env_check
model_train
```

---

## 🔧 核心配置文件

### FTRL (FTRL-Proximal-LR)

| 文件 | 作用 |
|------|------|
| `train.conf` | 训练参数（损失函数、采样率、权重） |
| `para.conf` | FTRL 超参数 (alpha, beta, L1, L2) |
| `column_name` | 特征列索引映射 |
| `combine_schema` | 特征交叉配置 |

### DNN (MetaSpore)

| 文件 | 作用 |
|------|------|
| `widedeep.yaml` | 模型配置（架构、超参、数据路径） |
| `combine_schema` | 特征和交叉特征定义 |

---

## 📈 评估指标

所有模型统一支持以下评估指标：

- **AUC** — 排序能力
- **LogLoss** — 概率校准
- **PCOC** — 预测/实际比值（校准度）
- **KDD 指标** — 分桶校准分析

---

## 🔗 相关链接

- **GitLab**: https://gitlab.yeahmobi.com/dsp-algo/alpha_lr_ftrl.git

## 📚 参考论文

- [Ad Click Prediction: a View from the Trenches (FTRL, Google 2013)](https://research.google/pubs/pub41159/)
- [Factorization Machines (Rendle 2010)](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
- [Wide & Deep Learning (Google 2016)](https://arxiv.org/abs/1606.07792)
- [DeepFM (Huawei 2017)](https://arxiv.org/abs/1703.04247)
- [DCN V2 (Google 2020)](https://arxiv.org/abs/2008.13535)
