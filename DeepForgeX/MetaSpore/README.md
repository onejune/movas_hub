# MetaSpore DNN 训练框架

基于 PyTorch + PySpark 的分布式深度学习 CTR/CVR 预估训练框架，支持 Wide&Deep、DeepFM、DCN 等主流推荐模型，以及延迟反馈（DEFER）、多任务学习（MTL）、多域学习（MDL）等高级场景。

---

## ✨ 特性

- **分布式训练** — 基于 PySpark + PS（Parameter Server）架构
- **多种模型架构** — WideDeep、DeepFM、DCN、FFM、PNN、MTL、MDL 等 13+ 种模型
- **灵活的特征工程** — 通过 `combine_schema` 配置稀疏特征和特征交叉
- **增量训练** — 支持加载已有模型继续训练
- **多种损失函数** — CrossEntropy、WCE、MSE、ZILN、DEFER loss 等
- **飞书通知** — 训练完成自动发送飞书通知（读取 `readme` 文件内容）
- **统一 TrainFlow 体系** — 所有任务共用 `BaseTrainFlow`，扩展方便
- **现代化包结构** — 支持标准包导入，零文件复制架构

---

## 📦 项目结构

```
MetaSpore/
├── python/
│   └── metaspore/
│       ├── algos/                      # 模型算法库
│       │   ├── widedeep_net.py         # Wide & Deep
│       │   ├── deepfm_net.py           # DeepFM
│       │   ├── dcn_net.py / dcn_v2_net.py  # DCN / DCN V2
│       │   ├── ffm_net.py              # FFM
│       │   ├── pnn_net.py              # PNN
│       │   ├── xdeepfm_net.py          # xDeepFM
│       │   ├── autoint_net.py          # AutoInt
│       │   ├── ziln_model.py           # ZILN (LTV)
│       │   ├── deep_censored_model.py  # DeepCensor (Winrate)
│       │   ├── delay_feedback/         # 延迟反馈模型
│       │   │   ├── defer_models.py     # WinAdaptDNN / DeferDNN
│       │   │   ├── defer_loss.py       # delay_win_select_loss_v1/v2
│       │   │   └── delf.py             # DELF (Weibull + IPW)
│       │   ├── multi_task/             # 多任务学习
│       │   └── multi_domain/           # 多域学习
│       ├── estimator.py                # 训练器（支持多维标签）
│       ├── loss_utils.py               # 损失函数注册表
│       ├── trainflows/                 # TrainFlow 包模块
│       │   ├── __init__.py             # 包入口
│       │   ├── base.py                 # BaseTrainFlow
│       │   ├── dnn.py                  # DNNTrainFlow
│       │   ├── mtl.py                  # MTLTrainFlow
│       │   ├── defer.py                # DeferTrainFlow
│       │   ├── delf.py                 # DELFTrainFlow
│       │   ├── ltv.py                  # LTVTrainFlow
│       │   ├── mdl.py                  # MDLTrainFlow
│       │   ├── winrate.py              # WinRateTrainFlow
│       │   └── metacpl.py              # MetaCPLTrainFlow
│       └── utils/                      # 工具类
│           ├── __init__.py             # 包入口
│           ├── logger.py               # 日志
│           ├── metrics.py              # 评估指标
│           └── notifier.py             # 飞书通知
│
├── workshop/                           # 实验目录（按任务分类）
│   ├── ctr/                            # CTR/CVR 单任务
│   ├── delay_feedback/                 # 延迟反馈 (DEFER/DELF)
│   ├── ltv/                            # LTV 预估
│   ├── mtl/                            # 多任务学习
│   ├── mdl/                            # 多域学习
│   ├── winrate/                        # 竞价/出价模型
│   └── archive/                        # 归档
│
└── utils/                              # 工具脚本（源文件，勿直接改 src/）
    ├── trainflows/                     # TrainFlow 源文件（已迁移至包结构）
    │   ├── base_trainFlow.py           # 基类（已迁移）
    │   ├── dnn_trainFlow.py            # CTR/CVR 单任务（已迁移）
    │   ├── defer_trainFlow.py          # 延迟反馈 DEFER（已迁移）
    │   ├── MTL_trainFlow.py            # 多任务学习（已迁移）
    │   ├── MDL_trainFlow.py            # 多域学习（已迁移）
    │   ├── ltv_trainFlow.py            # LTV 预估（已迁移）
    │   ├── winrate_trainFlow.py        # Winrate（已迁移）
    │   └── DELF_trainFlow.py           # DELF（已迁移）
    ├── tools/                          # 工具类（已迁移至 metaspore.utils）
    │   ├── feishu_notifier.py          # 飞书通知（已迁移）
    │   ├── movas_logger.py             # 日志（已迁移）
    │   └── metrics_eval.py             # 评估指标（已迁移）
    └── scripts/
        ├── dnn_lib_common.sh           # 公共 shell 函数
        └── lib_common.sh
```

---

## 🚀 新建实验项目

### 1. 选择合适的实验目录

```bash
# CTR/CVR 单任务
cd workshop/ctr/

# 延迟反馈
cd workshop/delay_feedback/defer/

# 多任务学习
cd workshop/mtl/
```

### 2. 创建新实验（零文件复制架构）

```bash
# 创建新实验目录
mkdir workshop/ctr/my_experiment
cd workshop/ctr/my_experiment

# 使用标准模板创建项目
# 无需复制任何源代码文件！
```

### 3. 实验目录标准结构

```
my_experiment/
├── conf/
│   ├── config.yaml         # 主配置文件（必须）
│   ├── combine_schema      # 特征配置（必须）
│   └── column_name         # 列名映射（可选）
├── train.py                # 训练入口（新架构，零复制）
├── run_train.sh            # 训练启动脚本（统一脚本）
├── readme                  # 实验说明（必须，飞书通知会读取）
├── log/                    # 日志输出（gitignore）
└── output/                 # 模型输出（gitignore）
```

### 4. 编写 `readme` 文件（必须）

`readme` 文件（注意：不是 `README.md`）会被 `train.py` 自动读取，并在训练完成后发送到飞书群。

```
实验名称: my_experiment

实验变量:
- 新增 xxx 特征
- 调整 embedding_size: 8 -> 12

训练配置:
- 训练数据: 2026-02-11 ~ 2026-02-17 (7天)
- 验证数据: 2026-02-18
- 数据源: ivr_sample_v7
```

### 5. 配置 `train.py`（新架构）

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MetaSpore DNN Training Framework
Zero-Copy Architecture Template
"""
import argparse
import sys
import os

def ensure_metaspore_path():
    """确保 metaspore 包路径正确"""
    # 添加当前项目的 metaspore 路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..')
    metaspore_path = os.path.join(project_root, 'MetaSpore', 'python')
    
    if metaspore_path not in sys.path:
        sys.path.insert(0, metaspore_path)

def parse_args():
    parser = argparse.ArgumentParser(description='DNN Training')
    parser.add_argument('--config', type=str, required=True, help='config yaml path')
    parser.add_argument('--business-type', type=str, default='all', help='business type')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 确保包路径正确
    ensure_metaspore_path()
    
    # 导入训练流程
    from metaspore.trainflows import DNNTrainFlow
    
    # 运行完整训练流程
    trainer = DNNTrainFlow()
    trainer.run_complete_flow(args.config, args.business_type)

if __name__ == "__main__":
    main()
```

### 6. 配置 `run_train.sh`（统一脚本）

```bash
#!/bin/bash

# 统一训练脚本 - 支持 train/validate/help 命令
# 使用标准包结构，无需复制任何源文件

# 设置 Python 路径
export METASPORE_DIR="/mnt/workspace/walter.wan/git_project/github_onejune/movas_hub/DeepForgeX/MetaSpore/python"
export PYTHONPATH="$METASPORE_DIR:$PYTHONPATH"

case "${1:-train}" in
    train)
        echo "Starting training..."
        python train.py --config ./conf/config.yaml --business-type all
        ;;
    validate)
        echo "Starting validation..."
        # Add validation command if needed
        ;;
    help)
        echo "Usage: $0 [train|validate|help]"
        echo "  train: Start training (default)"
        echo "  validate: Start validation"
        echo "  help: Show this help"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Usage: $0 [train|validate|help]"
        exit 1
        ;;
esac
```

---

## ⚙️ 配置文件说明

### `conf/config.yaml` — 主配置

```yaml
# ============ 基础信息 ============
app_name: "my_experiment"
experiment_name: "my_experiment"

# ============ 数据路径 ============
train_path_prefix: "/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v7/parquet"
train_start_date: 2026-02-11
train_end_date: 2026-02-17
validation_date: 2026-02-18
movas_log_output_path: "./log/train_log.txt"

# ============ 模型输出 ============
model_out_base_path: "./output"
model_in_path: null             # 增量训练时填写，如 "./output/model_2026-02-17"

# ============ 模型类型 ============
model_type: "WideDeep"          # 见下方支持列表

# ============ 损失函数 ============
loss_func: "cross_entropy"      # 见下方支持列表

# ============ 特征配置 ============
combine_schema_path: "./conf/combine_schema"
wide_combine_schema_path: "./conf/combine_schema"
input_label_column_index: 0     # 标签列索引（或用 input_label_column_name）

# ============ 模型超参 ============
embedding_size: 8
dnn_hidden_units: [256, 256, 128]
use_wide: true
batch_norm: true
net_dropout: 0.5

# ============ 优化器 ============
adam_learning_rate: 0.001
ftrl_l1: 1.0
ftrl_l2: 120.0
ftrl_alpha: 0.5
ftrl_beta: 1.0

# ============ Spark 配置 ============
local_spark: true               # true=本地模式, false=YARN
worker_count: 8
server_count: 2
batch_size: 512
worker_memory: "8G"
server_memory: "8G"
coordinator_memory: "4G"
```

### 支持的模型类型

| `model_type` | 说明 |
|---|---|
| `WideDeep` | Wide & Deep Learning |
| `DeepFM` | Deep Factorization Machine |
| `DCN` | Deep & Cross Network |
| `DCNV2` | DCN V2 |
| `FFM` | Field-aware FM |
| `PNN` | Product-based NN |
| `xDeepFM` | eXtreme Deep FM |
| `AutoInt` | AutoInt |
| `WinAdaptDNN` | DEFER 专用，4输出头（cv+3时间窗口权重） |
| `DeferDNN` | DEFER 专用，多头输出 |

### 支持的损失函数

| `loss_func` | 说明 | 适用场景 |
|---|---|---|
| `cross_entropy` | 标准交叉熵 | CTR/CVR |
| `wce_loss` | 加权交叉熵 | LTV |
| `mse_loss` | 均方误差 | 回归 |
| `ziln_loss` | Zero-Inflated LogNormal | LTV |
| `defer_loss_v1` | 14维标签，SPM loss，短/长窗口均可 | DEFER |
| `defer_loss_v2` | 8维标签，简化版 | DEFER |

### `conf/combine_schema` — 特征配置

```
# 单特征（一行一个）
campaignsetid
campaignid
country
bundle
osv
make
model

# 交叉特征（用 # 分隔）
adx#country
bundle#campaignid
osv#make#campaignsetid
```

---

## 🔧 TrainFlow 包结构

所有 TrainFlow 现在作为标准 Python 包提供，支持直接导入：

```
metaspore.trainflows (包结构)
├── __init__.py                    # 包入口，导出所有 TrainFlow
├── base.py                       # BaseTrainFlow
├── dnn.py                        # DNNTrainFlow
├── mtl.py                        # MTLTrainFlow
├── defer.py                      # DeferTrainFlow
├── delf.py                       # DELFTrainFlow
├── ltv.py                        # LTVTrainFlow
├── mdl.py                        # MDLTrainFlow
├── winrate.py                    # WinRateTrainFlow
└── metacpl.py                    # MetaCPLTrainFlow
```

### 标准导入方式

```python
# 新架构 - 直接从包导入
from metaspore.trainflows import DNNTrainFlow, BaseTrainFlow
from metaspore.trainflows import DeferTrainFlow, DELFTrainFlow
from metaspore.trainflows import LTVTrainFlow, MDLTrainFlow
from metaspore.trainflows import WinRateTrainFlow, MTLTrainFlow

# 也可以直接导入特定模块
from metaspore.trainflows.dnn import DNNTrainFlow
from metaspore.trainflows.mtl import MTLTrainFlow
```

### BaseTrainFlow 提供的公共方法

| 方法 | 说明 |
|---|---|
| `_load_config` | 加载 YAML 配置 |
| `_init_spark` / `_stop_spark` | Spark 会话管理 |
| `random_sample` | 数据采样 |
| `_read_dataset_by_date` | 按日期读取数据（子类可重写） |
| `_train_model` | 单日模型训练 |
| `_predict_data` | 模型预测 |
| `_evaluate_model` | AUC/PCOC 评估 |
| `_run_training_loop` | 多日增量训练循环 |
| `run_complete_flow` | 完整训练+验证流程 |
| `_preprocess` / `_postprocess` | 训练前后钩子（子类可重写） |

### 自定义 TrainFlow（示例）

```python
from metaspore.trainflows import BaseTrainFlow

class MyTrainFlow(BaseTrainFlow):
    def _build_model_module(self):
        """必须实现：构建模型"""
        import metaspore as ms
        model = MyCustomModel(...)
        module = ms.PyTorchAgent(model, ...)
        return module

    def _read_dataset_by_date(self, date_str):
        """可选重写：自定义数据读取"""
        df = super()._read_dataset_by_date(date_str)
        # 添加自定义特征
        df = df.withColumn("my_feature", ...)
        return df
```

## 📊 支持的模型

| 模型 | 实现 | 说明 |
|------|------|------|
| Wide & Deep | MetaSpore | Wide (LR) + Deep (DNN) |
| **WideDeepDense** | MetaSpore | Wide & Deep + Dense 特征支持 |
| DeepFM | MetaSpore | FM + DNN |
| DCN | MetaSpore | Deep & Cross Network |
| FFM | MetaSpore | Field-aware FM |
| xDeepFM | MetaSpore | Compressed Interaction Network |

---

## 🆕 Dense 特征支持

MetaSpore 框架新增 **Dense 特征** 处理能力，支持连续值特征（如统计特征、数值特征）与稀疏特征联合建模。

### 支持的编码器

| 编码器 | 说明 | 输出维度 |
|--------|------|----------|
| `linear` | BatchNorm + Linear + Dropout | 可配置 |
| `minmax` | Min-Max 归一化到 [0,1] | = 特征数 |
| `standard` | Z-score 标准化 | = 特征数 |
| `log` | 对数变换 log(x+1) | = 特征数 |
| `numeric` | 每特征独立 MLP (AutoDis) | 特征数 × embedding_dim |

### 配置方式

**1. 创建 dense 特征列表文件** (`conf/dense_features`)

```
# 每行一个特征名
duf_imp_3d
duf_imp_7d
duf_re_3d
duf_re_7d
...
```

**2. 配置 YAML**

```yaml
model_type: WideDeepDense

# Dense 特征配置
dense_features_path: ./conf/dense_features
dense_encoder_type: linear    # linear|minmax|standard|log|numeric
dense_output_dim: 32          # 可选，映射到固定维度

# numeric 编码器专用
dense_embedding_dim: 16
dense_hidden_dim: 64
```

### 数据流

```
Parquet 数据
    ↓
base_trainFlow 加载 sparse_fea_list + dense_fea_list
    ↓
WideDeepDense 模型
    ├── Sparse → EmbeddingSumConcat → [batch, sparse_dim]
    ├── Dense  → Encoder → [batch, dense_dim]
    └── Concat → MLP → output
```

### 示例

```python
# train.py 自动处理
from metaspore.trainflows import DNNTrainFlow
self.model_module = WideDeepDense(
    dense_fea_list=self.dense_fea_list,  # 从配置加载
    dense_encoder_type='numeric',
    dense_output_dim=32,
    ...
)
```

---

## 📝 DEFER 延迟反馈实验

### 标签约定

| 列名 | 维度 | 用途 |
|---|---|---|
| `label` | 1维，0/1 | AUC/PCOC 评估（原始标签） |
| `defer_label` | 14维或8维 JSON 字符串 | DEFER loss 计算 |

> ⚠️ `defer_label` 使用 JSON 字符串存储（非 ArrayType），避免触发 PS Agent 注册问题。

### 14维标签格式（defer_loss_v1）

```
[0]  label_11: 延迟正样本（label=1, diff > delay）
[1]  label_10: 真负样本（label=0）
[2]  label_01_win1: 窗口1内转化（增量）
[3]  label_01_win2: 窗口2内转化（增量）
[4]  label_01_win3: 窗口3内转化（增量）
[5]  cum_win1: 窗口1累计
[6]  cum_win2: 窗口2累计
[7]  mask_win1: 样本年龄掩码（是否可观测）
[8]  mask_win2: 样本年龄掩码
[9]  reserved
[10] label_01: 所有窗口内转化
[11-13] label_11 细分
```

### 8维标签格式（defer_loss_v2）

```
[0] label_win1: 窗口1内累计转化
[1] label_win2: 窗口2内累计转化
[2] label_win3: 窗口3内累计转化
[3] label_oracle: 归因窗口内最终转化
[4] observable_win1: 样本年龄 >= win1
[5] observable_win2: 样本年龄 >= win2
[6] observable_win3: 样本年龄 >= win3
[7] observable_oracle: 通常全为 1
```

### 新建 DEFER 实验

```bash
# 创建新实验目录（零复制）
mkdir workshop/delay_feedback/defer/my_defer_experiment
cd workshop/delay_feedback/defer/my_defer_experiment

# 使用标准模板
# 无需复制任何源文件！
```

---

## 📊 评估指标

训练和验证过程输出以下指标，**必须按 `business_type` 分组评估**：

| 指标 | 说明 |
|---|---|
| AUC | Area Under ROC Curve |
| PCOC | Predicted / Observed Conversion（校准度，越接近1越好） |
| LogLoss | 对数损失 |

### 评估输出示例

```
+---------------+--------+--------+
| business_type |    AUC |   PCOC |
+---------------+--------+--------+
| type_a        | 0.8283 | 1.0234 |
| type_b        | 0.8012 | 0.9876 |
+---------------+--------+--------+
```

---

## ⚠️ MetaSpore 使用注意事项

### 1. PYTHONPATH 只能有一个 MetaSpore 路径

```bash
# ✅ 正确
export PYTHONPATH=/mnt/workspace/walter.wan/git_project/github_onejune/movas_hub/DeepForgeX/MetaSpore/python

# ❌ 错误：两个 MetaSpore 路径会导致 PS Agent 注册冲突
export PYTHONPATH=/path/A/MetaSpore/python:/path/B/MetaSpore/python
```

### 2. 避免 ArrayType 列触发 PS Agent 问题

Spark 处理 `ArrayType` 列时会启动新 worker 进程，该进程未注册 PS Agent，导致：
```
RuntimeError: no ps agent registered for thread 0x... on pid ...
```
**解决方案**：多维标签用 JSON 字符串存储，在 estimator 中解析。

### 3. 自定义 Loss Function 签名

```python
# ✅ 正确签名
def my_loss(logits, labels, minibatch=None, **kwargs) -> tuple:
    loss = ...
    return loss, loss  # 必须返回 tuple

# ❌ 错误：缺少 minibatch 参数
def my_loss(logits, labels) -> float:
    ...
```

### 4. 模型必须实现 `predict` 方法

```python
class MyModel(torch.nn.Module):
    def predict(self, logits, minibatch=None):
        """AUC 评估需要 1 维概率输出"""
        return torch.sigmoid(logits[:, 0])
```

### 5. python.zip 打包格式

```bash
# ✅ 正确：必须包含 python/ 前缀
cd MetaSpore && zip -r python.zip python/

# ❌ 错误：直接打包内容
cd MetaSpore/python && zip -r ../python.zip .
```

### 6. Python 版本

```bash
# MetaSpore 编译版本为 Python 3.8，必须使用
/root/anaconda3/envs/spore/bin/python3.8
```

### 7. 新架构 - 无需复制文件

```bash
# ✅ 正确：使用包结构，无需复制任何源文件
from metaspore.trainflows import DNNTrainFlow

# 旧架构（已废弃）：
# vim utils/trainflows/dnn_trainFlow.py  # 不再需要
```

---

## 🗂️ Workshop 实验说明

| 目录 | 任务 | 代表实验 |
|---|---|---|
| `ctr/wd_v*` | CTR Wide&Deep | wd_v5（当前基线） |
| `delay_feedback/defer/` | 延迟反馈 | defer_win_v2（AUC 0.8283） |
| `ltv/ltv_ziln_v*` | LTV 预估 | ltv_ziln_v2 |
| `mtl/mtl_mmoe_v*` | 多任务学习 | mtl_mmoe_v8 |
| `mdl/mdl_star_v*` | 多域学习 | mdl_star_v9 |
| `winrate/winrate_deepcensor_v*` | Winrate | winrate_deepcensor_v3 |

---

## 💡 调参建议

| 场景 | 建议 |
|---|---|
| 特征稀疏 | 增大 `ftrl_l1`，减小 `embedding_size` |
| 过拟合 | 增大 `net_dropout`，启用 `batch_norm` |
| 欠拟合 | 增大 `dnn_hidden_units`，增大 `embedding_size` |
| 训练慢 | 增大 `batch_size`，增大 `worker_count` |
| LTV 预估 | 使用 `wce_loss` 或 `ziln_loss` |
| DEFER 校准差 | 检查 `defer_delay_hours` 是否与数据归因窗口一致 |

### CTR 推荐基础配置

```yaml
embedding_size: 8
dnn_hidden_units: [256, 256, 128]
adam_learning_rate: 0.001
ftrl_l1: 1.0
ftrl_l2: 120.0
batch_size: 512
net_dropout: 0.5
batch_norm: true
```

---

## 🆕 零文件复制架构 (Zero-Copy Architecture)

### 优势
- **无需复制**: 新实验无需复制任何源代码文件
- **统一管理**: 所有 trainFlow 在 metaspore.trainflows 包中统一管理
- **易于维护**: 一次修改，所有项目受益
- **标准化**: 统一的导入接口和使用方式

### 使用方式
```python
# 旧方式（已废弃）
# cp -r utils/trainflows/* workshop/my_proj/src/
# from src.dnn_trainFlow import DNNTrainFlow

# 新方式（推荐）
from metaspore.trainflows import DNNTrainFlow
from metaspore.trainflows import BaseTrainFlow, MTLTrainFlow
```

---

## 📚 参考资料

- [MetaSpore 官方文档](https://github.com/meta-soul/MetaSpore)
- [Wide & Deep Learning (Google 2016)](https://arxiv.org/abs/1606.07792)
- [DeepFM (Huawei 2017)](https://arxiv.org/abs/1703.04247)
- [DCN V2 (Google 2020)](https://arxiv.org/abs/2008.13535)
- [DEFER: Delayed Feedback in CTR](https://arxiv.org/abs/1907.06558)