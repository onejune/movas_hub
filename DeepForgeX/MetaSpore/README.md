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
│       └── loss_utils.py               # 损失函数注册表
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
    ├── trainflows/                     # TrainFlow 体系
    │   ├── base_trainFlow.py           # 基类（所有 TrainFlow 继承此类）
    │   ├── dnn_trainFlow.py            # CTR/CVR 单任务
    │   ├── defer_trainFlow.py          # 延迟反馈 DEFER
    │   ├── MTL_trainFlow.py            # 多任务学习
    │   ├── MDL_trainFlow.py            # 多域学习
    │   ├── ltv_trainFlow.py            # LTV 预估
    │   ├── winrate_trainFlow.py        # Winrate
    │   └── DELF_trainFlow.py           # DELF
    ├── tools/                          # 工具类
    │   ├── feishu_notifier.py          # 飞书通知
    │   ├── movas_logger.py             # 日志
    │   └── metrics_eval.py             # 评估指标
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

### 2. 复制已有实验作为模板

```bash
# 以 wd_v5 为模板新建 wd_v6
cp -r workshop/ctr/wd_v5 workshop/ctr/wd_v6
cd workshop/ctr/wd_v6
```

### 3. 实验目录标准结构

```
my_experiment/
├── conf/
│   ├── config.yaml         # 主配置文件（必须）
│   ├── combine_schema      # 特征配置（必须）
│   └── column_name         # 列名映射（可选）
├── run_train.sh            # 训练启动脚本（必须）
├── validation.sh           # 验证脚本（可选）
├── readme                  # 实验说明（必须，飞书通知会读取）
├── src/                    # 训练代码（由 init_env 从 utils/ 复制）
├── log/                    # 日志输出（gitignore）
└── output/                 # 模型输出（gitignore）
```

### 4. 编写 `readme` 文件（必须）

`readme` 文件（注意：不是 `README.md`）会被 `dnn_trainFlow.py` 自动读取，并在训练完成后发送到飞书群。

```
实验名称: wd_v6

实验变量:
- 新增 xxx 特征
- 调整 embedding_size: 8 -> 12

训练配置:
- 训练数据: 2026-02-11 ~ 2026-02-17 (7天)
- 验证数据: 2026-02-18
- 数据源: ivr_sample_v7
```

### 5. 配置 `run_train.sh`

```bash
#!/bin/bash

source /mnt/workspace/walter.wan/git_project/github_onejune/movas_hub/DeepForgeX/utils/scripts/dnn_lib_common.sh

# 普通 DNN 训练
TRAINER_SCRIPT_PATH="./src/dnn_trainFlow.py"
env_check
model_train ./conf/config.yaml business_type

# DEFER 延迟反馈训练
# TRAINER_SCRIPT_PATH="./src/defer_trainFlow.py"
# env_check
# model_train ./conf/config.yaml business_type
```

> ⚠️ `env_check` 会调用 `init_env()`，将 `utils/` 下的文件复制到 `src/`。**修改代码请改 `utils/` 源文件，不要改 `src/`**。

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

## 🔧 TrainFlow 体系

所有 TrainFlow 继承 `BaseTrainFlow`，结构如下：

```
BaseTrainFlow (base_trainFlow.py)
├── DNNModelTrainFlow (dnn_trainFlow.py)   — CTR/CVR 单任务，13种模型
├── DeferTrainFlow (defer_trainFlow.py)    — 延迟反馈 DEFER
├── MTLModelTrainFlow (MTL_trainFlow.py)   — 多任务学习
├── MDLModelTrainFlow (MDL_trainFlow.py)   — 多域学习
├── LtvModelTrainFlow (ltv_trainFlow.py)   — LTV 预估
├── WinrateModelTrainFlow (winrate_trainFlow.py) — Winrate
├── DELFModelTrainFlow (DELF_trainFlow.py) — DELF
└── MetaCPLTrainFlow (MetaCPL_trainFlow.py)
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
from base_trainFlow import BaseTrainFlow

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
# dnn_trainFlow.py 自动处理
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
cp -r workshop/delay_feedback/defer/defer_win_v2 workshop/delay_feedback/defer/defer_win_v3
cd workshop/delay_feedback/defer/defer_win_v3

# 编辑配置
vim conf/config.yaml

# 编辑实验说明（飞书通知会读取）
vim readme

# 启动训练
bash run_train.sh
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
export PYTHONPATH=/mnt/workspace/walter.wan/git_project/movas_hub/DeepForgeX/MetaSpore/python

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

### 7. 修改代码改 utils/，不要改 src/

`init_env()` 每次训练前会将 `utils/` 文件复制到 `src/`，覆盖本地修改：

```bash
# ✅ 正确：修改 utils/ 源文件
vim utils/trainflows/defer_trainFlow.py

# ❌ 错误：修改 src/ 会被下次 init_env 覆盖
vim workshop/my_exp/src/defer_trainFlow.py
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

## 📚 参考资料

- [MetaSpore 官方文档](https://github.com/meta-soul/MetaSpore)
- [Wide & Deep Learning (Google 2016)](https://arxiv.org/abs/1606.07792)
- [DeepFM (Huawei 2017)](https://arxiv.org/abs/1703.04247)
- [DCN V2 (Google 2020)](https://arxiv.org/abs/2008.13535)
- [DEFER: Delayed Feedback in CTR](https://arxiv.org/abs/1907.06558)
