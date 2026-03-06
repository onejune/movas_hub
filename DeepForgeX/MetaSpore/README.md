# MetaSpore DNN 训练框架

基于 PyTorch + PySpark 的分布式深度学习 CTR/CVR 预估训练框架，支持 Wide&Deep、DeepFM、DCN 等主流推荐模型。

## ✨ 特性

- **分布式训练** — 基于 PySpark 的分布式数据处理和模型训练
- **多种模型架构** — 支持 WideDeep、DeepFM、DCN、FFM、PNN 等
- **灵活的特征工程** — 通过 combine_schema 配置特征和特征交叉
- **增量训练** — 支持加载已有模型继续训练
- **多种损失函数** — 支持 CrossEntropy、WCE、MSE、ZILN 等
- **飞书通知** — 训练完成自动发送飞书通知

## 📦 项目结构

```
MetaSpore/
├── python/
│   └── metaspore/
│       ├── algos/                  # 模型算法库
│       │   ├── widedeep_net.py     # Wide&Deep
│       │   ├── deepfm_net.py       # DeepFM
│       │   ├── dcn_net.py          # DCN (Deep & Cross Network)
│       │   ├── dcn_v2_net.py       # DCN V2
│       │   ├── ffm_net.py          # FFM
│       │   ├── pnn_net.py          # PNN
│       │   ├── xdeepfm_net.py      # xDeepFM
│       │   ├── autoint_net.py      # AutoInt
│       │   ├── ziln_model.py       # ZILN (LTV 预估)
│       │   ├── deep_censored_model.py  # Deep Censored Model
│       │   ├── layers.py           # 通用网络层
│       │   ├── multi_task/         # 多任务学习
│       │   └── multi_domain/       # 多域学习
│       └── ...
├── workshop/                       # 实验配置
│   ├── wd_v1/                      # Wide&Deep 实验
│   ├── dfm_v1/                     # DeepFM 实验
│   ├── dcn_v1/                     # DCN 实验
│   ├── ltv_wce_v1/                 # LTV WCE 损失实验
│   ├── ltv_ziln_v1/                # LTV ZILN 损失实验
│   ├── winrate_deepcensor_v1/      # Winrate 预估实验
│   └── ...
├── demo/                           # 官方 Demo
├── examples/                       # 示例代码
└── utils/                          # 工具脚本
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.x / 2.x
- PySpark 3.x
- MetaSpore (美团开源框架)

### 安装依赖

```bash
pip install metaspore pyspark torch numpy pandas pyyaml tabulate
```

---

## 📖 使用指南

### 训练流程

```bash
cd workshop/wd_v1

# 训练模型
bash run_train.sh

# 验证模型
bash validation.sh <model_date> <validation_date> <eval_keys>
```

### 目录结构

每个实验目录包含：

```
workshop/wd_v1/
├── conf/
│   ├── widedeep.yaml         # 主配置文件
│   ├── combine_schema        # 特征交叉配置
│   └── column_name           # 特征列映射（可选）
├── ms_dnn_wd.py              # 训练主程序
├── metrics_eval.py           # 评估指标
├── movas_logger.py           # 日志工具
├── feishu_notifier.py        # 飞书通知
├── run_train.sh              # 训练脚本
├── validation.sh             # 验证脚本
├── readme                    # 实验说明
└── output/                   # 模型输出目录
```

---

## ⚙️ 配置文件说明

### `conf/widedeep.yaml` — 主配置

```yaml
## Spark 配置
app_name: 'dnn_ivr7_v1'
local_spark: local              # local 或 yarn
worker_count: 10                # Worker 数量
server_count: 2                 # PS Server 数量
batch_size: 256                 # 批大小
worker_memory: '10G'            # Worker 内存
server_memory: '10G'            # Server 内存
coordinator_memory: '10G'       # Coordinator 内存

## 数据集配置
train_path_prefix: /path/to/parquet/data
train_start_date: 2025-01-01
train_end_date: 2025-01-07
validation_date: 2025-01-08
movas_log_output_path: ./log/movas.log

## 模型配置
model_type: WideDeep            # 模型类型
combine_schema_path: ./conf/combine_schema
wide_combine_schema_path: ./conf/combine_schema
model_out_base_path: ./output/
model_in_path: null             # 增量训练时指定
experiment_name: my_experiment
input_label_column_index: 113   # 标签列索引

## 超参数
embedding_size: 12              # Embedding 维度
dnn_hidden_units: [512, 256, 64]  # DNN 隐层
adam_learning_rate: 0.00001     # Adam 学习率
ftrl_alpha: 0.005               # FTRL alpha
ftrl_beta: 0.05                 # FTRL beta
ftrl_l1: 1.0                    # L1 正则化
ftrl_l2: 10.0                   # L2 正则化

## 可选配置
use_wide: True                  # 是否使用 Wide 部分
batch_norm: False               # 是否使用 BatchNorm
net_dropout: 0.5                # Dropout 比率
loss_func: 'cross_entropy'      # 损失函数
```

### 支持的模型类型

| model_type | 说明 |
|------------|------|
| `WideDeep` | Wide & Deep Learning |
| `DeepFM` | Deep Factorization Machine |
| `DCN` | Deep & Cross Network |
| `FFM` | Field-aware FM |
| `PNN` | Product-based NN |
| `xDeepFM` | eXtreme Deep FM |
| `AutoInt` | AutoInt |

### 支持的损失函数

| loss_func | 说明 | 适用场景 |
|-----------|------|----------|
| `cross_entropy` | 交叉熵 | 二分类 CTR |
| `wce_loss` | 加权交叉熵 | LTV 预估 |
| `mse_loss` | 均方误差 | 回归任务 |
| `ziln_loss` | Zero-Inflated LogNormal | LTV 预估 |

### `conf/combine_schema` — 特征配置

```
# 单特征
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

# 序列特征交叉
user_click_seq#item_id
```

---

## 🧠 支持的模型架构

### Wide & Deep

```
Wide 部分: LR + Embedding + FTRL 优化
Deep 部分: Embedding + MLP + Adam 优化
```

### DeepFM

```
FM 部分: 二阶特征交互
Deep 部分: Embedding + MLP
```

### DCN (Deep & Cross Network)

```
Cross 部分: 显式特征交叉
Deep 部分: Embedding + MLP
```

---

## 📊 评估指标

训练和验证过程会计算以下指标：

| 指标 | 说明 |
|------|------|
| AUC | Area Under ROC Curve |
| PCOC | Predicted/Observed Click |
| LogLoss | 对数损失 |
| AUPR | Area Under PR Curve |

### 评估输出示例

```
+---------------+--------+--------+----------+
| demand_pkgname|    AUC |   PCOC | LogLoss  |
+---------------+--------+--------+----------+
| com.app.game  | 0.7823 | 1.0234 | 0.4521   |
| com.app.shop  | 0.8012 | 0.9876 | 0.3987   |
+---------------+--------+--------+----------+
```

---

## 🗂️ Workshop 实验说明

| 目录 | 模型 | 说明 |
|------|------|------|
| `wd_v1` ~ `wd_v5` | WideDeep | Wide&Deep 基础实验 |
| `dfm_v1` ~ `dfm_v3` | DeepFM | DeepFM 实验 |
| `dcn_v1` ~ `dcn_v2` | DCN | Deep & Cross Network |
| `ltv_mse_v1` | WideDeep + MSE | LTV MSE 损失 |
| `ltv_wce_v1` | WideDeep + WCE | LTV 加权交叉熵 |
| `ltv_ziln_v1` | ZILN | LTV Zero-Inflated LogNormal |
| `winrate_deepcensor_v1` | DeepCensor | Winrate 预估 |
| `wd_sample_exp_v1` | WideDeep | 采样实验 |
| `wd_diff_comb_v1` | WideDeep | 不同特征组合实验 |

---

## 🔧 高级用法

### 增量训练

在配置文件中指定 `model_in_path`：

```yaml
model_in_path: ./output/model_2025-01-07
model_out_base_path: ./output/
```

### 自定义模型

继承 `torch.nn.Module` 并实现 `forward` 方法：

```python
import torch
import metaspore as ms

class MyModel(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_units, combine_schema_path):
        super().__init__()
        self.sparse = ms.EmbeddingSumConcat(
            embedding_dim, 
            combine_schema_path=combine_schema_path
        )
        self.dnn = MLPLayer(
            input_dim=self.sparse.feature_count * embedding_dim,
            hidden_units=hidden_units,
            output_dim=1
        )
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        emb = self.sparse(x)
        out = self.dnn(emb)
        return self.sigmoid(out)
```

### 飞书通知

训练完成后自动发送飞书通知，需配置 webhook：

```python
# feishu_notifier.py
WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/xxx"
```

---

## 💡 调参建议

| 场景 | 建议配置 |
|------|----------|
| 特征稀疏 | 增大 `ftrl_l1`，减小 `embedding_size` |
| 过拟合 | 增大 `net_dropout`，启用 `batch_norm` |
| 欠拟合 | 增大 `dnn_hidden_units`，增大 `embedding_size` |
| 训练慢 | 增大 `batch_size`，增大 `worker_count` |
| LTV 预估 | 使用 `wce_loss` 或 `ziln_loss` |

### 推荐参数组合

```yaml
# CTR 预估基础配置
embedding_size: 12
dnn_hidden_units: [512, 256, 64]
adam_learning_rate: 0.00001
ftrl_l1: 1.0
ftrl_l2: 10.0
batch_size: 256
net_dropout: 0.3
```

---

## 📚 参考资料

- [MetaSpore 官方文档](https://github.com/meta-soul/MetaSpore)
- [Wide & Deep Learning (Google 2016)](https://arxiv.org/abs/1606.07792)
- [DeepFM (Huawei 2017)](https://arxiv.org/abs/1703.04247)
- [DCN (Google 2017)](https://arxiv.org/abs/1708.05123)
- [DCN V2 (Google 2020)](https://arxiv.org/abs/2008.13535)
