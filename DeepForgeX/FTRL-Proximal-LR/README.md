# FTRL-Proximal LR 训练框架

基于 FTRL-Proximal 优化算法的 Logistic Regression 训练框架，专为大规模 CTR/CVR 预估场景设计。

## 📦 项目结构

```
ftrl_maven-master/
├── src/main/java/com/mobvista/ftrl/
│   ├── exec/Main.java              # 训练入口
│   ├── train/LocalParaTrainer.java # 核心训练逻辑
│   ├── struct/                     # 数据结构
│   │   ├── Model.java              # 模型定义
│   │   ├── ModelText.java          # 模型读写
│   │   ├── ModelConfig.java        # 配置解析
│   │   ├── Sample.java             # 样本结构
│   │   └── FeatureInfo.java        # 特征信息
│   ├── datasource/                 # 数据源
│   │   └── SingleFeatureDataSource.java
│   ├── collector/                  # 指标收集
│   │   ├── AucCollector.java       # AUC 计算
│   │   └── LossCollector.java      # Loss 收集
│   ├── tools/                      # 工具类
│   │   ├── OnlineReplay.java       # 离线验证/回放
│   │   └── TextToBinaryConverter.java
│   └── linereader/                 # 数据读取
├── conf/                           # 示例配置
├── data/                           # 示例数据
├── tools/                          # 评估脚本
│   └── figure_auc_regression.py    # AUC 可视化
└── workshop/                       # 业务模型配置
    ├── ivr7/                       # IVR7 业务
    ├── ruf/                        # RUF 业务
    ├── duf_inner/                  # DUF 内部模型
    └── ...
```

## 🚀 快速开始

### 环境要求

- JDK 1.8+
- Maven 3.x
- Python 3.x (评估脚本)

### 编译

```bash
mvn clean package -DskipTests
```

生成 JAR 包：`target/ftrl_maven_walter-0.0.1-SNAPSHOT.jar`

### 运行训练

```bash
java -jar -Xmx80g target/ftrl_maven_walter-0.0.1-SNAPSHOT.jar \
    -i /path/to/train_data \
    -c /path/to/conf/ \
    -f f \
    -n experiment_name
```

**参数说明：**

| 参数 | 说明 | 必填 |
|------|------|:----:|
| `-i` | 训练数据路径（文件或目录） | ✅ |
| `-c` | 配置文件目录 | ✅ |
| `-f` | 数据格式，固定为 `f` | ✅ |
| `-n` | 实验名称 | ❌ |

---

## 📖 配置文件说明

### 1. `train.conf` — 训练主配置

```properties
# 特征采样率 (0-1)，用于加速训练
fir=0.3

# 特征频次阈值，低于此值的特征不参与训练
fft=3

# 正样本权重，用于处理样本不平衡
pos_weight=1

# 损失函数：cross_entropy | focal_loss | mse | wce | mae | msle | huber_loss
loss_func=cross_entropy

# Focal Loss 参数（仅 loss_func=focal_loss 时生效）
fl_alpha=0.25
fl_gamma=2.0

# 模型路径（增量训练时指定上次模型，首次训练设为 none 或不存在的路径）
model=./train_output/base

# 输出目录
output=./train_output

# FTRL 超参数配置文件
conf=./conf/para.conf

# 模型名称标识
name=my_model

# 标签列名
label=label

# 分组键（可选，用于分组统计）
keys=business_type
```

### 2. `para.conf` — FTRL 超参数

```
# 格式: name,alpha,beta,lambda1,lambda2
base,0.1,1,1,300
```

| 参数 | 说明 | 推荐范围 |
|------|------|----------|
| `name` | 参数组名称 | - |
| `alpha` | 学习率参数 | 0.05 ~ 1.0 |
| `beta` | 学习率参数 | 1.0 |
| `lambda1` | L1 正则化（稀疏性） | 0.1 ~ 10 |
| `lambda2` | L2 正则化（防过拟合） | 10 ~ 500 |

**FTRL 学习率公式：**
```
η = alpha / (beta + sqrt(sum_of_squared_gradients))
```

### 3. `column_name` — 特征列映射

```
0 campaignsetid
1 campaignid
2 adid
3 creativeid
4 country
5 city
...
```

每行格式：`列索引<空格>特征名`

### 4. `combine_schema` — 特征与特征交叉配置

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
bundle#country
bundle#campaignid
osv#make#campaignsetid
campaignid#make#osv#language
```

---

## 📊 数据格式

训练数据为文本文件，每行一个样本，字段用特定分隔符分隔：

```
特征值1<分隔符>特征值2<分隔符>...<分隔符>label
```

- 特征值按 `column_name` 定义的顺序排列
- `label`：分类任务为 0/1，回归任务为连续值

---

## 🔧 模型验证

使用 `OnlineReplay` 工具进行离线验证：

```bash
java -Xmx70g -cp target/ftrl_maven_walter-0.0.1-SNAPSHOT.jar \
    com.mobvista.ftrl.tools.OnlineReplay \
    -data /path/to/validation_data \
    -conf ./conf/ \
    -model ./train_output/base.filt \
    -out ./train_output/ \
    -name "val-2024-01-01"
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `-data` | 验证数据路径 |
| `-conf` | 配置目录 |
| `-model` | 模型文件路径 |
| `-out` | 输出目录 |
| `-name` | 验证标签名 |

---

## 📈 评估指标

### AUC 与校准度分析

```bash
python tools/figure_auc_regression.py \
    --input ./train_output/auc/ \
    --output ./figures/
```

支持的评估指标：
- **AUC** — 排序能力
- **PCOC** — Predicted/Observed Click 校准度
- **WRMSE** — 加权均方根误差
- **NWMAE** — 归一化加权平均绝对误差

---

## 🗂️ Workshop 使用示例

### 目录结构

每个业务模型目录包含：

```
workshop/ivr7/ivr7_outer_v1/
├── conf/
│   ├── train.conf        # 训练配置
│   ├── para.conf         # FTRL 参数
│   ├── column_name       # 特征列映射
│   └── combine_schema    # 特征交叉配置
├── run.sh                # 一键训练脚本
├── train_by_date.sh      # 按日期增量训练
├── val_by_model.sh       # 模型验证
├── score_kdd.py          # 评估脚本
└── train_output/         # 模型输出目录
```

### 训练流程

```bash
cd workshop/ivr7/ivr7_outer_v1

# 方式一：一键训练+验证
bash run.sh

# 方式二：按日期范围训练
bash train_by_date.sh 20241001 20241015

# 方式三：验证指定模型
bash val_by_model.sh 2024-10-14 2024-10-15
```

### 完整训练脚本示例 (train_by_date.sh)

```bash
#!/bin/bash
source ./lib_common.sh

start_date=$(format_date "$1")
end_date=$(format_date "$2")
current_date=${start_date}

while [[ "$current_date" < "$end_date" || "$current_date" == "$end_date" ]]; do
    echo "======== 开始训练 ${current_date} ========"
    
    local_data_path="../sample/sample_${current_date}"
    
    # 执行训练
    java -jar -Xmx80g ../../target/ftrl_maven_walter-0.0.1-SNAPSHOT.jar \
        -i "${local_data_path}" \
        -c "./conf" \
        -f "f" \
        -n "$current_date"
    
    # 保存当天模型快照
    cp "./train_output/base" "./train_output/base.${current_date}"
    
    current_date=$(date -d "${current_date} + 1 day" +"%Y-%m-%d")
done

# 过滤零权重特征
awk -F'\002' '$2!=0' "./train_output/base" | sort -t$'\002' -k2,2 -rg > "./train_output/base.filt"
```

---

## 📝 模型文件格式

### 文本格式 (分类任务)

```
特征名<\002>权重<\002>z值<\002>正样本数<\002>负样本数
```

### 文本格式 (回归任务)

```
特征名<\002>权重<\002>z值<\002>n值<\002>标签累计<\002>样本数
```

**注：** `<\002>` 为 ASCII 控制字符（Ctrl+B）

### 模型过滤

训练完成后，通常需要过滤零权重特征：

```bash
awk -F'\002' '$2!=0' train_output/base | sort -t$'\002' -k2,2 -rg > train_output/base.filt
```

---

## ⚙️ 支持的损失函数

| 损失函数 | 配置值 | 适用场景 | 预测公式 |
|----------|--------|----------|----------|
| Cross Entropy | `cross_entropy` | 二分类 CTR | `sigmoid(z)` |
| Focal Loss | `focal_loss` | 极度不平衡分类 | `sigmoid(z)` |
| Weighted CE | `wce` | 加权分类 | `exp(z)` |
| MSE | `mse` | 回归 | `z` |
| MAE | `mae` | 回归（鲁棒） | `z` |
| MSLE | `msle` | 大跨度回归 | `z` |
| Huber Loss | `huber_loss` | 回归（鲁棒） | `z` |

---

## 🔄 增量训练

FTRL 天然支持增量训练，只需指定上次模型路径：

```properties
# train.conf
model=./train_output/base.2024-10-14
```

训练时会加载已有模型参数，在新数据上继续更新。

---

## 💡 调参建议

| 场景 | 建议配置 |
|------|----------|
| 特征稀疏、需要特征选择 | 增大 `lambda1` (L1) |
| 防止过拟合 | 增大 `lambda2` (L2) |
| 正负样本不平衡 | 调整 `pos_weight` |
| 训练速度优先 | 降低 `fir` 采样率 |
| 过滤低频噪声特征 | 增大 `fft` 阈值 |

---

## 📚 参考资料

- [Ad Click Prediction: a View from the Trenches (Google 2013)](https://research.google/pubs/pub41159/)
- [FTRL-Proximal 算法详解](http://castellanzhang.github.io/2016/10/16/fm_ftrl_softmax/)
