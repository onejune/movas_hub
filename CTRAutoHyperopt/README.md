<p align="center">
  <img src="docs/logo.png" width="200"/>
</p>

<h1 align="center">CTR Auto HyperOpt</h1>

<p align="center">
  <strong>🚀 Automated Model Selection & Hyperparameter Optimization for CTR Prediction</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#supported-models">Models</a> •
  <a href="#benchmark">Benchmark</a> •
  <a href="#api">API</a>
</p>

---

## 🎯 Overview

**CTR Auto HyperOpt** is an automated machine learning framework for Click-Through Rate (CTR) prediction. It automatically searches for the best model architecture and hyperparameters from 50+ state-of-the-art models.

### Why CTR Auto HyperOpt?

| Challenge | Solution |
|-----------|----------|
| Too many CTR models to choose from | Automated model selection across 50+ models |
| Manual hyperparameter tuning is tedious | Optuna-based Bayesian optimization |
| Hard to compare ML vs DL models | Unified framework for tree models & deep learning |
| Feature engineering is complex | Built-in feature crossing & high-cardinality handling |

## ✨ Features

- 🔍 **Auto Model Selection**: Search across 50+ CTR models (DeepFM, xDeepFM, DCN, AutoInt, etc.)
- ⚡ **Auto Hyperparameter Tuning**: Optuna-based Bayesian optimization with pruning
- 🌲 **ML + DL Support**: XGBoost, LightGBM, CatBoost + Deep Learning models
- 🔗 **Auto Feature Engineering**: Automatic feature crossing & encoding
- 📊 **Comprehensive Metrics**: AUC, LogLoss, PCOC, and more
- 🎛️ **Flexible API**: Easy to use, highly customizable

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ctr_auto_hyperopt.git
cd ctr_auto_hyperopt

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0
tensorflow>=2.10
optuna>=3.0
flaml>=2.0
deepctr-torch>=0.2.9
mlgb>=0.8.0
lightgbm>=4.0
xgboost>=2.0
catboost>=1.2
scikit-learn>=1.0
pandas>=2.0
numpy>=1.20
```

## 🚀 Quick Start

### Basic Usage

```python
from ctr_auto_hyperopt import AutoCTR

# Initialize
auto_ctr = AutoCTR(
    task='binary',
    metric='auc',
    time_budget=600,  # 10 minutes
)

# Fit
auto_ctr.fit(X_train, y_train, X_val, y_val)

# Predict
y_pred = auto_ctr.predict(X_test)

# Get best model info
print(f"Best Model: {auto_ctr.best_model_name}")
print(f"Best AUC: {auto_ctr.best_score:.4f}")
print(f"Best Params: {auto_ctr.best_params}")
```

### Advanced Usage

```python
from ctr_auto_hyperopt import AutoCTR

# Custom model pool
auto_ctr = AutoCTR(
    task='binary',
    metric='auc',
    time_budget=1800,
    
    # Model selection
    include_models=['xDeepFM', 'AutoInt', 'DCN', 'FiBiNet', 'lgbm', 'xgboost'],
    exclude_models=['AFM'],  # Exclude slow models
    
    # Search space
    embed_dim_range=[8, 16, 32, 64],
    dnn_layers_range=[1, 4],
    dnn_units_range=[64, 128, 256, 512],
    
    # Optimization
    n_trials=100,
    pruning=True,
    n_jobs=4,
)

# Fit with feature config
auto_ctr.fit(
    X_train, y_train, X_val, y_val,
    sparse_features=['user_id', 'item_id', 'category'],
    dense_features=['price', 'age'],
    cross_features=[('user_id', 'category'), ('item_id', 'category')],
)
```

## 🏆 Supported Models

### Machine Learning Models

| Model | Library | Description |
|-------|---------|-------------|
| LightGBM | FLAML | Gradient boosting with leaf-wise growth |
| XGBoost | FLAML | Gradient boosting with level-wise growth |
| CatBoost | FLAML | Gradient boosting with categorical support |
| Random Forest | FLAML | Ensemble of decision trees |
| Extra Trees | FLAML | Extremely randomized trees |

### Deep Learning Models (50+)

#### Normal Models

| Model | Paper | Team | Year |
|-------|-------|------|------|
| LR | Predicting Clicks | Microsoft | 2007 |
| MLP/DNN | Neural Networks for Pattern Recognition | - | 1995 |
| DLRM | Deep Learning Recommendation Model | Meta | 2019 |
| MaskNet | Instance-Guided Mask | Weibo | 2021 |
| DCN/DCNv2 | Deep & Cross Network | Google | 2017/2020 |
| EDCN | Enhanced DCN | Huawei | 2021 |
| FM | Factorization Machines | Rendle | 2010 |
| FFM | Field-aware FM | NTU | 2016 |
| FwFM | Field-weighted FM | Yahoo | 2018 |
| AFM | Attentional FM | ZJU & NUS | 2017 |
| FNN | Deep Learning over Multi-field Data | UCL | 2016 |
| PNN | Product-based Neural Networks | SJTU | 2016 |
| ONN/NFFM | Operation-aware NN | NJU | 2019 |
| AFN | Adaptive Factorization Network | SJTU | 2020 |
| NFM | Neural FM | NUS | 2017 |
| WDL | Wide & Deep | Google | 2016 |
| DeepFM | FM + DNN | Huawei | 2017 |
| xDeepFM | Compressed Interaction Network | Microsoft | 2018 |
| FiBiNet | Bilinear Feature Interaction | Weibo | 2019 |
| AutoInt | Self-Attentive NN | PKU | 2019 |

#### Sequential Models

| Model | Paper | Team | Year |
|-------|-------|------|------|
| GRU4Rec | Session-based RNN | Telefonica | 2016 |
| Caser | Convolutional Sequence Embedding | SFU | 2018 |
| SASRec | Self-Attentive Sequential | UCSD | 2018 |
| BERT4Rec | BERT for Recommendation | Alibaba | 2019 |
| BST | Behavior Sequence Transformer | Alibaba | 2019 |
| DIN | Deep Interest Network | Alibaba | 2018 |
| DIEN | Deep Interest Evolution Network | Alibaba | 2018 |
| DSIN | Deep Session Interest Network | Alibaba | 2019 |

#### Multi-Task Models

| Model | Paper | Team | Year |
|-------|-------|------|------|
| SharedBottom | Multi-Task Learning Overview | - | 2017 |
| ESMM | Entire Space Multi-Task | Alibaba | 2018 |
| MMoE | Multi-gate Mixture-of-Experts | Google | 2018 |
| PLE | Progressive Layered Extraction | Tencent | 2020 |
| PEPNet | Parameter and Embedding Personalized | Kuaishou | 2023 |

## 📊 Benchmark

### Dataset: defer_sample (CTR Prediction)

| Rank | Model | Type | AUC | LogLoss | Search Time |
|------|-------|------|-----|---------|-------------|
| 🥇 1 | **xDeepFM** | DL | **0.7999** | 0.2220 | 2m 51s |
| 🥈 2 | AutoInt | DL | 0.7985 | 0.2225 | - |
| 🥉 3 | FiBiNet | DL | 0.7971 | 0.2230 | - |
| 4 | MLP (Optuna) | DL | 0.7968 | 0.2228 | 52s |
| 5 | XGBoost | ML | 0.7967 | 0.2226 | 60s |
| 6 | LightGBM | ML | 0.7932 | 0.2234 | 60s |
| 7 | Wide & Deep | DL | 0.7934 | 0.2237 | 50s |

*Benchmark on 10% sampled data (50K training, 24K test), 3 epochs per trial.*

## 🔧 API Reference

### AutoCTR

```python
class AutoCTR:
    def __init__(
        self,
        task: str = 'binary',              # 'binary' or 'regression'
        metric: str = 'auc',               # 'auc', 'logloss', 'rmse'
        time_budget: int = 600,            # Total search time in seconds
        n_trials: int = 50,                # Number of Optuna trials
        include_models: List[str] = None,  # Models to include
        exclude_models: List[str] = None,  # Models to exclude
        device: str = 'auto',              # 'cpu', 'cuda', 'auto'
        seed: int = 42,                    # Random seed
    ):
        ...
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame = None,
        y_val: np.ndarray = None,
        sparse_features: List[str] = None,
        dense_features: List[str] = None,
        cross_features: List[Tuple[str, str]] = None,
    ) -> 'AutoCTR':
        ...
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...
    
    @property
    def best_model_name(self) -> str:
        ...
    
    @property
    def best_score(self) -> float:
        ...
    
    @property
    def best_params(self) -> dict:
        ...
```

## 📁 Project Structure

```
ctr_auto_hyperopt/
├── README.md
├── requirements.txt
├── setup.py
├── ctr_auto_hyperopt/
│   ├── __init__.py
│   ├── auto_ctr.py          # Main AutoCTR class
│   ├── search/
│   │   ├── ml_search.py     # ML model search (FLAML)
│   │   ├── dl_search.py     # DL model search (Optuna)
│   │   └── space.py         # Search space definitions
│   ├── models/
│   │   ├── deepctr_models.py
│   │   └── mlgb_models.py
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── feature_eng.py
│   └── utils/
│       ├── metrics.py
│       └── callbacks.py
├── examples/
│   ├── quick_start.py
│   ├── custom_search.py
│   └── benchmark.py
└── tests/
    └── test_auto_ctr.py
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use CTR Auto HyperOpt in your research, please cite:

```bibtex
@software{ctr_auto_hyperopt,
  title = {CTR Auto HyperOpt: Automated Model Selection for CTR Prediction},
  year = {2026},
  url = {https://github.com/your-username/ctr_auto_hyperopt}
}
```

## 🙏 Acknowledgements

- [DeepCTR](https://github.com/shenweichen/DeepCTR) - Deep Learning CTR Models
- [MLGB](https://github.com/UlionTse/mlgb) - Machine Learning Great Boss
- [Optuna](https://github.com/optuna/optuna) - Hyperparameter Optimization
- [FLAML](https://github.com/microsoft/FLAML) - Fast AutoML Library

---

<p align="center">
  Made with ❤️ for the CTR prediction community
</p>
