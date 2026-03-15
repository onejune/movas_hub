"""
CTR 深度学习模型 - DeepCTR + Optuna 自动搜索
=============================================

使用 DeepCTR-Torch 的多种 CTR 模型 + Optuna 超参寻优:
- DeepFM, DCN, xDeepFM, AutoInt, NFM, AFM, PNN, WDL, FiBiNET 等
- 自动搜索模型类型 + 超参数
"""

import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

import optuna
from optuna.trial import Trial

# DeepCTR
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import (
    DeepFM, DCN, DCNMix, xDeepFM, AutoInt, 
    NFM, AFM, PNN, WDL, FiBiNET, AFN
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 配置
DATA_PATH = '/mnt/data/oss_wanjun/pai_work/defer_sample_parquet/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FEATURE_COLS = [
    'business_type', 'offerid', 'country', 'bundle',
    'adx', 'make', 'model', 'demand_pkgname', 'campaignid'
]

CROSS_FEATURES = [
    ('country', 'adx'),
    ('country', 'business_type'),
    ('make', 'model'),
    ('adx', 'business_type'),
]

# 可用模型
AVAILABLE_MODELS = {
    'DeepFM': DeepFM,
    'DCN': DCN,
    'DCNMix': DCNMix,
    'xDeepFM': xDeepFM,
    'AutoInt': AutoInt,
    'NFM': NFM,
    'AFM': AFM,
    'PNN': PNN,
    'WDL': WDL,
    'FiBiNET': FiBiNET,
    'AFN': AFN,
}


# ============================================================================
# 数据加载
# ============================================================================

_DATA_CACHE = {}

def load_data(n_train_days=2, n_test_days=1, sample_frac=0.1):
    """加载并预处理数据"""
    cache_key = (n_train_days, n_test_days, sample_frac)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]
    
    date_dirs = sorted([d for d in os.listdir(DATA_PATH) if d.startswith('sample_date=')])
    
    train_df = pd.concat([
        pd.read_parquet(os.path.join(DATA_PATH, d)) 
        for d in date_dirs[:n_train_days]
    ], ignore_index=True)
    test_df = pd.read_parquet(os.path.join(DATA_PATH, date_dirs[n_train_days]))
    
    train_df = train_df.sample(frac=sample_frac, random_state=42)
    test_df = test_df.sample(frac=sample_frac, random_state=42)
    
    # 特征工程
    for col1, col2 in CROSS_FEATURES:
        train_df[f'{col1}_x_{col2}'] = train_df[col1].astype(str) + '_' + train_df[col2].astype(str)
        test_df[f'{col1}_x_{col2}'] = test_df[col1].astype(str) + '_' + test_df[col2].astype(str)
    
    feature_cols = FEATURE_COLS + [f'{c1}_x_{c2}' for c1, c2 in CROSS_FEATURES]
    
    # 高基数处理 + Label 编码
    encoders = {}
    for col in feature_cols:
        # 高基数处理
        vc = train_df[col].value_counts()
        valid = set(vc[vc >= 50].index)
        train_df[col] = train_df[col].apply(lambda x: x if x in valid else 'OTHER')
        test_df[col] = test_df[col].apply(lambda x: x if x in valid else 'OTHER')
        
        # Label 编码
        le = LabelEncoder()
        all_vals = pd.concat([train_df[col], test_df[col]]).astype(str).unique()
        le.fit(all_vals)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        encoders[col] = le
    
    # 构建 DeepCTR 特征列
    sparse_features = feature_cols
    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=train_df[feat].max() + 1, embedding_dim=16)
        for feat in sparse_features
    ]
    
    feature_names = get_feature_names(fixlen_feature_columns)
    
    # 准备数据
    train_model_input = {name: train_df[name].values for name in feature_names}
    test_model_input = {name: test_df[name].values for name in feature_names}
    
    y_train = train_df['label'].values.astype(np.float32)
    y_test = test_df['label'].values.astype(np.float32)
    
    result = {
        'train_input': train_model_input,
        'test_input': test_model_input,
        'y_train': y_train,
        'y_test': y_test,
        'feature_columns': fixlen_feature_columns,
        'feature_names': feature_names,
    }
    
    _DATA_CACHE[cache_key] = result
    return result


# ============================================================================
# Optuna 目标函数
# ============================================================================

def create_model(model_name: str, feature_columns, trial: Trial):
    """根据模型名称和 trial 创建模型"""
    
    # 通用超参
    embed_dim = trial.suggest_categorical('embed_dim', [8, 16, 32])
    
    # 更新 feature_columns 的 embedding_dim
    updated_feature_columns = [
        SparseFeat(feat.name, vocabulary_size=feat.vocabulary_size, embedding_dim=embed_dim)
        for feat in feature_columns
    ]
    
    # DNN 配置
    n_layers = trial.suggest_int('dnn_n_layers', 1, 4)
    dnn_hidden_units = tuple([
        trial.suggest_categorical(f'dnn_hidden_{i}', [64, 128, 256, 512])
        for i in range(n_layers)
    ])
    dnn_dropout = trial.suggest_float('dnn_dropout', 0.0, 0.5)
    
    # 模型特定参数
    model_class = AVAILABLE_MODELS[model_name]
    
    if model_name == 'DeepFM':
        model = model_class(
            updated_feature_columns, updated_feature_columns,
            dnn_hidden_units=dnn_hidden_units,
            dnn_dropout=dnn_dropout,
            device=DEVICE
        )
    
    elif model_name in ['DCN', 'DCNMix']:
        cross_num = trial.suggest_int('cross_num', 1, 4)
        model = model_class(
            updated_feature_columns, updated_feature_columns,
            dnn_hidden_units=dnn_hidden_units,
            cross_num=cross_num,
            dnn_dropout=dnn_dropout,
            device=DEVICE
        )
    
    elif model_name == 'xDeepFM':
        cin_layer_size = trial.suggest_categorical('cin_layer_size', [(64,), (128,), (64, 64), (128, 128)])
        model = model_class(
            updated_feature_columns, updated_feature_columns,
            dnn_hidden_units=dnn_hidden_units,
            cin_layer_size=cin_layer_size,
            dnn_dropout=dnn_dropout,
            device=DEVICE
        )
    
    elif model_name == 'AutoInt':
        att_layer_num = trial.suggest_int('att_layer_num', 1, 4)
        att_head_num = trial.suggest_categorical('att_head_num', [1, 2, 4])
        model = model_class(
            updated_feature_columns, updated_feature_columns,
            dnn_hidden_units=dnn_hidden_units,
            att_layer_num=att_layer_num,
            att_head_num=att_head_num,
            dnn_dropout=dnn_dropout,
            device=DEVICE
        )
    
    elif model_name == 'NFM':
        model = model_class(
            updated_feature_columns, updated_feature_columns,
            dnn_hidden_units=dnn_hidden_units,
            dnn_dropout=dnn_dropout,
            device=DEVICE
        )
    
    elif model_name == 'AFM':
        attention_factor = trial.suggest_categorical('attention_factor', [4, 8, 16])
        model = model_class(
            updated_feature_columns, updated_feature_columns,
            attention_factor=attention_factor,
            device=DEVICE
        )
    
    elif model_name == 'PNN':
        use_inner = trial.suggest_categorical('use_inner', [True, False])
        use_outter = trial.suggest_categorical('use_outter', [True, False])
        model = model_class(
            updated_feature_columns, updated_feature_columns,
            dnn_hidden_units=dnn_hidden_units,
            use_inner=use_inner,
            use_outter=use_outter,
            dnn_dropout=dnn_dropout,
            device=DEVICE
        )
    
    elif model_name == 'WDL':
        model = model_class(
            updated_feature_columns, updated_feature_columns,
            dnn_hidden_units=dnn_hidden_units,
            dnn_dropout=dnn_dropout,
            device=DEVICE
        )
    
    elif model_name == 'FiBiNET':
        bilinear_type = trial.suggest_categorical('bilinear_type', ['all', 'each', 'interaction'])
        model = model_class(
            updated_feature_columns, updated_feature_columns,
            dnn_hidden_units=dnn_hidden_units,
            bilinear_type=bilinear_type,
            dnn_dropout=dnn_dropout,
            device=DEVICE
        )
    
    elif model_name == 'AFN':
        afn_dnn_hidden_units = trial.suggest_categorical('afn_dnn_hidden_units', [(64,), (128,), (256,)])
        model = model_class(
            updated_feature_columns, updated_feature_columns,
            dnn_hidden_units=dnn_hidden_units,
            afn_dnn_hidden_units=afn_dnn_hidden_units,
            dnn_dropout=dnn_dropout,
            device=DEVICE
        )
    
    else:
        # 默认配置
        model = model_class(
            updated_feature_columns, updated_feature_columns,
            dnn_hidden_units=dnn_hidden_units,
            dnn_dropout=dnn_dropout,
            device=DEVICE
        )
    
    return model


def objective(trial: Trial, data: dict, model_names: List[str], n_epochs: int = 3) -> float:
    """Optuna 目标函数"""
    
    # 选择模型
    model_name = trial.suggest_categorical('model', model_names)
    
    # 训练超参
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])
    
    try:
        # 创建模型
        model = create_model(model_name, data['feature_columns'], trial)
        
        # 编译
        model.compile(
            optimizer=torch.optim.Adam(model.parameters(), lr=lr),
            loss='binary_crossentropy',
            metrics=['binary_crossentropy', 'auc']
        )
        
        # 训练
        model.fit(
            data['train_input'],
            data['y_train'],
            batch_size=batch_size,
            epochs=n_epochs,
            verbose=0,
            validation_data=(data['test_input'], data['y_test'])
        )
        
        # 预测
        y_pred = model.predict(data['test_input'], batch_size=batch_size)
        auc = roc_auc_score(data['y_test'], y_pred)
        
        return auc
    
    except Exception as e:
        print(f"  Trial failed ({model_name}): {e}")
        return 0.5  # 失败返回随机猜测的 AUC


# ============================================================================
# 主函数
# ============================================================================

def run_search(
    model_names: List[str] = None,
    n_trials: int = 30,
    n_epochs: int = 3,
    timeout: int = 600
):
    """运行搜索"""
    if model_names is None:
        model_names = list(AVAILABLE_MODELS.keys())
    
    print("="*70)
    print("DeepCTR + Optuna 模型搜索")
    print("="*70)
    print(f"开始时间: {datetime.now()}")
    print(f"设备: {DEVICE}")
    print(f"候选模型: {model_names}")
    print(f"Trials: {n_trials}, Epochs: {n_epochs}, Timeout: {timeout}s")
    
    # 加载数据
    print("\n加载数据...")
    data = load_data()
    print(f"训练集: {len(data['y_train']):,}, 测试集: {len(data['y_test']):,}")
    print(f"特征数: {len(data['feature_names'])}")
    
    # 创建 study
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # 运行搜索
    print("\n开始搜索...")
    study.optimize(
        lambda trial: objective(trial, data, model_names, n_epochs),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        catch=(Exception,)
    )
    
    # 结果
    print("\n" + "="*70)
    print("搜索完成!")
    print("="*70)
    print(f"最优 AUC: {study.best_value:.6f}")
    print(f"最优模型: {study.best_params.get('model', 'N/A')}")
    print(f"\n最优参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 各模型最优结果
    print("\n" + "-"*50)
    print("各模型最优 AUC:")
    print("-"*50)
    
    model_best = {}
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            model = trial.params.get('model', 'unknown')
            if model not in model_best or trial.value > model_best[model]:
                model_best[model] = trial.value
    
    for model, auc in sorted(model_best.items(), key=lambda x: -x[1]):
        print(f"  {model:15s}: {auc:.6f}")
    
    print(f"\n完成时间: {datetime.now()}")
    
    return study


def main():
    # 选择要搜索的模型（去掉一些可能有问题的）
    model_names = [
        'DeepFM', 'DCN', 'xDeepFM', 'AutoInt', 
        'NFM', 'PNN', 'WDL', 'FiBiNET'
    ]
    
    study = run_search(
        model_names=model_names,
        n_trials=40,
        n_epochs=3,
        timeout=600  # 10 分钟
    )
    
    return study


if __name__ == '__main__':
    study = main()
