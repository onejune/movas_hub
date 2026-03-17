"""
Top Models Hyperparameter Optimization
======================================

对 DeepFM 和 FiBiNET 做超参搜索，找到最优配置。
基于 DeepCTR-Torch 框架。
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime

import optuna
from optuna.trial import Trial
import torch
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THIRD_PARTY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'third_party')
sys.path.insert(0, os.path.join(THIRD_PARTY, 'DeepCTR-Torch'))

from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM, FiBiNET

# 飞书通知
sys.path.append('/mnt/workspace/walter.wan/utils')
from feishu_notifier import FeishuNotifier


DATA_PATH = '/mnt/data/oss_wanjun/pai_work/defer_sample_by_train_date/'
SPARSE_FEATURES = ['business_type', 'offerid', 'country', 'bundle', 'adx', 'make', 'model', 'demand_pkgname', 'campaignid']
CROSS_FEATURES = [('country', 'adx'), ('make', 'model')]
HIGH_CARD_THRESHOLD = 50


def calc_pcoc(y_true, y_pred):
    """PCOC = mean(pred) / mean(true)"""
    return np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) > 0 else float('inf')


def load_data(train_days=30, test_days=3, sample_ratio=1.0):
    """加载数据
    
    Args:
        sample_ratio: 采样比例，用于加速超参搜索
    """
    date_dirs = sorted([d for d in os.listdir(DATA_PATH) if d.startswith('sample_date=')])
    
    train_dirs = date_dirs[:train_days]
    test_dirs = date_dirs[train_days:train_days + test_days]
    
    print(f"训练集: {train_days} 天 ({train_dirs[0]} ~ {train_dirs[-1]})")
    print(f"测试集: {test_days} 天 ({test_dirs[0]} ~ {test_dirs[-1]})")
    
    # 读取 CSV 文件
    train_dfs = []
    for d in train_dirs:
        dir_path = os.path.join(DATA_PATH, d)
        for f in os.listdir(dir_path):
            if f.endswith('.csv'):
                train_dfs.append(pd.read_csv(os.path.join(dir_path, f)))
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    test_dfs = []
    for d in test_dirs:
        dir_path = os.path.join(DATA_PATH, d)
        for f in os.listdir(dir_path):
            if f.endswith('.csv'):
                test_dfs.append(pd.read_csv(os.path.join(dir_path, f)))
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # 采样
    if sample_ratio < 1.0:
        train_df = train_df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
        test_df = test_df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
        print(f"采样比例: {sample_ratio*100:.0f}%")
    
    return train_df, test_df


def preprocess(train_df, test_df, embed_dim=8):
    """特征预处理"""
    sparse_features = SPARSE_FEATURES.copy()
    
    # 交叉特征
    for col1, col2 in CROSS_FEATURES:
        cross_name = f'{col1}_x_{col2}'
        train_df[cross_name] = train_df[col1].astype(str) + '_' + train_df[col2].astype(str)
        test_df[cross_name] = test_df[col1].astype(str) + '_' + test_df[col2].astype(str)
        sparse_features.append(cross_name)
    
    # 高基数处理
    for col in sparse_features:
        vc = train_df[col].value_counts()
        valid = set(vc[vc >= HIGH_CARD_THRESHOLD].index)
        train_df[col] = train_df[col].apply(lambda x: x if x in valid else 'OTHER')
        test_df[col] = test_df[col].apply(lambda x: x if x in valid else 'OTHER')
    
    # Label Encoding
    for col in sparse_features:
        le = LabelEncoder()
        all_vals = pd.concat([train_df[col], test_df[col]]).astype(str).unique()
        le.fit(all_vals)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
    
    # 构建特征列
    feature_columns = []
    for col in sparse_features:
        vocab_size = int(max(train_df[col].max(), test_df[col].max())) + 2
        feature_columns.append(SparseFeat(col, vocabulary_size=vocab_size, embedding_dim=embed_dim))
    
    # 构建输入
    train_input = {col: train_df[col].values for col in sparse_features}
    test_input = {col: test_df[col].values for col in sparse_features}
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    return feature_columns, train_input, y_train, test_input, y_test


def train_model(model, train_input, y_train, test_input, y_test, epochs, batch_size, lr):
    """训练并评估模型"""
    model.compile(
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        loss='binary_crossentropy',
        metrics=['binary_crossentropy', 'auc'],
    )
    
    history = model.fit(
        train_input, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
    )
    
    y_pred = model.predict(test_input, batch_size=batch_size * 2)
    
    auc = roc_auc_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred)
    pcoc = calc_pcoc(y_test, y_pred)
    
    return auc, logloss, pcoc


def run_hyperopt(model_name, train_df, test_df, n_trials=30, n_epochs=2, time_budget=1800):
    """对单个模型做超参搜索"""
    
    print(f"\n{'='*60}")
    print(f"超参搜索: {model_name}")
    print(f"Trials: {n_trials}, Epochs/trial: {n_epochs}, Time budget: {time_budget}s")
    print(f"{'='*60}")
    
    best_result = {'auc': 0, 'params': None}
    
    def objective(trial: Trial) -> float:
        nonlocal best_result
        
        try:
            # 超参
            embed_dim = trial.suggest_categorical('embed_dim', [8, 16, 32])
            dnn_hidden_units = trial.suggest_categorical('dnn_hidden_units', [
                '128,64',
                '256,128',
                '256,128,64',
                '512,256,128',
            ])
            dnn_hidden_units = tuple(int(x) for x in dnn_hidden_units.split(','))
            
            dnn_dropout = trial.suggest_float('dnn_dropout', 0.0, 0.5, step=0.1)
            lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])
            l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
            
            # 预处理（每次重新做，因为 embed_dim 可能不同）
            feature_columns, train_input, y_train, test_input, y_test = preprocess(
                train_df.copy(), test_df.copy(), embed_dim=embed_dim
            )
            
            # 创建模型
            if model_name == 'DeepFM':
                model = DeepFM(
                    linear_feature_columns=feature_columns,
                    dnn_feature_columns=feature_columns,
                    dnn_hidden_units=dnn_hidden_units,
                    dnn_dropout=dnn_dropout,
                    l2_reg_embedding=l2_reg,
                    l2_reg_dnn=l2_reg,
                    device='cpu',
                )
            elif model_name == 'FiBiNET':
                bilinear_type = trial.suggest_categorical('bilinear_type', ['all', 'each', 'interaction'])
                
                model = FiBiNET(
                    linear_feature_columns=feature_columns,
                    dnn_feature_columns=feature_columns,
                    dnn_hidden_units=dnn_hidden_units,
                    dnn_dropout=dnn_dropout,
                    l2_reg_embedding=l2_reg,
                    l2_reg_dnn=l2_reg,
                    bilinear_type=bilinear_type,
                    device='cpu',
                )
            
            # 训练
            auc, logloss, pcoc = train_model(
                model, train_input, y_train, test_input, y_test,
                epochs=n_epochs, batch_size=batch_size, lr=lr
            )
            
            if auc > best_result['auc']:
                best_result = {
                    'auc': auc,
                    'logloss': logloss,
                    'pcoc': pcoc,
                    'params': trial.params.copy()
                }
                print(f"  ★ New best: AUC={auc:.6f}, LogLoss={logloss:.6f}, PCOC={pcoc:.4f}")
            
            return auc
            
        except Exception as e:
            print(f"  Trial failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.5
    
    # 运行搜索
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=time_budget,
        show_progress_bar=True,
    )
    
    return best_result, study


def main():
    print("="*60)
    print("CTR 模型超参搜索")
    print("="*60)
    print(f"开始时间: {datetime.now()}")
    
    # 配置
    MODELS = ['DeepFM', 'FiBiNET']
    N_TRIALS = 30           # 每个模型30次试验
    N_EPOCHS = 2            # 每次试验2轮
    TIME_BUDGET = 1800      # 每个模型30分钟
    SAMPLE_RATIO = 0.1      # 10% 采样加速搜索
    
    # 加载数据
    print("\n[1] 加载数据...")
    train_df, test_df = load_data(train_days=30, test_days=3, sample_ratio=SAMPLE_RATIO)
    print(f"训练集: {len(train_df):,}, 测试集: {len(test_df):,}")
    print(f"正样本率: 训练 {train_df['label'].mean():.4f}, 测试 {test_df['label'].mean():.4f}")
    
    # 超参搜索
    print("\n[2] 超参搜索...")
    results = {}
    
    for model_name in MODELS:
        start = time.time()
        best_result, study = run_hyperopt(
            model_name, train_df, test_df,
            n_trials=N_TRIALS,
            n_epochs=N_EPOCHS,
            time_budget=TIME_BUDGET,
        )
        elapsed = time.time() - start
        
        results[model_name] = {
            **best_result,
            'time': elapsed,
            'n_trials': len(study.trials),
        }
    
    # 汇总结果
    print("\n" + "="*60)
    print("超参搜索结果")
    print("="*60)
    
    for model_name, r in results.items():
        print(f"\n【{model_name}】")
        print(f"  AUC: {r['auc']:.6f}")
        print(f"  LogLoss: {r['logloss']:.6f}")
        print(f"  PCOC: {r['pcoc']:.4f}")
        print(f"  Trials: {r['n_trials']}")
        print(f"  Time: {r['time']:.1f}s")
        print(f"  Best params:")
        for k, v in r['params'].items():
            print(f"    {k}: {v}")
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'hyperopt_results_{timestamp}.txt')
    
    with open(output_file, 'w') as f:
        f.write("CTR 模型超参搜索结果\n")
        f.write(f"时间: {datetime.now()}\n")
        f.write(f"配置: N_TRIALS={N_TRIALS}, N_EPOCHS={N_EPOCHS}\n\n")
        
        for model_name, r in results.items():
            f.write(f"\n【{model_name}】\n")
            f.write(f"AUC: {r['auc']:.6f}\n")
            f.write(f"LogLoss: {r['logloss']:.6f}\n")
            f.write(f"PCOC: {r['pcoc']:.4f}\n")
            f.write(f"Best params:\n")
            for k, v in r['params'].items():
                f.write(f"  {k}: {v}\n")
    
    print(f"\n结果已保存: {output_file}")
    
    # 飞书通知
    try:
        msg = f"""⚔️ CTR 超参搜索完成 (10%采样)

"""
        for model_name, r in results.items():
            msg += f"【{model_name}】\n"
            msg += f"  AUC: {r['auc']:.6f}\n"
            msg += f"  LogLoss: {r['logloss']:.6f}\n"
            msg += f"  PCOC: {r['pcoc']:.4f}\n"
            msg += f"  Best: embed_dim={r['params'].get('embed_dim')}, "
            msg += f"dnn={r['params'].get('dnn_hidden_units')}, "
            msg += f"lr={r['params'].get('learning_rate', 0):.6f}\n\n"
        
        FeishuNotifier.notify(msg)
    except Exception as e:
        print(f"飞书通知失败: {e}")
    
    print(f"\n完成时间: {datetime.now()}")


if __name__ == '__main__':
    main()
