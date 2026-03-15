"""
CTR 预估建模 - FLAML 自动化模型选择与超参优化
==============================================

使用 FLAML (Fast and Lightweight AutoML) 对 defer_sample_parquet 数据进行:
1. 自动特征编码
2. 模型选择 (LightGBM, XGBoost, CatBoost, etc.)
3. 超参数优化
4. 模型集成
"""

import os
import sys
import warnings
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

from flaml import AutoML

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


# ============================================================================
# 配置参数
# ============================================================================

DATA_PATH = '/mnt/data/oss_wanjun/pai_work/defer_sample_parquet/'
OUTPUT_DIR = '/mnt/workspace/walter.wan/open_research/ctr_autopytorch/output'

# 排除的列（不作为特征）
EXCLUDE_COLS = ['req_hour', 'sample_bid_date', 'diff_hours', 'label']

# 实际使用的特征列
FEATURE_COLS = [
    'business_type', 'offerid', 'country', 'bundle',
    'adx', 'make', 'model', 'demand_pkgname', 'campaignid'
]

# 交叉特征组合
CROSS_FEATURES = [
    ('country', 'adx'),
    ('country', 'business_type'),
    ('make', 'model'),
    ('adx', 'business_type'),
    ('country', 'demand_pkgname'),
]

# 高基数特征的频次阈值
HIGH_CARDINALITY_THRESHOLD = 100

# 高基数特征列表
HIGH_CARDINALITY_COLS = ['offerid', 'bundle', 'model', 'campaignid']

# 训练配置
TRAIN_DAYS = 7
TEST_DAYS = 2
SAMPLE_FRAC = 0.3

# FLAML 搜索配置
TIME_BUDGET = 600  # 搜索时间（秒）
METRIC = 'roc_auc'
ESTIMATOR_LIST = ['lgbm', 'xgboost', 'catboost']  # 候选模型
RANDOM_SEED = 42


# ============================================================================
# 数据加载与预处理
# ============================================================================

def load_data(data_path: str, n_train_days: int, n_test_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载分天的 parquet 数据"""
    date_dirs = sorted([d for d in os.listdir(data_path) if d.startswith('sample_date=')])
    print(f"发现 {len(date_dirs)} 天数据")
    
    train_dirs = date_dirs[:n_train_days]
    test_dirs = date_dirs[n_train_days:n_train_days + n_test_days]
    
    print(f"训练集日期: {train_dirs[0]} ~ {train_dirs[-1]}")
    print(f"测试集日期: {test_dirs[0]} ~ {test_dirs[-1]}")
    
    train_dfs = [pd.read_parquet(os.path.join(data_path, d)) for d in train_dirs]
    test_dfs = [pd.read_parquet(os.path.join(data_path, d)) for d in test_dirs]
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    print(f"训练集大小: {len(train_df):,}")
    print(f"测试集大小: {len(test_df):,}")
    
    return train_df, test_df


def create_cross_features(df: pd.DataFrame, cross_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """创建交叉特征"""
    df = df.copy()
    for col1, col2 in cross_pairs:
        cross_name = f"{col1}_x_{col2}"
        df[cross_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
        print(f"创建交叉特征: {cross_name}, 基数: {df[cross_name].nunique()}")
    return df


def handle_high_cardinality(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    cols: List[str], 
    threshold: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """处理高基数特征"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    for col in cols:
        if col not in train_df.columns:
            continue
        value_counts = train_df[col].value_counts()
        valid_values = set(value_counts[value_counts >= threshold].index)
        train_df[col] = train_df[col].apply(lambda x: x if x in valid_values else 'OTHER')
        test_df[col] = test_df[col].apply(lambda x: x if x in valid_values else 'OTHER')
        print(f"处理高基数特征 {col}: 保留 {len(valid_values)} 个值")
    
    return train_df, test_df


def encode_categorical_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Label 编码类别特征"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        # 合并训练和测试的唯一值
        all_values = pd.concat([train_df[col], test_df[col]]).astype(str).unique()
        le.fit(all_values)
        
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        encoders[col] = le
    
    return train_df, test_df, encoders


def preprocess_data(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    feature_cols: List[str],
    cross_features: List[Tuple[str, str]],
    high_card_cols: List[str],
    high_card_threshold: int,
    sample_frac: float = 1.0
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """数据预处理主流程"""
    print("\n" + "="*60)
    print("数据预处理")
    print("="*60)
    
    # 采样
    if sample_frac < 1.0:
        train_df = train_df.sample(frac=sample_frac, random_state=RANDOM_SEED)
        test_df = test_df.sample(frac=sample_frac, random_state=RANDOM_SEED)
        print(f"采样后 - 训练集: {len(train_df):,}, 测试集: {len(test_df):,}")
    
    # 提取标签
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    # 选择特征列
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    # 创建交叉特征
    X_train = create_cross_features(X_train, cross_features)
    X_test = create_cross_features(X_test, cross_features)
    
    # 处理高基数特征
    all_high_card_cols = high_card_cols + [f"{c1}_x_{c2}" for c1, c2 in cross_features]
    all_high_card_cols = [c for c in all_high_card_cols if c in X_train.columns]
    X_train, X_test = handle_high_cardinality(X_train, X_test, all_high_card_cols, high_card_threshold)
    
    # 填充缺失值
    X_train = X_train.fillna('MISSING')
    X_test = X_test.fillna('MISSING')
    
    # Label 编码
    cat_cols = X_train.columns.tolist()
    X_train, X_test, _ = encode_categorical_features(X_train, X_test, cat_cols)
    
    print(f"\n最终特征数: {len(X_train.columns)}")
    print(f"特征列表: {X_train.columns.tolist()}")
    print(f"训练集正样本率: {y_train.mean():.4f}")
    print(f"测试集正样本率: {y_test.mean():.4f}")
    
    return X_train, y_train, X_test, y_test


# ============================================================================
# FLAML 训练与评估
# ============================================================================

def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    time_budget: int,
    metric: str,
    estimator_list: List[str]
) -> Tuple[AutoML, dict]:
    """执行 FLAML 搜索并评估"""
    print("\n" + "="*60)
    print("开始 FLAML 自动机器学习搜索")
    print("="*60)
    print(f"搜索时间: {time_budget}s ({time_budget/60:.1f}min)")
    print(f"优化指标: {metric}")
    print(f"候选模型: {estimator_list}")
    
    # 创建 AutoML 实例
    automl = AutoML()
    
    # 执行搜索
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task='classification',
        metric=metric,
        time_budget=time_budget,
        estimator_list=estimator_list,
        seed=RANDOM_SEED,
        verbose=1,
        early_stop=True,
        ensemble=True,  # 启用集成
    )
    
    print("\n" + "="*60)
    print("搜索完成，评估结果")
    print("="*60)
    
    # 预测
    y_pred_proba = automl.predict_proba(X_test)[:, 1]
    y_pred = automl.predict(X_test)
    
    # 计算指标
    auc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    pcoc = y_pred_proba.mean() / y_test.mean()
    
    results = {
        'auc': auc,
        'logloss': logloss,
        'pcoc': pcoc,
        'pred_ctr': y_pred_proba.mean(),
        'real_ctr': y_test.mean(),
        'best_model': automl.best_estimator,
        'best_config': automl.best_config,
    }
    
    print(f"\n{'='*40}")
    print("评估指标:")
    print(f"{'='*40}")
    print(f"AUC:      {auc:.6f}")
    print(f"LogLoss:  {logloss:.6f}")
    print(f"PCOC:     {pcoc:.6f}")
    print(f"预测CTR:  {y_pred_proba.mean():.6f}")
    print(f"实际CTR:  {y_test.mean():.6f}")
    
    print(f"\n{'='*40}")
    print("最优模型:")
    print(f"{'='*40}")
    print(f"模型类型: {automl.best_estimator}")
    print(f"最优配置: {automl.best_config}")
    
    # 特征重要性
    if hasattr(automl.model, 'feature_importances_'):
        print(f"\n{'='*40}")
        print("特征重要性 (Top 10):")
        print(f"{'='*40}")
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': automl.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(importance.head(10).to_string(index=False))
        results['feature_importance'] = importance
    
    return automl, results


def save_results(automl: AutoML, results: dict, output_dir: str):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存评估指标
    results_file = os.path.join(output_dir, f'results_{timestamp}.txt')
    with open(results_file, 'w') as f:
        f.write("CTR FLAML 实验结果\n")
        f.write("="*40 + "\n\n")
        f.write(f"AUC: {results['auc']:.6f}\n")
        f.write(f"LogLoss: {results['logloss']:.6f}\n")
        f.write(f"PCOC: {results['pcoc']:.6f}\n")
        f.write(f"预测CTR: {results['pred_ctr']:.6f}\n")
        f.write(f"实际CTR: {results['real_ctr']:.6f}\n")
        f.write(f"\n最优模型: {results['best_model']}\n")
        f.write(f"最优配置: {results['best_config']}\n")
        
        if 'feature_importance' in results:
            f.write("\n特征重要性:\n")
            f.write(results['feature_importance'].to_string(index=False))
    
    print(f"\n结果已保存至: {results_file}")
    
    # 保存模型
    import pickle
    model_file = os.path.join(output_dir, f'model_{timestamp}.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(automl, f)
    print(f"模型已保存至: {model_file}")


# ============================================================================
# 主函数
# ============================================================================

def main(
    train_days: int = TRAIN_DAYS,
    test_days: int = TEST_DAYS,
    sample_frac: float = SAMPLE_FRAC,
    time_budget: int = TIME_BUDGET
):
    print("="*60)
    print("CTR 预估建模 - FLAML AutoML")
    print("="*60)
    print(f"开始时间: {datetime.now()}")
    print(f"配置: 训练{train_days}天, 测试{test_days}天, 采样{sample_frac*100:.0f}%, 搜索{time_budget}s")
    
    # 1. 加载数据
    print("\n[1/3] 加载数据...")
    train_df, test_df = load_data(DATA_PATH, train_days, test_days)
    
    # 2. 预处理
    print("\n[2/3] 数据预处理...")
    X_train, y_train, X_test, y_test = preprocess_data(
        train_df, test_df,
        feature_cols=FEATURE_COLS,
        cross_features=CROSS_FEATURES,
        high_card_cols=HIGH_CARDINALITY_COLS,
        high_card_threshold=HIGH_CARDINALITY_THRESHOLD,
        sample_frac=sample_frac
    )
    
    # 3. 训练与评估
    print("\n[3/3] 训练与评估...")
    automl, results = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        time_budget=time_budget,
        metric=METRIC,
        estimator_list=ESTIMATOR_LIST
    )
    
    # 保存结果
    save_results(automl, results, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print(f"完成时间: {datetime.now()}")
    print("="*60)
    
    return automl, results


if __name__ == '__main__':
    automl, results = main()
