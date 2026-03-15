"""
CTR 预估建模 - Auto-PyTorch 自动化模型选择与超参优化
=====================================================

使用 Auto-PyTorch 对 defer_sample_parquet 数据进行:
1. 自动特征编码
2. 特征选择
3. 网络结构搜索
4. 超参数优化
5. 模型集成
"""

import os
import sys
import warnings
import tempfile as tmp
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

# 环境配置
os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# 添加 Auto-PyTorch 路径
sys.path.insert(0, '/mnt/workspace/walter.wan/open_research/Auto-Pytorch')

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


# ============================================================================
# 配置参数
# ============================================================================

DATA_PATH = '/mnt/data/oss_wanjun/pai_work/defer_sample_parquet/'
OUTPUT_DIR = '/mnt/workspace/walter.wan/open_research/ctr_autopytorch/output'
TMP_DIR = '/mnt/workspace/walter.wan/open_research/ctr_autopytorch/tmp'

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

# 高基数特征的频次阈值（出现次数少于此值的归为 'OTHER'）
HIGH_CARDINALITY_THRESHOLD = 100

# 高基数特征列表
HIGH_CARDINALITY_COLS = ['offerid', 'bundle', 'model', 'campaignid']

# 训练配置
TRAIN_DAYS = 7  # 使用多少天数据训练
TEST_DAYS = 2   # 使用多少天数据测试
SAMPLE_FRAC = 0.3  # 采样比例（加速实验）

# Auto-PyTorch 搜索配置
TOTAL_WALLTIME_LIMIT = 1800  # 总搜索时间（秒）
FUNC_EVAL_TIME_LIMIT = 180   # 单次评估时间上限（秒）
RANDOM_SEED = 42


# ============================================================================
# 数据加载与预处理
# ============================================================================

def load_data(data_path: str, n_train_days: int, n_test_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载分天的 parquet 数据，按时间切分训练/测试集
    """
    # 获取所有日期目录
    date_dirs = sorted([d for d in os.listdir(data_path) if d.startswith('sample_date=')])
    print(f"发现 {len(date_dirs)} 天数据")
    
    # 切分训练和测试日期
    train_dirs = date_dirs[:n_train_days]
    test_dirs = date_dirs[n_train_days:n_train_days + n_test_days]
    
    print(f"训练集日期: {train_dirs[0]} ~ {train_dirs[-1]}")
    print(f"测试集日期: {test_dirs[0]} ~ {test_dirs[-1]}")
    
    # 加载训练数据
    train_dfs = []
    for d in train_dirs:
        df = pd.read_parquet(os.path.join(data_path, d))
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    # 加载测试数据
    test_dfs = []
    for d in test_dirs:
        df = pd.read_parquet(os.path.join(data_path, d))
        test_dfs.append(df)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    print(f"训练集大小: {len(train_df):,}")
    print(f"测试集大小: {len(test_df):,}")
    
    return train_df, test_df


def create_cross_features(df: pd.DataFrame, cross_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    创建交叉特征
    """
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
    """
    处理高基数特征：低频值归为 'OTHER'
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    for col in cols:
        if col not in train_df.columns:
            continue
            
        # 统计训练集中的频次
        value_counts = train_df[col].value_counts()
        valid_values = set(value_counts[value_counts >= threshold].index)
        
        # 替换低频值
        train_df[col] = train_df[col].apply(lambda x: x if x in valid_values else 'OTHER')
        test_df[col] = test_df[col].apply(lambda x: x if x in valid_values else 'OTHER')
        
        print(f"处理高基数特征 {col}: 保留 {len(valid_values)} 个值, 阈值={threshold}")
    
    return train_df, test_df


def preprocess_data(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    feature_cols: List[str],
    cross_features: List[Tuple[str, str]],
    high_card_cols: List[str],
    high_card_threshold: int,
    sample_frac: float = 1.0
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    数据预处理主流程
    """
    print("\n" + "="*60)
    print("数据预处理")
    print("="*60)
    
    # 1. 采样（加速实验）
    if sample_frac < 1.0:
        train_df = train_df.sample(frac=sample_frac, random_state=RANDOM_SEED)
        test_df = test_df.sample(frac=sample_frac, random_state=RANDOM_SEED)
        print(f"采样后 - 训练集: {len(train_df):,}, 测试集: {len(test_df):,}")
    
    # 2. 提取标签
    y_train = train_df['label']
    y_test = test_df['label']
    
    # 3. 选择特征列
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    # 4. 创建交叉特征
    X_train = create_cross_features(X_train, cross_features)
    X_test = create_cross_features(X_test, cross_features)
    
    # 5. 处理高基数特征（包括新创建的交叉特征）
    all_high_card_cols = high_card_cols + [f"{c1}_x_{c2}" for c1, c2 in cross_features]
    X_train, X_test = handle_high_cardinality(
        X_train, X_test, all_high_card_cols, high_card_threshold
    )
    
    # 6. 填充缺失值
    X_train = X_train.fillna('MISSING')
    X_test = X_test.fillna('MISSING')
    
    # 7. 转换为字符串类型（确保类别编码正常）
    for col in X_train.columns:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
    
    print(f"\n最终特征数: {len(X_train.columns)}")
    print(f"特征列表: {X_train.columns.tolist()}")
    print(f"训练集正样本率: {y_train.mean():.4f}")
    print(f"测试集正样本率: {y_test.mean():.4f}")
    
    return X_train, y_train, X_test, y_test


# ============================================================================
# Auto-PyTorch 搜索配置
# ============================================================================

def get_search_space_updates() -> HyperparameterSearchSpaceUpdates:
    """
    自定义超参数搜索空间
    """
    updates = HyperparameterSearchSpaceUpdates()
    
    # 数据加载器配置
    updates.append(
        node_name="data_loader",
        hyperparameter="batch_size",
        value_range=[128, 1024],
        default_value=256
    )
    
    # 学习率调度器
    updates.append(
        node_name="lr_scheduler",
        hyperparameter="CosineAnnealingLR:T_max",
        value_range=[10, 50],
        default_value=20
    )
    
    # ResNet dropout
    updates.append(
        node_name='network_backbone',
        hyperparameter='ResNetBackbone:dropout',
        value_range=[0.0, 0.5],
        default_value=0.1
    )
    
    # MLP dropout
    updates.append(
        node_name='network_backbone',
        hyperparameter='MLPBackbone:dropout',
        value_range=[0.0, 0.5],
        default_value=0.1
    )
    
    return updates


def create_autopytorch_task(
    output_dir: str,
    tmp_dir: str,
    seed: int
) -> TabularClassificationTask:
    """
    创建 Auto-PyTorch 任务
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    
    # 配置搜索空间
    search_space_updates = get_search_space_updates()
    
    # 包含的组件
    include_components = {
        'network_backbone': ['MLPBackbone', 'ResNetBackbone', 'ShapedMLPBackbone'],
        'encoder': ['OneHotEncoder'],
        'feature_preprocessor': ['SelectPercentileClassification', 'NoFeaturePreprocessor'],
    }
    
    api = TabularClassificationTask(
        temporary_directory=tmp_dir,
        output_directory=output_dir,
        delete_tmp_folder_after_terminate=False,
        delete_output_folder_after_terminate=False,
        seed=seed,
        search_space_updates=search_space_updates,
        include_components=include_components,
    )
    
    return api


# ============================================================================
# 训练与评估
# ============================================================================

def train_and_evaluate(
    api: TabularClassificationTask,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    total_walltime: int,
    func_eval_time: int
) -> dict:
    """
    执行 Auto-PyTorch 搜索并评估
    """
    print("\n" + "="*60)
    print("开始 Auto-PyTorch 搜索")
    print("="*60)
    print(f"总搜索时间: {total_walltime}s ({total_walltime/60:.1f}min)")
    print(f"单次评估上限: {func_eval_time}s")
    
    # 执行搜索
    api.search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test.copy(),
        y_test=y_test.copy(),
        dataset_name='ctr_defer_sample',
        optimize_metric='roc_auc',
        total_walltime_limit=total_walltime,
        func_eval_time_limit_secs=func_eval_time,
    )
    
    print("\n" + "="*60)
    print("搜索完成，评估结果")
    print("="*60)
    
    # 预测
    y_pred_proba = api.predict_proba(X_test)
    y_pred = api.predict(X_test)
    
    # 计算指标
    # 二分类取正类概率
    if y_pred_proba.ndim == 2:
        y_score = y_pred_proba[:, 1]
    else:
        y_score = y_pred_proba
    
    auc = roc_auc_score(y_test, y_score)
    logloss = log_loss(y_test, y_score)
    
    # PCOC (Predicted CTR / Observed CTR)
    pcoc = y_score.mean() / y_test.mean()
    
    results = {
        'auc': auc,
        'logloss': logloss,
        'pcoc': pcoc,
        'pred_ctr': y_score.mean(),
        'real_ctr': y_test.mean(),
    }
    
    print(f"AUC:      {auc:.6f}")
    print(f"LogLoss:  {logloss:.6f}")
    print(f"PCOC:     {pcoc:.6f}")
    print(f"预测CTR:  {y_score.mean():.6f}")
    print(f"实际CTR:  {y_test.mean():.6f}")
    
    # 打印搜索统计
    print("\n" + "-"*40)
    print("搜索统计:")
    print(api.sprint_statistics())
    
    # 打印最优模型
    print("\n" + "-"*40)
    print("最优模型集成:")
    print(api.show_models())
    
    return results


def save_results(api: TabularClassificationTask, results: dict, output_dir: str):
    """
    保存结果
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存评估指标
    results_file = os.path.join(output_dir, f'results_{timestamp}.txt')
    with open(results_file, 'w') as f:
        f.write("CTR Auto-PyTorch 实验结果\n")
        f.write("="*40 + "\n\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
        f.write("\n" + "="*40 + "\n")
        f.write("搜索统计:\n")
        f.write(api.sprint_statistics())
        f.write("\n" + "="*40 + "\n")
        f.write("模型集成:\n")
        f.write(str(api.show_models()))
    
    print(f"\n结果已保存至: {results_file}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("="*60)
    print("CTR 预估建模 - Auto-PyTorch")
    print("="*60)
    print(f"开始时间: {datetime.now()}")
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    train_df, test_df = load_data(DATA_PATH, TRAIN_DAYS, TEST_DAYS)
    
    # 2. 预处理
    print("\n[2/4] 数据预处理...")
    X_train, y_train, X_test, y_test = preprocess_data(
        train_df, test_df,
        feature_cols=FEATURE_COLS,
        cross_features=CROSS_FEATURES,
        high_card_cols=HIGH_CARDINALITY_COLS,
        high_card_threshold=HIGH_CARDINALITY_THRESHOLD,
        sample_frac=SAMPLE_FRAC
    )
    
    # 3. 创建 Auto-PyTorch 任务
    print("\n[3/4] 创建 Auto-PyTorch 任务...")
    api = create_autopytorch_task(OUTPUT_DIR, TMP_DIR, RANDOM_SEED)
    
    # 4. 训练与评估
    print("\n[4/4] 训练与评估...")
    results = train_and_evaluate(
        api, X_train, y_train, X_test, y_test,
        total_walltime=TOTAL_WALLTIME_LIMIT,
        func_eval_time=FUNC_EVAL_TIME_LIMIT
    )
    
    # 保存结果
    save_results(api, results, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print(f"完成时间: {datetime.now()}")
    print("="*60)
    
    return api, results


if __name__ == '__main__':
    api, results = main()
