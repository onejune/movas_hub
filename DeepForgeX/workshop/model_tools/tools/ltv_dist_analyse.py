import numpy as np
import pandas as pd
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from scipy import stats
from scipy.stats import lognorm, kstest, anderson
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_ltv_distribution(positive_ltv_values, sample_size=None):
    """
    分析正LTV值的分布特征，验证是否符合对数正态分布
    
    Args:
        positive_ltv_values: 正LTV值数组
        sample_size: 如果样本过大，可以随机采样进行分析
    """
    if sample_size and len(positive_ltv_values) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(positive_ltv_values), sample_size, replace=False)
        positive_ltv_values = positive_ltv_values[indices]
    
    print(f"分析样本数量: {len(positive_ltv_values)}")
    print(f"基本统计信息:")
    print(f"  均值: {np.mean(positive_ltv_values):.2f}")
    print(f"  中位数: {np.median(positive_ltv_values):.2f}")
    print(f"  标准差: {np.std(positive_ltv_values):.2f}")
    print(f"  最小值: {np.min(positive_ltv_values):.2f}")
    print(f"  最大值: {np.max(positive_ltv_values):.2f}")
    print(f"  偏度: {stats.skew(positive_ltv_values):.2f}")
    print(f"  峰度: {stats.kurtosis(positive_ltv_values):.2f}")
    print()
    
    # 1. 可视化原始分布
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # 原始LTV分布 - 线性尺度
    axes[0, 0].hist(positive_ltv_values, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('原始LTV分布 (线性尺度)')
    axes[0, 0].set_xlabel('LTV')
    axes[0, 0].set_ylabel('密度')
    
    # 原始LTV分布 - 对数尺度X轴
    axes[0, 1].hist(positive_ltv_values, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_title('原始LTV分布 (对数X轴)')
    axes[0, 1].set_xlabel('LTV (log scale)')
    axes[0, 1].set_ylabel('密度')
    
    # 对数LTV分布
    log_ltv_values = np.log(positive_ltv_values)
    axes[0, 2].hist(log_ltv_values, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 2].set_title('log(LTV)分布')
    axes[0, 2].set_xlabel('log(LTV)')
    axes[0, 2].set_ylabel('密度')
    
    # 2. Q-Q图比较
    # log(LTV) vs 正态分布Q-Q图
    stats.probplot(log_ltv_values, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('log(LTV) vs 正态分布 Q-Q图')
    axes[1, 0].grid(True)
    
    # 原始LTV vs 对数正态分布Q-Q图
    # 先拟合对数正态分布参数
    shape, loc, scale = lognorm.fit(positive_ltv_values, floc=0)
    stats.probplot(positive_ltv_values, dist=lognorm(shape, loc=loc, scale=scale), plot=axes[1, 1])
    axes[1, 1].set_title('LTV vs 拟合对数正态分布 Q-Q图')
    axes[1, 1].grid(True)
    
    # 正态分布Q-Q图 (用于对比)
    stats.probplot(log_ltv_values, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('log(LTV)正态Q-Q图 (理想应为直线)')
    axes[1, 2].grid(True)
    
    # 3. 箱形图和分位点图
    # 箱形图
    axes[2, 0].boxplot(positive_ltv_values)
    axes[2, 0].set_title('LTV 箱形图')
    axes[2, 0].set_ylabel('LTV')
    axes[2, 0].set_yscale('log')  # 使用对数刻度以便更好地显示长尾数据
    
    # 分位点图 - 累积分布函数
    sorted_ltv = np.sort(positive_ltv_values)
    cdf = np.arange(1, len(sorted_ltv) + 1) / len(sorted_ltv)
    axes[2, 1].plot(sorted_ltv, cdf, marker='.', linestyle='-', markersize=1, linewidth=0.5)
    axes[2, 1].set_title('LTV 累积分布函数 (CDF)')
    axes[2, 1].set_xlabel('LTV')
    axes[2, 1].set_ylabel('累积概率')
    axes[2, 1].set_xscale('log')  # 使用对数刻度
    
    # 分位点图 - 指定分位点
    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
    quantile_values = np.percentile(positive_ltv_values, [q * 100 for q in quantiles])
    
    axes[2, 2].plot(quantiles, quantile_values, marker='o', linestyle='-', linewidth=2, markersize=6)
    axes[2, 2].set_title('LTV 分位点图')
    axes[2, 2].set_xlabel('分位数')
    axes[2, 2].set_ylabel('LTV值')
    axes[2, 2].set_yscale('log')  # 使用对数刻度
    axes[2, 2].grid(True, alpha=0.3)
    
    # 在分位点图上标注关键分位点
    for i, (q, val) in enumerate(zip(quantiles, quantile_values)):
        if i % 2 == 0:  # 每隔一个点标注，避免重叠
            axes[2, 2].annotate(f'{q:.3f}\n{val:.2f}', 
                              (q, val), 
                              textcoords="offset points", 
                              xytext=(0,10), 
                              ha='center',
                              fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # 4. 分位点详细信息
    print("=== LTV分位点详细信息 ===")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
    for p in percentiles:
        value = np.percentile(positive_ltv_values, p)
        users_above = (positive_ltv_values > value).sum()
        pct_above = users_above / len(positive_ltv_values) * 100
        print(f"{p}%分位数: {value:.2f} (高于此值的用户占比: {pct_above:.2f}%)")
    
    top_1_pct = np.percentile(positive_ltv_values, 99)
    top_1_pct_contribution = positive_ltv_values[positive_ltv_values >= top_1_pct].sum() / positive_ltv_values.sum() * 100
    print(f"前1%高价值用户贡献总收入的: {top_1_pct_contribution:.2f}%")
    
    top_5_pct = np.percentile(positive_ltv_values, 95)
    top_5_pct_contribution = positive_ltv_values[positive_ltv_values >= top_5_pct].sum() / positive_ltv_values.sum() * 100
    print(f"前5%高价值用户贡献总收入的: {top_5_pct_contribution:.2f}%")
    
    # 5. 统计检验
    print("\n=== 分布拟合检验 ===")
    
    # Kolmogorov-Smirnov检验 (LTV vs 拟合对数正态)
    ks_stat, ks_p = kstest(positive_ltv_values, lambda x: lognorm.cdf(x, shape, loc=loc, scale=scale))
    print(f"Kolmogorov-Smirnov检验 (LTV vs 拟合对数正态):")
    print(f"  统计量: {ks_stat:.4f}, p值: {ks_p:.4f}")
    print(f"  结论: {'符合对数正态分布' if ks_p > 0.05 else '不符合对数正态分布'} (α=0.05)")
    print()
    
    # Anderson-Darling检验 (log(LTV) vs 正态)
    ad_result = anderson(log_ltv_values, dist='norm')
    print(f"Anderson-Darling检验 (log(LTV) vs 正态分布):")
    print(f"  统计量: {ad_result.statistic:.4f}")
    for i in range(len(ad_result.critical_values)):
        sl, cv = ad_result.significance_level[i], ad_result.critical_values[i]
        print(f"  {sl}%显著性水平临界值: {cv:.4f}")
    conclusion = "符合正态分布" if ad_result.statistic < ad_result.critical_values[2] else "不符合正态分布"
    print(f"  结论: {conclusion} (1%显著性水平)")
    print()
    
    # 6. 其他对数正态分布特征验证
    print("=== 对数正态分布特征验证 ===")
    
    # 理论vs实际矩验证
    mean_theory = scale * np.exp(shape**2 / 2)
    var_theory = (scale**2) * (np.exp(shape**2) - 1) * np.exp(shape**2)
    mean_actual = np.mean(positive_ltv_values)
    var_actual = np.var(positive_ltv_values)
    
    print(f"理论均值: {mean_theory:.2f}, 实际均值: {mean_actual:.2f}")
    print(f"理论方差: {var_theory:.2f}, 实际方差: {var_actual:.2f}")
    print(f"均值差异率: {abs(mean_theory - mean_actual) / mean_actual * 100:.2f}%")
    print(f"方差差异率: {abs(var_theory - var_actual) / var_actual * 100:.2f}%")
    print()
    
    # 7. 返回拟合参数和结论
    return {
        'shape': shape,
        'loc': loc,
        'scale': scale,
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'ad_stat': ad_result.statistic,
        'mean_diff_rate': abs(mean_theory - mean_actual) / mean_actual * 100,
        'var_diff_rate': abs(var_theory - var_actual) / var_actual * 100,
        'top_1_pct_contribution': top_1_pct_contribution,
        'top_5_pct_contribution': top_5_pct_contribution
    }


def compare_with_other_distributions(positive_ltv_values, sample_size=None):
    """
    比较对数正态分布与其他常见分布的拟合效果
    """
    if sample_size and len(positive_ltv_values) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(positive_ltv_values), sample_size, replace=False)
        positive_ltv_values = positive_ltv_values[indices]
    
    # 拟合不同分布
    distributions = {
        'Log-normal': lognorm,
        'Weibull': stats.weibull_min,
        'Gamma': stats.gamma,
        'Pareto': stats.pareto
    }
    
    results = {}
    for name, dist in distributions.items():
        try:
            if name == 'Pareto':
                # Pareto需要额外参数
                params = dist.fit(positive_ltv_values, floc=0)
            else:
                params = dist.fit(positive_ltv_values, floc=0)
            
            # 计算KS检验
            ks_stat, ks_p = kstest(positive_ltv_values, lambda x: dist.cdf(x, *params))
            log_likelihood = dist.logpdf(positive_ltv_values, *params).sum()
            n = len(positive_ltv_values)
            k = len(params) - 1  # 减去floc=0的固定参数
            aic = 2 * k - 2 * log_likelihood
            
            results[name] = {
                'params': params,
                'ks_stat': ks_stat,
                'ks_p': ks_p,
                'aic': aic,
                'log_likelihood': log_likelihood
            }
        except Exception as e:
            print(f"{name} 拟合失败: {e}")
            continue
    
    # 打印比较结果
    print("=== 分布拟合比较 ===")
    print(f"{'分布':<12} {'KS统计量':<10} {'p值':<10} {'AIC':<10} {'对数似然':<12}")
    print("-" * 60)
    for name, result in results.items():
        print(f"{name:<12} {result['ks_stat']:<10.4f} {result['ks_p']:<10.4f} {result['aic']:<10.2f} {result['log_likelihood']:<12.2f}")
    
    # 根据AIC选择最佳分布
    best_dist = min(results.keys(), key=lambda x: results[x]['aic'])
    print(f"\n根据AIC准则，最佳分布是: {best_dist}")
    
    return results

def run_analyse(positive_ltv_values):
    # 主要分析
    fit_results = analyze_ltv_distribution(positive_ltv_values, sample_size=100000)
    
    print("\n" + "="*50)
    print("与其他分布比较...")
    compare_results = compare_with_other_distributions(positive_ltv_values, sample_size=50000)
    
    print("\n" + "="*50)
    print("分析结论:")
    
    if fit_results['ks_p'] > 0.05:
        print("✓ KS检验表明数据符合对数正态分布 (p > 0.05)")
    else:
        print("✗ KS检验表明数据不符合对数正态分布 (p ≤ 0.05)")
    
    if fit_results['ad_stat'] < 0.75:  # Anderson-Darling临界值约0.75左右
        print("✓ Anderson-Darling检验支持对数正态分布假设")
    else:
        print("✗ Anderson-Darling检验不支持对数正态分布假设")
    
    if fit_results['mean_diff_rate'] < 10 and fit_results['var_diff_rate'] < 20:
        print("✓ 理论矩与实际矩差异较小，拟合良好")
    else:
        print("✗ 理论矩与实际矩差异较大，拟合一般")
    
    if fit_results['top_1_pct_contribution'] > 30:
        print(f"✓ 长尾特征明显，前1%用户贡献{fit_results['top_1_pct_contribution']:.1f}%收入")
    else:
        print(f"✗ 长尾特征不明显，前1%用户仅贡献{fit_results['top_1_pct_contribution']:.1f}%收入")
    
    return fit_results

#查找 ltv 分桶中密度最高的桶以及对应的 ltv 值
def find_exception_bins(positive_ltv_values):
    log_ltv = np.log(positive_ltv_values)

    # 绘制直方图并返回 bin 边界和频次
    # 方式一：统计 log(LTV) 的直方图 bin 中哪个 bin 最高
    counts, bin_edges, _ = plt.hist(log_ltv, bins=100, density=True)
    plt.close()  # 不显示图

    # 找出密度最高的 bin
    max_bin_index = np.argmax(counts)
    left_edge = bin_edges[max_bin_index]
    right_edge = bin_edges[max_bin_index + 1]

    print(f"密度最高的 log(LTV) 区间: [{left_edge:.3f}, {right_edge:.3f})")
    print(f"对应的原始 LTV 区间: [{np.exp(left_edge):.6f}, {np.exp(right_edge):.6f})")

    # 查看该区间内的实际LTV值样本
    mask = (log_ltv >= left_edge) & (log_ltv < right_edge)
    sample_ltv_in_peak = positive_ltv_values[mask][:10]  # 取前10个样本
    print("该区间内的部分LTV值示例:", sample_ltv_in_peak)

    # 方式二：用 pandas 查看 LTV 的频次分布（适合离散值）
    import pandas as pd
    df = pd.DataFrame({'ltv': positive_ltv_values})
    # 四舍五入到小数点后4位（避免浮点误差）
    df['ltv_rounded'] = df['ltv'].round(6)

    # 统计最常见的10个LTV值
    top_values = df['ltv_rounded'].value_counts().head(10)
    print("最常见的10个LTV值（四舍五入后）:")
    print(top_values)

    # 计算它们占总体的比例
    total = len(df)
    for val, count in top_values.items():
        print(f"LTV = {val}: {count} 用户 ({count/total*100:.2f}%)")

def run_ltv_analysis_with_spark_data(spark, start_date='2025-11-15', end_date='2025-11-18', revenue_column='revenue'):
    """
    从Spark读取数据并运行完整的LTV分布分析
    """
    print("开始从Spark读取数据...")
    print(f"日期范围: {start_date} 到 {end_date}")
    
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD.")

    data_path_list = []
    current = start
    while current <= end:
        path = 'oss://spark-ml-train-new/dsp_algo/ivr/sample/ivr_sample_v7/parquet/part=%s/*' % (current.strftime("%Y-%m-%d"))
        data_path_list.append(path)
        current += timedelta(days=1)

    print("数据路径列表:")
    for path in data_path_list:
        print(f"  {path}")
    
    # 读取数据
    df = spark.read.option("inferSchema", "true").parquet(*data_path_list)
    
    # 转换revenue列为float类型并过滤出 > 0 的记录
    df = df.withColumn(revenue_column, df[revenue_column].cast("float"))
    df = df.filter("business_type not in('shein', 'aerta', 'lazada_rta') and revenue>0 and revenue<100")
    # 定义目标值和容差
    target = 0.0012345
    epsilon = 1e-7  # 根据你的数据精度调整，比如 0.0000001

    # 过滤掉 revenue 接近 target 的行
    df_positive = df.filter(
        F.abs(F.col(revenue_column) - target) > epsilon
    )
    
    print(f"\n原始数据总数: {df.count()}")
    print(f"正LTV记录数: {df_positive.count()}")
    
    if df_positive.count() == 0:
        print("错误: 没有找到正LTV的记录")
        return None
    
    # 转换为Pandas进行分析（如果数据量大，可以先采样）
    if df_positive.count() > 100000:
        print(f"数据量较大({df_positive.count()})，进行随机采样...")
        sample_fraction = 100000 / df_positive.count()
        df_sample = df_positive.sample(withReplacement=False, fraction=sample_fraction)
        print(f"采样后记录数: {df_sample.count()}")
    else:
        df_sample = df_positive
    
    # 提取正LTV值
    positive_ltv_values = df_sample.select(revenue_column).rdd.flatMap(lambda x: x).collect()
    positive_ltv_values = np.array(positive_ltv_values)
    
    print(f"提取到的正LTV值数量: {len(positive_ltv_values)}")
    print(f"LTV值范围: {np.min(positive_ltv_values):.2f} - {np.max(positive_ltv_values):.2f}")
    
    print("\n" + "="*50)
    print("开始LTV分布分析...")
    
    #find_exception_bins(positive_ltv_values)
    run_analyse(positive_ltv_values)

if __name__ == "__main__":
    run_ltv_analysis_with_spark_data(spark, start_date='2025-11-15', end_date='2025-11-20')



