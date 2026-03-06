import metaspore as ms
import pyspark
import numpy as np
import pandas as pd
import yaml
import subprocess
import argparse
import sys
import os
from operator import itemgetter
from pyspark.sql.types import StringType
from pyspark.sql import functions as F
from metrics_eval import compute_auc_pcoc
sys.path.append('../MetaSpore/') 
from metaspore.algos.delay_feedback import DEFER
from metaspore.algos.delay_feedback import DEFERAgent
from pyspark.sql.functions import rand
from datetime import datetime, timedelta
from tabulate import tabulate
import torch
# movas_logger.py 与 dnn_ms.py 在同一目录下
from movas_logger import MovasLogger, how_much_time
import matplotlib.pyplot as plt

# matplotlib用于统计分析，非必须

class MsModelTrainFlow:
    def __init__(self, config_path):
        self.config_path = config_path
        self.params = self._load_config(config_path)
        self._initialize_params()
        
        self.spark_session = None
        self.model_module = None
        self.trained_model_path = None

        # logging.getLogger("py4j").setLevel(logging.ERROR) # 移除
        # logging.getLogger("pyspark").setLevel(logging.ERROR) # 移除

    def _load_config(self, path):
        params = dict()
        with open(path, 'r') as stream:
            params = yaml.load(stream, Loader=yaml.FullLoader)
            # 使用 MovasLogger 记录配置加载信息
            # MovasLogger.add_log(content=f'Config loaded: {params}') # 注意：MovasLogger此时可能还未初始化
            print(f'Debug--load config: {params}') # 保留 print 或在 MovasLogger 初始化后记录
        required_keys = ['train_path_prefix', 'model_out_base_path', 'worker_count', 
                         'server_count', 'embedding_size', 
                         'combine_schema_path', 'dnn_hidden_units',
                         'ftrl_l1', 'ftrl_l2', 'ftrl_alpha', 'ftrl_beta', 'adam_learning_rate',
                         'batch_size', 'app_name', 'local_spark', 'worker_memory', 'server_memory',
                         'coordinator_memory', 'input_label_column_index', 'experiment_name',
                         'train_start_date', 'train_end_date', 'validation_date', 'movas_log_output_path'] # 确保 movas_log_output_path 在配置中
        for key in required_keys:
            assert key in params, f'Missing required config: {key}'
        return params

    def _initialize_params(self):
        for key, value in self.params.items():
            setattr(self, key, value)
        self.train_start_date_dt = self.train_start_date
        self.train_end_date_dt = self.train_end_date
        # self.movas_log_output_path 已通过 setattr 设置，如果配置文件中有的话
        # 如果配置文件中没有 movas_log_output_path，需要提供一个默认值或确保它存在
        # print(f"Debug--initialize params: {self.params}")
        if not hasattr(self, 'movas_log_output_path'):
             self.movas_log_output_path = self.params.get('movas_log_output_path', 'dnn_ms_movas_log.txt')


    def _init_spark(self):
        MovasLogger.add_log(content='Debug -- spark init') # MovasLogger 在此之前应已可用，但 init() 在之后
                                                        # 这意味着 MovasLogger 的静态方法 add_log 必须能在 init 前被调用
                                                        # 或者将此日志移到 MovasLogger.init() 之后
        spark_confs = {
            "spark.eventLog.enabled": "false",
            "spark.driver.memory": "20g",
            "spark.executor.memory": "16g",
            "spark.executor.instances": "1",
            "spark.executor.cores": "16",
            "spark.default.parallelism": "32",
            "spark.local.dir": "/data/spark/tmp", 
            "spark.network.timeout": "500",
            "spark.ui.showConsoleProgress": "false",
            "spark.sql.parquet.writeLegacyFormat": "true",
            "spark.executorEnv.PYSPARK_PYTHON": "/usr/bin/python3.8",
            "spark.executorEnv.PYSPARK_DRIVER_PYTHON": "/usr/bin/python3.8",
        }
        try:
            # subprocess.run(['zip', '-r', 'python.zip', '../MetaSpore/python'], cwd='.', check=True)
            # spark_confs["spark.submit.pyFiles"] = "python.zip"
            pass
        except subprocess.CalledProcessError as e:
            # 假设 MovasLogger.add_log 可以在 init 之前调用，或者有默认的输出方式（如print）
            MovasLogger.add_log(level='ERROR', content=f"Failed to create python.zip. Error: {e}")
        except FileNotFoundError:
            MovasLogger.add_log(level='ERROR', content=f"'zip' command not found or '../../../python' directory does not exist.")

        self.spark_session = ms.spark.get_session(local=self.local_spark,
                                                  app_name=self.app_name,
                                                  batch_size=self.batch_size,
                                                  worker_count=self.worker_count,
                                                  server_count=self.server_count,
                                                  worker_memory=self.worker_memory,
                                                  server_memory=self.server_memory,
                                                  coordinator_memory=self.coordinator_memory,
                                                  spark_confs=spark_confs)
        sc = self.spark_session.sparkContext
        sc.setLogLevel("ERROR") # PySpark 日志级别控制，可以保留或移除，取决于是否希望完全依赖 MovasLogger
        
        # 初始化 MovasLogger，确保 output_path 已设置
        MovasLogger.init(spark=self.spark_session, output_path=self.movas_log_output_path)
        MovasLogger.add_log(content="MovasLogger initialized.")
        MovasLogger.add_log(content=f'Spark version: {sc.version}, appId: {sc.applicationId}, uiWebUrl: {sc.uiWebUrl}')
        assert self.spark_session is not None, 'Spark session failed to initialize.'

    def _stop_spark(self):
        if self.spark_session:
            MovasLogger.add_log(content='Debug--spark stop')
            self.spark_session.sparkContext.stop()
            self.spark_session = None
   
    def random_sample(self, df):
        '''
        # 分离正样本和负样本
        df_pos = df[df['purchase'] == 1]
        df_neg = df[df['purchase'] == 0]

        # 对负样本进行随机采样10%
        df_neg_sampled = df_neg.sample(fraction=0.1, seed=42)

        # 合并正样本和采样后的负样本
        df_balanced = df_pos.union(df_neg_sampled)

        # 洗牌数据集
        df_balanced = df_balanced.orderBy(rand(seed=29))
        '''

        df_filtered = df.filter(
            (F.col("purchase") == 1) |
            (F.rand(seed=42) < 0.1)
        )
        return df_filtered
 
    #@how_much_time   
    def _read_dataset_by_date(self, base_path, data_path):
        MovasLogger.add_log(content=f"Reading Parquet data from directory: {data_path}")
        df = self.spark_session.read.parquet(data_path)
        
        for col_name in df.columns:
            if col_name == 'purchase':
                df = df.withColumn(col_name, F.col(col_name).cast("float"))
            else:
                df = df.withColumn(col_name, F.col(col_name).cast("string"))
        # df = self.random_sample(df)
        df = df.filter(F.col("demand_pkgname").isin(["COM.ALIBABA.ALIEXPRESSHD"]))
        df = df.filter(F.col("purchase").isin([0, 1]))
        df = df.fillna('unknown')
        return df

    def _build_model_module(self):
        self.model_module = DEFER(
                                batch_norm=False,
                                defer_embedding_dim=self.embedding_size,
                                deep_embedding_dim=self.embedding_size,
                                #wide_column_name_path=self.column_name_path,
                                defer_combine_schema_path=self.defer_combine_schema_path,
                                #deep_column_name_path=self.column_name_path,
                                deep_combine_schema_path=self.combine_schema_path,
                                dnn_hidden_units=self.dnn_hidden_units,
                                defer_hidden_units=self.defer_hidden_units,
                                dnn_hidden_activations=self.dnn_activation,
                                defer_hidden_activations=self.defer_activation,
                                # net_dropout=self.net_dropout,
                                ftrl_l1=self.ftrl_l1,
                                ftrl_l2=self.ftrl_l2,
                                ftrl_alpha=self.ftrl_alpha,
                                ftrl_beta=self.ftrl_beta)
        MovasLogger.add_log(content="Model module built.")

    #@how_much_time
    def _train_model(self, train_dataset, model_in_path_current, aux_model_in_path, model_out_path_current, model_version_current):
        if not self.model_module:
            self._build_model_module()
        # print(self.model_module)
        # input() 
        estimator = ms.PyTorchEstimator(module=self.model_module,
                                      agent_class=DEFERAgent,
                                      worker_count=self.worker_count,
                                      server_count=self.server_count,
                                      model_in_path=model_in_path_current,
                                      aux_model_in_path=aux_model_in_path,
                                      model_out_path=model_out_path_current,
                                      model_export_path=None, 
                                      model_version=model_version_current,
                                      experiment_name=self.experiment_name,
                                      # shuffle_training_dataset = self.shuffle_training_dataset,
                                      # input_label_column_index=self.input_label_column_index,
                                      input_label_column_name=self.train_input_label_column_name,
                                      metric_update_interval=100)
        estimator.updater = ms.AdamTensorUpdater(self.adam_learning_rate)
        
        MovasLogger.add_log(content=f"Starting training for version: {model_version_current}, Model In: {model_in_path_current}, Model Out: {model_out_path_current}")
        model = estimator.fit(train_dataset)
        
        if os.path.exists(model_out_path_current):
            pass
            #MovasLogger.add_log(content=f"Model output (local path {model_out_path_current}): {os.listdir(model_out_path_current)}")
        else:
            MovasLogger.add_log(level='WARNING', content=f"Model output path {model_out_path_current} does not exist.")
        self.trained_model_path = model_out_path_current 
        return model

    @how_much_time
    def _transform_data(self, dataset_to_transform, aux_model_in_path, model_in_path_current):
        if not self.model_module:
            self._build_model_module()

        model_transformer = ms.PyTorchModel(module=self.model_module,
                                agent_class=DEFERAgent,
                                worker_count=self.worker_count,
                                server_count=self.server_count,
                                model_in_path=model_in_path_current,
                                aux_model_in_path=aux_model_in_path,
                                experiment_name=self.experiment_name,
                                input_label_column_name=self.val_input_label_column_name,)
                                # input_label_column_index=self.input_label_column_index)

        MovasLogger.add_log(content=f"Transforming data using model from: {model_in_path_current}")
        test_result = model_transformer.transform(dataset_to_transform)
        #MovasLogger.add_log(content=f"Test result sample:\n{MovasLogger.get_df_showString(test_result, lines=5)}")
        return test_result
    
    def plot_pos_coeff_distribution(self, coeff_np, pkg_name, pos, save_path="/data/kailiang/ruf2/defer/picture/"):
        """
        绘制并保存pos_coeff分布直方图（适配大数据量）
        
        :param coeff_np: 通过 pos_coeff.numpy() 得到的NumPy数组
        :param pkg_name: 包名（用于标题和文件名）
        :param save_path: 图像保存路径
        """
        plt.figure(figsize=(12, 6), dpi=100)  # 提高分辨率和尺寸
        log_scale = True
        # 自动计算合适的bins数量（Sturges规则 + 上限限制）
        if pos:
            bins = np.linspace(0, 10, 21)
        else:
            max_bins = min(200, int(1 + 3.322 * np.log10(len(coeff_np))))  # 避免bins过多导致内存问题
            bins = np.linspace(np.min(coeff_np), np.max(coeff_np), max_bins)
        
        # 绘制直方图（优化参数）
        counts, edges, patches = plt.hist(
            coeff_np, 
            bins=bins,
            alpha=0.7,          # 透明度（0=完全透明，1=不透明）
            edgecolor='black',   # 边框颜色
            linewidth=0.5,       # 边框粗细
            density=False,       # 设为True可显示概率密度
            log=log_scale         # 设为True可启用y轴对数刻度（适合长尾分布）
        )
        for i, (count, patch) in enumerate(zip(counts, patches)):
            if count > 0:  # 只标注有数据的柱子
                height = patch.get_height()
                x_pos = patch.get_x() + patch.get_width() / 2
                y_pos = height * 1.02 if not log_scale else height * 1.5
                
                # 自动调整标注位置避免重叠
                if log_scale and (y_pos < 10 or y_pos > 0.9 * plt.ylim()[1]):
                    continue
                    
                plt.text(
                    x_pos, y_pos,
                    f"{int(count):,}",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=0
                    # rotation=45 if bin_count > 50 else 0  # 柱子多时旋转标注
                )
        # 添加统计信息标注
        stats_text = f"""
        Mean: {np.mean(coeff_np):.2f}
        Std: {np.std(coeff_np):.2f}
        Median: {np.median(coeff_np):.2f}
        Samples: {len(coeff_np):,}
        """
        plt.gca().text(
            0.95, 0.95, stats_text,
            transform=plt.gca().transAxes,
            ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        # 标题和标签
        plt.title(f'Pos_coeff Distribution - {pkg_name}\n(n={len(coeff_np):,})', pad=20)
        plt.xlabel('Pos_coeff Value')
        plt.ylabel('Count (Log Scale)' if plt.yscale == 'log' else 'Count')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # 保存图像（自动创建目录）
        import os
        os.makedirs(save_path, exist_ok=True)
        if pos:
            pic_name = "pos_coeff"
        else:
            pic_name = "neg_coeff"
        plt.savefig(
            f"{save_path}/{pic_name}_{pkg_name}.png",
            bbox_inches='tight',  # 避免标题被截断
            facecolor='white'      # 背景色
        )
        plt.close()  # 显式关闭图形，释放内存

    @how_much_time
    def _evaluate_model(self, test_result_df, test_date_str_formatted):
        pkg_names = [row["demand_pkgname"] for row in test_result_df.select("demand_pkgname").distinct().collect()]
        # pkg_names.append("Overall")  # 添加整体评估
        results = {}  # 存储评估结果
        eps = 1e-12
        clip_max = 10.0  # 与训练时保持一致
        for pkg_name in pkg_names:
            if pkg_name == "Overall":
                filtered_df = test_result_df
            else:
                filtered_df = test_result_df.filter(F.col("demand_pkgname") == pkg_name)
            
            # 提取 (purchase, rawPrediction) 列并收集到 Driver 端
            label_pred_list = (
                filtered_df.select("real_purchase", "rawPrediction", "rawDP")
                .rdd.map(lambda row: (float(row.real_purchase), float(row.rawPrediction), float(row.rawDP)))
                .collect()
            )
            # 转换为Tensor（假设使用CPU）
            labels = torch.tensor([x[0] for x in label_pred_list])
            probs = torch.tensor([x[1] for x in label_pred_list])  # f_theta(x)
            dp_probs = torch.tensor([x[2] for x in label_pred_list])  # f_dp(x)

            # 计算pos_coeff（与训练逻辑一致）
            raw_pos_coeff = probs / (probs - 0.5 * dp_probs + eps)
            # pos_coeff = raw_pos_coeff.clamp(min=0, max=10)  # 截断到[1.0, clip_max]范围
            neg_coeff = ((1 - probs) / (1 - probs + 0.5 * dp_probs + eps))
            pos_coeff = torch.where(raw_pos_coeff < 1.0,
                                torch.tensor(10.0),
                                raw_pos_coeff)
            pos_coeff = pos_coeff.clamp(min=1.0, max=10.0)

            # 计算统计指标
            # 计算完整分布统计量
            coeff_np = pos_coeff.numpy()  # 转为NumPy便于分析
            coeff_neg = neg_coeff.numpy()  # 负系数也转为NumPy
            # 筛选labels=1的样本
            # coeff_np = coeff_np[labels.numpy() == 1.0]
            # coeff_neg = coeff_neg[labels.numpy() == 0.0]
            stats = {
                'mean': float(np.mean(coeff_np)),
                'std': float(np.std(coeff_np)),
                'min': float(np.min(coeff_np)),
                'max': float(np.max(coeff_np)),
                'median': float(np.median(coeff_np)),
                'percentile_25': float(np.percentile(coeff_np, 25)),
                'percentile_75': float(np.percentile(coeff_np, 75)),
                'skewness': float(pd.Series(coeff_np).skew()),  # 偏度
                'kurtosis': float(pd.Series(coeff_np).kurt()),   # 峰度
                'samples_gt_2': int((coeff_np > 2.0).sum()),     # 原始值>2的数量（即使不再截断）
                'samples_lt_15': int((coeff_np < 1.5).sum())      # 原始值<1的数量
            }

            # 存储结果
            results[pkg_name] = stats
            MovasLogger.log(f"""
            Package: {pkg_name}
            Pos_coeff Distribution:
            Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}
            Range: [{stats['min']:.4f}, {stats['max']:.4f}]
            Median (IQR): {stats['median']:.4f} ({stats['percentile_25']:.4f}-{stats['percentile_75']:.4f})
            Skewness/Kurtosis: {stats['skewness']:.2f}/{stats['kurtosis']:.2f}
            Samples > 2.0: {stats['samples_gt_2']} ({stats['samples_gt_2']/len(coeff_np):.1%})
            Samples < 1.5: {stats['samples_lt_15']} ({stats['samples_lt_15']/len(coeff_np):.1%})
            """)
            self.plot_pos_coeff_distribution(coeff_np, pkg_name, True)
            self.plot_pos_coeff_distribution(coeff_neg, pkg_name, False)
            # 初始化评估指标和样本数
            auc, pcoc = 0.0, 0.0
            positive_count, negative_count = 0, 0
            label_pred_list = [(x[0], x[1]) for x in label_pred_list] # 只保留前两列 (label, rawPrediction)
            if label_pred_list:
                # 计算 AUC 和 PCOC
                auc, pcoc = compute_auc_pcoc(label_pred_list)
                
                # 统计正负样本数（label == 1.0）
                positive_count = sum(1 for label, _ in label_pred_list if label == 1.0)
                negative_count = len(label_pred_list) - positive_count

            
            # 计算 IMPRESSION 和 IVR
            impression = positive_count + negative_count
            ivr = round(positive_count / impression, 6) if impression > 0 else 0.0
            results[pkg_name] = (pkg_name, auc, pcoc, positive_count, impression, ivr)

        # 过滤正样本数 < 500 的项
        filtered_results = {
            k: v for k, v in results.items() if v[3] >= 500
        }
        # 按正样本数降序排序
        sorted_results = sorted(
            filtered_results.values(),
            key=lambda x: (-x[3], x[0])
        )
        date_str = test_date_str_formatted
        # 构建表格数据
        table_data = [
            [date_str, key, round(auc, 4), round(pcoc, 4), pos, imp, round(ivr, 6)]
            for key, auc, pcoc, pos, imp, ivr in sorted_results
        ]
        # 使用 tabulate 生成表格字符串
        headers = ["DATE", "Key", "AUC", "PCOC", "PURCHASE", "IMPRESSION", "IVR"]
        table_str = tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".6f")
        MovasLogger.log('\n' + table_str + '\n')
        return

    @how_much_time
    def _run_training_loop(self):
        # 计算训练开始日期的前一天
        previous_date = self.train_start_date_dt - timedelta(days=1)
        previous_date_str = previous_date.strftime("%Y-%m-%d")
        previous_model_path = os.path.join(self.model_out_base_path, f"model_{previous_date_str}")

        # 检查前一天的模型路径是否存在
        if os.path.exists(previous_model_path):
            current_model_in_path = previous_model_path
            MovasLogger.add_log(f"Found model from previous day: {previous_model_path}")
        else:
            current_model_in_path = None
            MovasLogger.add_log(f"No model found for previous day: {previous_model_path}. Starting from empty model.")

        current_date = self.train_start_date_dt
        MovasLogger.add_log("Starting training loop.")

        while current_date <= self.train_end_date_dt:
            aux_model_date = current_date - timedelta(days=1)
            aux_model_date_str = aux_model_date.strftime("%Y-%m-%d")

            date_str = current_date.strftime("%Y-%m-%d")
            current_date7 = current_date + timedelta(days=7)
            date7_str = current_date7.strftime("%Y-%m-%d")
            MovasLogger.add_log(f"--- Training for date: {date_str} ---")

            if not os.path.exists(self.model_out_base_path):
                os.makedirs(self.model_out_base_path, exist_ok=True)

            aux_model_in_path = os.path.join(self.aux_model_in_path, f"model_{aux_model_date_str}") # {aux_model_date_str}
            model_out_path_current = os.path.join(self.model_out_base_path, f"model_{date_str}")
            real_data_path = os.path.join(self.train_path_prefix, f"sampled_{date_str}")
            train_real_dataset = self._read_dataset_by_date(self.train_path_prefix, real_data_path)
            obs_data_path = os.path.join(self.obs_train_path_prefix, f"obs={date7_str}")
            train_obs_dataset = self._read_dataset_by_date(self.obs_train_path_prefix, obs_data_path)
            train_dataset = train_real_dataset.unionByName(train_obs_dataset, allowMissingColumns=True)
            train_dataset = train_dataset.sample(withReplacement=False, fraction=1.0, seed=42)
            train_dataset = train_dataset.coalesce(200)
            self._train_model(train_dataset, current_model_in_path, aux_model_in_path, model_out_path_current, date_str)

            current_model_in_path = model_out_path_current
            current_date += timedelta(days=1)

        MovasLogger.add_log("Finished training loop.")

    @how_much_time
    def _run_evaluation_phase(self):
        if self.trained_model_path and self.validation_date: 
            MovasLogger.log('\n' + '=' * 50 + f"Evaluating model from: {self.trained_model_path} " + '=' * 50)
            # 判断self.validation_date类型为str还是datetime
            # 如果是字符串类型，转换为datetime对象
            if isinstance(self.validation_date, str):
                validation_dt = datetime.strptime(self.validation_date, "%Y-%m-%d")
            else:
                validation_dt = self.validation_date
                self.validation_date = validation_dt.strftime("%Y-%m-%d")  # 确保 validation_date 是字符串格式
            # validation_dt = datetime.strptime(self.validation_date, "%Y-%m-%d")
            aux_model_in_path = os.path.join(self.aux_model_in_path, f"model_{self.validation_date}")
            test_date = validation_dt # + timedelta(days=9)
            # test_date = self.validation_date + timedelta(days=9)
            test_date_str_formatted = test_date.strftime("%Y-%m-%d")
            test_data_path = os.path.join(self.test_path_prefix, f"obs={test_date_str_formatted}")
            test_dataset = self._read_dataset_by_date(self.test_path_prefix, test_data_path)
            test_result_df = self._transform_data(test_dataset, aux_model_in_path, self.trained_model_path)
            self._evaluate_model(test_result_df, test_date_str_formatted)
            MovasLogger.log('\n' + '=' * 50 + f"Evaluating model from: {self.trained_model_path} " + '=' * 50)
        else:
            MovasLogger.add_log(level='WARNING', content="No model was trained, skipping evaluation.")

    @how_much_time
    def _run_evaluation_manual(self, model_date, sample_date):
        self.trained_model_path = os.path.join(self.model_out_base_path, f"model_{model_date}")

        self.validation_date = sample_date

        if self.trained_model_path and self.validation_date: 
            MovasLogger.log('\n' + '=' * 50 + f"Evaluating model from: {self.trained_model_path} " + '=' * 50)
            if isinstance(self.validation_date, str):
                validation_dt = datetime.strptime(self.validation_date, "%Y-%m-%d")
            else:
                validation_dt = self.validation_date
                self.validation_date = validation_dt.strftime("%Y-%m-%d")  # 确保 validation_date 是字符串格式
            # aux model理论上可用self.validation_date-n 5(7-3+1) model（DP信息泄露、已经完成当天训练）排除DP干扰，验证主网络情况
            aux_model_in_path = os.path.join(self.aux_model_in_path, f"model_{model_date}")

            # test_date_str_formatted = self.validation_date
            # test_data_path = os.path.join(self.test_path_prefix, f"sampled_{test_date_str_formatted}")
            # test_dataset = self._read_dataset_by_date(self.test_path_prefix, test_data_path)
            validation_dt = datetime.strptime(self.validation_date, "%Y-%m-%d")
            test_date = validation_dt # + timedelta(days=9)
            # test_date = self.validation_date + timedelta(days=9)
            test_date_str_formatted = test_date.strftime("%Y-%m-%d")
            test_data_path = os.path.join(self.test_path_prefix, f"obs={test_date_str_formatted}")
            
            test_dataset = self._read_dataset_by_date(self.test_path_prefix, test_data_path)
            test_result_df = self._transform_data(test_dataset, aux_model_in_path, self.trained_model_path)
            self._evaluate_model(test_result_df, test_date_str_formatted)
            MovasLogger.log('\n' + '=' * 50 + f"Evaluating model from: {self.trained_model_path} " + '=' * 50)
        else:
            MovasLogger.add_log(level='WARNING', content="No model was trained, skipping evaluation.")

    @how_much_time
    def run_complete_flow(self, validation = False, model_date = None, sample_date = None):
        # MovasLogger.init() 在 _init_spark() 内部调用
        self._init_spark() 
        MovasLogger.add_log(content=f'Config loaded: {self.params}')
        MovasLogger.add_log(content="Starting complete flow.")
        self._build_model_module()
        if validation != True:
            self._run_training_loop()
        if validation and model_date and sample_date:
            self._run_evaluation_manual(model_date, sample_date)
        else:
            self._run_evaluation_phase()
        self._stop_spark()
        MovasLogger.add_log(content="Completed complete flow.")
        MovasLogger.save_to_local() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', required=True, help='config file path')
    parser.add_argument('--validation', type=bool, action='store', required=False, help='config validation')
    parser.add_argument('--model', type=str, action='store', required=False, help='config model date')
    parser.add_argument('--sample', type=str, action='store', required=False, help='config sample date')
    args = parser.parse_args()

    trainer = MsModelTrainFlow(config_path=args.conf)
    trainer.run_complete_flow(validation=args.validation, model_date=args.model, sample_date=args.sample) 
