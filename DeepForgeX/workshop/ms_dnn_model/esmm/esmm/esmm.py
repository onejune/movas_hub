import metaspore as ms
import pyspark
import numpy as np
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
from metaspore.algos.multitask import ESMM
from metaspore.algos.multitask import ESMMAgent

from pyspark.sql.functions import rand
from datetime import datetime, timedelta
from tabulate import tabulate

# movas_logger.py 与 dnn_ms.py 在同一目录下
from movas_logger import MovasLogger, how_much_time

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
                         'server_count', 'embedding_size', 'column_name_path', 
                         'combine_schema_path', 'dnn_hidden_units',
                         'ftrl_l1', 'ftrl_l2', 'ftrl_alpha', 'ftrl_beta', 'adam_learning_rate',
                         'batch_size', 'app_name', 'local_spark', 'worker_memory', 'server_memory',
                         'coordinator_memory', 'experiment_name',
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
        if not hasattr(self, 'movas_log_output_path'):
             self.movas_log_output_path = self.params.get('movas_log_output_path', 'dnn_ms_movas_log.txt')


    def _init_spark(self):
        MovasLogger.add_log(content='Debug -- spark init') # MovasLogger 在此之前应已可用，但 init() 在之后
                                                        # 这意味着 MovasLogger 的静态方法 add_log 必须能在 init 前被调用
                                                        # 或者将此日志移到 MovasLogger.init() 之后
        spark_confs = {
            "spark.eventLog.enabled": "false",
            "spark.driver.memory": "20g",
            "spark.executor.memory": "160g",
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
    def _read_dataset_by_date(self, base_path, date_str):
        data_path = os.path.join(base_path, f"sample_{date_str}")
        MovasLogger.add_log(content=f"Reading Parquet data from directory: {data_path}")
        df = self.spark_session.read.parquet(data_path)
        #df = spark.read.csv(train_path, sep='\002', header=False)
        #with open(self.column_name_path, "r") as f:
        #    column_names = [line.strip().split()[1] for line in f if line.strip()]
        #    df = df.toDF(*column_names)
        # df.printSchema()  # 检查各列的实际数据类型
        
        for col_name in df.columns:
            if col_name == 'purchase' or col_name == 're_engagement':
                df = df.withColumn(col_name, F.col(col_name).cast("float"))
            else:
                df = df.withColumn(col_name, F.col(col_name).cast("string"))
        df = self.random_sample(df)
        df = df.filter(F.col("demand_pkgname") == 'COM.ZZKKO')
        df = df.filter(F.col("purchase").isin([0, 1]))
        df = df.filter(F.col("re_engagement").isin([0, 1]))
        df = df.fillna('unknown')
        # df.show(1)
        # input()
        return df

    def _build_model_module(self):
        self.model_module = ESMM(embedding_dim=self.embedding_size,
                  column_name_path=self.column_name_path,
                  combine_schema_path=self.combine_schema_path,
                  sparse_init_var=self.sparse_init_var,
                  dnn_hidden_units=self.dnn_hidden_units,
                  dnn_hidden_activations=self.dnn_hidden_activations,
                  use_bias=self.use_bias,
                  batch_norm=self.batch_norm,
                  net_dropout=self.net_dropout,
                  net_regularizer=self.net_regularizer,
                  ftrl_l1=self.ftrl_l1,
                  ftrl_l2=self.ftrl_l2,
                  ftrl_alpha=self.ftrl_alpha,
                  ftrl_beta=self.ftrl_beta)
        # self.model_module = WideDeep(use_wide=False,
        #                         batch_norm=False, 
        #                         wide_embedding_dim=self.embedding_size,
        #                         deep_embedding_dim=self.embedding_size,
        #                         wide_column_name_path=self.column_name_path, 
        #                         wide_combine_schema_path=self.wide_combine_schema_path,
        #                         deep_column_name_path=self.column_name_path,
        #                         deep_combine_schema_path=self.combine_schema_path,
        #                         dnn_hidden_units=self.dnn_hidden_units,
        #                         dnn_hidden_activations="dice",
        #                         ftrl_l1=self.ftrl_l1,
        #                         ftrl_l2=self.ftrl_l2,
        #                         ftrl_alpha=self.ftrl_alpha,
        #                         ftrl_beta=self.ftrl_beta)
        # print(self.module)
        MovasLogger.add_log(content="Model module built.")

    #@how_much_time
    def _train_model(self, train_dataset, model_in_path_current, model_out_path_current, model_version_current):
        if not self.model_module:
            self._build_model_module()
        print(self.model_module)
        # input()
        estimator = ms.PyTorchEstimator(module=self.model_module,
                                      agent_class=ESMMAgent,
                                      worker_count=self.worker_count,
                                      server_count=self.server_count,
                                      model_in_path=model_in_path_current,
                                      model_out_path=model_out_path_current,
                                      model_export_path=None, 
                                      model_version=model_version_current,
                                      experiment_name=self.experiment_name,
                                      ctr_loss_weight=self.ctr_loss_weight,
                                      ctcvr_loss_weight=self.ctcvr_loss_weight,
                                      output_label_column_name='purchase',
                                      output_prediction_column_name='rawPrediction',
                                      input_label_column_name= 'purchase',
                                      output_ctr_label_column_name='re_engagement',
                                      output_cvr_label_column_name='purchase',
                                      output_ctr_prediction_column_name='rawCTRPrediction',
                                      output_cvr_prediction_column_name='rawCVRPrediction',
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
    def _transform_data(self, dataset_to_transform, model_in_path_current):
        if not self.model_module:
            self._build_model_module()
        
        model_transformer = ms.PyTorchModel(module=self.model_module,
                                agent_class=ESMMAgent,
                                worker_count=self.worker_count,
                                server_count=self.server_count,
                                model_in_path=model_in_path_current, 
                                experiment_name=self.experiment_name,
                                input_label_column_name= 'purchase',
                                output_label_column_name='purchase',
                                output_ctr_label_column_name='re_engagement',
                                output_cvr_label_column_name='purchase',
                                output_ctr_prediction_column_name='rawCTRPrediction',
                                output_cvr_prediction_column_name='rawCVRPrediction',
                                output_raw_prediction_column_name='rawPrediction',)
                                # input_label_column_index=self.input_label_column_index)
        
        MovasLogger.add_log(content=f"Transforming data using model from: {model_in_path_current}")
        test_result = model_transformer.transform(dataset_to_transform)
        #MovasLogger.add_log(content=f"Test result sample:\n{MovasLogger.get_df_showString(test_result, lines=5)}")
        return test_result

    @how_much_time
    def _evaluate_model(self, test_result_df, test_date_str_formatted):
        pkg_names = [row["demand_pkgname"] for row in test_result_df.select("demand_pkgname").distinct().collect()]
        results = {}

        for pkg_name in pkg_names:
            filtered_df = test_result_df.filter(F.col("demand_pkgname") == pkg_name)

            # --------- 1. CTCVR (purchase vs rawPrediction) ----------
            label_pred_list = (
                filtered_df.select("purchase", "rawPrediction")
                .rdd.map(lambda row: (float(row.purchase), float(row.rawPrediction)))
                .collect()
            )
            auc, pcoc = 0.0, 0.0
            positive_count, negative_count = 0, 0
            if label_pred_list:
                auc, pcoc = compute_auc_pcoc(label_pred_list)
                positive_count = sum(1 for label, _ in label_pred_list if label == 1.0)
            impression = len(label_pred_list)

            # --------- 2. CTR (re_engagement vs rawCTRPrediction) ----------
            re_label_pred_list = (
                filtered_df.select("re_engagement", "rawCTRPrediction")
                .rdd.map(lambda row: (float(row.re_engagement), float(row.rawCTRPrediction)))
                .collect()
            )
            re_auc, re_pcoc = 0.0, 0.0
            re_positive_count = 0
            if re_label_pred_list:
                re_auc, re_pcoc = compute_auc_pcoc(re_label_pred_list)
                re_positive_count = sum(1 for label, _ in re_label_pred_list if label == 1.0)
            re_impression = len(re_label_pred_list)

            # --------- 3. CVR (purchase vs rawPrediction in re==1 subset) ----------
            cvr_df = filtered_df.filter((F.col("re_engagement") == 1) | (F.col("purchase") == 1))
            cvr_label_pred_list = (
                cvr_df.select("purchase", "rawCVRPrediction")
                .rdd.map(lambda row: (float(row.purchase), float(row.rawCVRPrediction)))
                .collect()
            )
            cvr_auc, cvr_pcoc = 0.0, 0.0
            cvr_positive_count = 0
            if cvr_label_pred_list:
                cvr_auc, cvr_pcoc = compute_auc_pcoc(cvr_label_pred_list)
                cvr_positive_count = sum(1 for label, _ in cvr_label_pred_list if label == 1.0)
            cvr_impression = len(cvr_label_pred_list)
            if impression != re_impression:
                MovasLogger.add_log(level='WARNING', content=f"Impression mismatch for package {pkg_name}: CTCVR={impression}, CTR={re_impression}")
            if positive_count != cvr_positive_count:
                MovasLogger.add_log(level='WARNING', content=f"Positive count mismatch for package {pkg_name}: CTCVR={positive_count}, CVR={cvr_positive_count}")
            # --------- 记录指标 ----------
            results[pkg_name] = (
                pkg_name,
                auc, pcoc, positive_count, impression,
                re_auc, re_pcoc, re_positive_count, re_impression,
                cvr_auc, cvr_pcoc, cvr_positive_count, cvr_impression
            )

        # 只保留 CTCVR 正样本数 ≥ 500 的包
        filtered_results = {
            k: v for k, v in results.items() if v[3] >= 500
        }
        sorted_results = sorted(filtered_results.values(), key=lambda x: (-x[3], x[0]))
        date_str = test_date_str_formatted

        # --------- 打印表格 ----------
        table_data = [
            [
                date_str, key,
                round(auc, 4), round(pcoc, 4),
                round(re_auc, 4), round(re_pcoc, 4),
                round(cvr_auc, 4), round(cvr_pcoc, 4), pos, re_pos, imp
            ]
            for key, auc, pcoc, pos, imp,
                re_auc, re_pcoc, re_pos, re_imp,
                cvr_auc, cvr_pcoc, cvr_pos, cvr_imp in sorted_results
        ]

        headers = [
            "DATE", "Key",
            "CTCVR_AUC", "CTCVR_PCOC",
            "CTR_AUC", "CTR_PCOC",
            "CVR_AUC", "CVR_PCOC", "PURCHASE", "CLICK", "IMPRESSION"
        ]

        table_str = tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".5f")
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
            date_str = current_date.strftime("%Y-%m-%d")
            MovasLogger.add_log(f"--- Training for date: {date_str} ---")

            if not os.path.exists(self.model_out_base_path):
                os.makedirs(self.model_out_base_path, exist_ok=True)

            model_out_path_current = os.path.join(self.model_out_base_path, f"model_{date_str}")

            train_dataset = self._read_dataset_by_date(self.train_path_prefix, date_str)
            self._train_model(train_dataset, current_model_in_path, model_out_path_current, date_str)

            current_model_in_path = model_out_path_current
            current_date += timedelta(days=1)

        MovasLogger.add_log("Finished training loop.")

    @how_much_time
    def _run_evaluation_phase(self):
        if self.trained_model_path and self.validation_date: 
            MovasLogger.log('\n' + '=' * 50 + f"Evaluating model from: {self.trained_model_path} " + '=' * 50)
            test_date_str_formatted = self.validation_date.strftime("%Y-%m-%d")
            test_dataset = self._read_dataset_by_date(self.train_path_prefix, test_date_str_formatted)
            test_result_df = self._transform_data(test_dataset, self.trained_model_path)
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
            test_date_str_formatted = self.validation_date
            test_dataset = self._read_dataset_by_date(self.train_path_prefix, test_date_str_formatted)
            test_result_df = self._transform_data(test_dataset, self.trained_model_path)
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
