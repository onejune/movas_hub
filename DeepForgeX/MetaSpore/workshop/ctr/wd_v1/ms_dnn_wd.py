import metaspore as ms
import pyspark
import numpy as np
import pandas as pd
import yaml
import subprocess
import argparse
import sys
import os, shutil
from operator import itemgetter
from pyspark.sql.types import StringType
from pyspark.sql import functions as F
from metrics_eval import compute_auc_pcoc, compute_auc_pcoc2, calculate_logloss
sys.path.append('../MetaSpore/')
from metaspore.algos.widedeep_net import WideDeep
from metaspore.algos.deepfm_net import DeepFM
from metaspore.algos.ffm_net import FFM
from metaspore.algos.dcn_net import DCN

from pyspark.sql.functions import rand
from datetime import datetime, timedelta
from tabulate import tabulate
from collections import namedtuple

# movas_logger.py 与 dnn_ms.py 在同一目录下
from movas_logger import MovasLogger, how_much_time
from feishu_notifier import FeishuNotifier

class MsModelTrainFlow:
    def __init__(self, config_path):
        self.config_path = config_path
        self.params = self._load_config(config_path)
        self._load_combine_schema()
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
                         'combine_schema_path', 'wide_combine_schema_path', 'dnn_hidden_units',
                         'ftrl_l1', 'ftrl_l2', 'ftrl_alpha', 'ftrl_beta', 'adam_learning_rate',
                         'batch_size', 'app_name', 'local_spark', 'worker_memory', 'server_memory',
                         'coordinator_memory', 'input_label_column_index', 'experiment_name',
                         'train_start_date', 'train_end_date', 'validation_date', 'movas_log_output_path'] # 确保 movas_log_output_path 在配置中
        for key in required_keys:
            assert key in params, f'Missing required config: {key}'
        # 读取./readme文件的全部内容
        self.exp_readme = ''
        try:
            with open('./readme', 'r', encoding='utf-8') as file:
                self.exp_readme = '实验信息:\n' + file.read()
        except Exception as e:
            pass
        return params

    def _load_combine_schema(self):
        fin = open('./conf/combine_schema', 'r')
        all_feas_list = []
        for line in fin:
            arr = line.strip().split('#')
            all_feas_list.extend(arr)
        all_feas_list = list(set(all_feas_list))
        le = len(all_feas_list)
        MovasLogger.add_log(content=f'Debug--load combine_schema: {le} features!')
        all_feas_list = all_feas_list + ['label']
        self.used_fea_list = all_feas_list
        return

    def _initialize_params(self):
        for key, value in self.params.items():
            setattr(self, key, value)
        self.train_start_date_dt = self.train_start_date
        self.train_end_date_dt = self.train_end_date
        # self.movas_log_output_path 已通过 setattr 设置，如果配置文件中有的话
        # 如果配置文件中没有 movas_log_output_path，需要提供一个默认值或确保它存在
        if not hasattr(self, 'movas_log_output_path'):
            self.movas_log_output_path = self.params.get('movas_log_output_path', 'dnn_ms_movas_log.txt')
        if not hasattr(self, 'use_wide'):
            self.use_wide = True
        if not hasattr(self, 'batch_norm'):
            self.batch_norm = False
        if not hasattr(self, 'net_dropout'):
            self.net_dropout = None

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
            subprocess.run(['zip', '-r', 'python.zip', '../MetaSpore/python'], cwd='.', check=True)
            spark_confs["spark.submit.pyFiles"] = "python.zip"
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
        df_pos = df[df['label'] == 1]
        df_neg = df[df['label'] == 0]

        # 对负样本进行随机采样10%
        df_neg_sampled = df_neg.sample(fraction=0.1, seed=42)

        # 合并正样本和采样后的负样本
        df_balanced = df_pos.union(df_neg_sampled)

        # 洗牌数据集
        df_balanced = df_balanced.orderBy(rand(seed=29))
        '''

        df_filtered = df.filter(
            (F.col("label") == 1) |
            (F.rand(seed=42) < 0.1)
        )
        return df_filtered
 
    #@how_much_time
    def _read_dataset_by_date(self, base_path, date_str):
        data_path = os.path.join(base_path, f"part={date_str}")
        MovasLogger.add_log(content=f"Reading Parquet data from directory: {data_path}")
        df = self.spark_session.read.parquet(data_path)
        df = df.select(*self.used_fea_list) #important：用 combine schema 过滤一遍 col，优化性能
        
        for col_name in df.columns:
            if col_name == 'label':
                df = df.withColumn(col_name, F.col(col_name).cast("float"))
            else:
                df = df.withColumn(col_name, F.col(col_name).cast("string"))
        df = self.random_sample(df)

        #df = df.filter(F.col("label").isin([0, 1]))
        df = df.fillna('unknown') #important: 会报错！
        MovasLogger.add_log(content=f"Reading Parquet over!!!")
        return df

    def _build_model_module(self):
        # 获取模型类型，默认为 WideDeep
        configed_model = self.params.get('model_type', "WideDeep")
        MovasLogger.add_log(content=f"Building model module with configed_model: {configed_model}")
        
        # 根据模型类型构建不同的模型
        if configed_model == "DeepFM":
            self.model_module = DeepFM(
                use_wide=self.use_wide,
                batch_norm=self.batch_norm,
                net_dropout=self.net_dropout,
                wide_embedding_dim=self.embedding_size,
                deep_embedding_dim=self.embedding_size,
                wide_combine_schema_path=self.wide_combine_schema_path,
                deep_combine_schema_path=self.combine_schema_path,
                dnn_hidden_units=self.dnn_hidden_units,
                dnn_hidden_activations="relu",
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta
            )
        elif configed_model == "WideDeep":
            self.model_module = WideDeep(
                use_wide=self.use_wide,
                batch_norm=self.batch_norm,
                net_dropout=self.net_dropout,
                wide_embedding_dim=self.embedding_size,
                deep_embedding_dim=self.embedding_size,
                wide_combine_schema_path=self.wide_combine_schema_path,
                deep_combine_schema_path=self.combine_schema_path,
                dnn_hidden_units=self.dnn_hidden_units,
                dnn_hidden_activations="relu",
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta
            )
        elif configed_model == "FFM":
            self.model_module = FFM(
                wide_embedding_dim=self.embedding_size,
                deep_embedding_dim=self.embedding_size,
                wide_combine_schema_path=self.wide_combine_schema_path,
                deep_combine_schema_path=self.combine_schema_path
            )
        elif configed_model == "DCN":
            self.model_module = DCN(
                use_wide=self.use_wide,
                batch_norm=self.batch_norm,
                net_dropout=self.net_dropout,
                wide_embedding_dim=self.embedding_size,
                deep_embedding_dim=self.embedding_size,
                wide_combine_schema_path=self.wide_combine_schema_path,
                deep_combine_schema_path=self.combine_schema_path,
                dnn_hidden_units=self.dnn_hidden_units,
                dnn_activations="relu",
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta
            )
        else:
            raise ValueError(f"Unsupported model type: {configed_model}. "
                            f"Supported types: DeepFM, WideDeep, FM, DNN")

    @how_much_time
    def _train_model(self, train_dataset, model_in_path_current, model_out_path_current, model_version_current):
        if not self.model_module:
            self._build_model_module()
            
        estimator = ms.PyTorchEstimator(module=self.model_module,
                                      worker_count=self.worker_count,
                                      server_count=self.server_count,
                                      model_in_path=model_in_path_current,
                                      model_out_path=model_out_path_current,
                                      model_export_path=None, 
                                      model_version=model_version_current,
                                      experiment_name=self.experiment_name,
                                      # input_label_column_index=self.input_label_column_index,
                                      input_label_column_name='label',
                                      metric_update_interval=1000)
        estimator.updater = ms.AdamTensorUpdater(self.adam_learning_rate)
        
        MovasLogger.add_log(content=f"Starting training for version: {model_version_current}, Model In: {model_in_path_current}, Model Out: {model_out_path_current}")
        model = estimator.fit(train_dataset)
        self.trained_model_path = model_out_path_current 
        return model
        
    @how_much_time
    def _predict_data(self, dataset_to_transform, model_in_path_current):
        if not self.model_module:
            self._build_model_module()

        model_transformer = ms.PyTorchModel(module=self.model_module,
                                worker_count=self.worker_count,
                                server_count=self.server_count,
                                model_in_path=model_in_path_current, 
                                experiment_name=self.experiment_name,
                                input_label_column_name='label')

        MovasLogger.add_log(content=f"Transforming data using model from: {model_in_path_current}")
        test_result = model_transformer.transform(dataset_to_transform)
        #MovasLogger.add_log(content=f"Test result sample:\n{MovasLogger.get_df_showString(test_result, lines=5)}")
        return test_result

    @how_much_time
    def _evaluate_model(self, test_result_df, test_date_str_formatted):
        MovasLogger.log(f"Evaluating model from: {self.trained_model_path}, eval_keys: {self.eval_keys} ")
        #指定评估列名，通过命令行传递，没有列名，则使用demand_pkgname
        if not self.eval_keys:
            self.eval_keys = "demand_pkgname"
        eval_col_list = self.eval_keys.split(",")
        eval_col_list.append("Overall")
        results = {}  # 存储评估结果

        for col_name in eval_col_list:
            if col_name == "Overall": # 整体评估
                filtered_df = test_result_df
                self._eval_by_df("Overall", col_name, filtered_df, results)
            else: # 按指定的特征评估
                col_values = [row[col_name] for row in test_result_df.select(col_name).distinct().collect()] 
                for col_value in col_values:
                    filtered_df = test_result_df.filter(F.col(col_name) == col_value)
                    self._eval_by_df(col_name, col_value, filtered_df, results)

        filtered_results = {
            k: v for k, v in results.items() if v.pos >= 100
        }
        
        sorted_results = sorted(filtered_results.values(), key=lambda x: (x.key1, -x.neg))
        tag = "val-" + self.name + '-' + test_date_str_formatted
        
        # 构建表格数据
        table_data = [
            [tag, result.key1, result.key2, round(result.auc, 4), round(result.pcoc, 4), round(result.loss, 4),
            result.pos, result.neg, round(result.ivr, 4)]
            for result in sorted_results
        ]
        
        # 使用 tabulate 生成表格字符串，结果写入日志
        headers = ["Date", "Key1", "Key2", "AUC", "PCOC", "LogLoss", "Pos", "Neg", "Ivr"]
        table_str = tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f")
        MovasLogger.log('\n' + table_str + '\n' + self.exp_readme + '\n')
        #************************************* 发飞书消息 ******************************************
        # 使用 zip 将 headers 和 values 组合成字典，然后转成 df，结果发送飞书消息
        formatted_results = [dict(zip(headers, row)) for row in table_data]
        df = pd.DataFrame(formatted_results)
        now = datetime.now()
        formatted_dtm = now.strftime("%Y-%m-%d %H:%M:%S")
        msg_text = f"send_time: {formatted_dtm}\n{self.exp_readme}" # 表格以外的消息正文
        table_title = "Validation Result" #表格的标题
        msg_title = f"{self.name} [{test_date_str_formatted}]" #消息的标题
        FeishuNotifier.send_dataframe_html(df, title = table_title, subject=msg_title, text = msg_text)
        return  sorted_results

    def _eval_by_df(self, col_name, col_value, filtered_df, result_dict):
        EvalResult = namedtuple('EvalResult', ['key1', 'key2', 'auc', 'pcoc', 'loss', 'pos', 'neg', 'ivr'])
        # 提取 (label, rawPrediction) 列并收集到 Driver 端
        label_pred_list = (
            filtered_df.select("label", "rawPrediction")
            .rdd.map(lambda row: (float(row.label), float(row.rawPrediction)))
            .collect()
        )
        # 初始化评估指标和样本数
        auc, pcoc, logloss = 0.0, 0.0, 0.0
        positive_count, negative_count = 0, 0
        
        if label_pred_list:
            # 计算 AUC 和 PCOC
            auc, pcoc = compute_auc_pcoc(label_pred_list)
            logloss = calculate_logloss(label_pred_list)
            # 统计正负样本数（label == 1.0）
            positive_count = sum(1 for label, _ in label_pred_list if label == 1.0)
            negative_count = len(label_pred_list) - positive_count

        # 计算 IMPRESSION 和 IVR
        impression = positive_count + negative_count
        ivr = round(positive_count / impression, 6) if impression > 0 else 0.0
        
        # 使用命名元组存储结果（更清晰）
        result_dict[col_value] = EvalResult(
            key1=col_name,
            key2=col_value,
            auc=auc,
            pcoc=pcoc,
            loss=logloss,
            pos=positive_count,
            neg=impression,
            ivr=ivr
        )

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
            self.delete_directories_before_date(self.model_out_base_path, date_str)
            current_date += timedelta(days=1)

        MovasLogger.add_log("Finished training loop.")
        #FeishuNotifier.notify(f"DNN Model Training Completed: {self.name}")

    @how_much_time
    def _run_evaluation_phase(self):
        if self.trained_model_path and self.validation_date: 
            MovasLogger.log('\n' + '=' * 50 + f"Evaluating model from: {self.trained_model_path} " + '=' * 50)
            test_date_str_formatted = self.validation_date.strftime("%Y-%m-%d")
            test_dataset = self._read_dataset_by_date(self.train_path_prefix, test_date_str_formatted)
            test_result_df = self._predict_data(test_dataset, self.trained_model_path)
            self._evaluate_model(test_result_df, test_date_str_formatted)
            MovasLogger.log('\n' + '=' * 50 + f"Evaluating model from: {self.trained_model_path} " + '=' * 50)
            #FeishuNotifier.notify(f"DNN Model Evaluation Completed for {self.name}, model_path:{self.trained_model_path}")
        else:
            MovasLogger.add_log(level='WARNING', content="No model was trained, skipping evaluation.")

    @how_much_time
    def _run_evaluation_manual(self, model_date, sample_date):
        self.trained_model_path = os.path.join(self.model_out_base_path, f"model_{model_date}")
        self.validation_date = sample_date
        if self.trained_model_path and self.validation_date: 
            MovasLogger.log('\n' + '=' * 20 + f"Evaluating model for: {self.trained_model_path}, eval date: {sample_date} " + '=' * 20)
            test_date_str_formatted = self.validation_date
            test_dataset = self._read_dataset_by_date(self.train_path_prefix, test_date_str_formatted)
            test_result_df = self._predict_data(test_dataset, self.trained_model_path)
            self._evaluate_model(test_result_df, test_date_str_formatted)
            MovasLogger.log('\n' + '=' * 20 + f"Evaluating over for: {self.trained_model_path} " + '=' * 20)
        else:
            MovasLogger.add_log(level='WARNING', content="No model was trained, skipping evaluation.")

    @how_much_time
    def run_complete_flow(self, validation = False, model_date = None, sample_date = None, name = None, eval_keys = None):
        # MovasLogger.init() 在 _init_spark() 内部调用
        self._init_spark() 
        MovasLogger.add_log(content=f'Config loaded: {self.params}')
        MovasLogger.add_log(content="Starting complete flow.")

        self.eval_keys = eval_keys
        self.name = name

        self._build_model_module()
        if validation != True:
            self._run_training_loop()
        if validation and model_date and sample_date:
            self._run_evaluation_manual(model_date, sample_date)
        else:
            self._run_evaluation_phase()
        self._stop_spark()
        MovasLogger.add_log(content="Completed complete flow.")

    def delete_directories_before_date(self, directory, specified_date_str, days = 3):
        try:
            specified_date = datetime.strptime(specified_date_str, '%Y-%m-%d')
        except ValueError:
            print(f"Invalid date format: {specified_date_str}. Please use YYYY-MM-DD format.")
            return
        
        target_date = specified_date - timedelta(days=days)
        target_date_str = target_date.strftime('%Y-%m-%d')
        
        # 构建目标文件名
        target_filename = f"model_{target_date_str}"
        file_path = os.path.join(directory, target_filename)
        
        # 检查目录是否存在并删除
        if os.path.exists(file_path) and os.path.isdir(file_path):
            shutil.rmtree(file_path)
            print(f"Deleted {file_path} (Date: {target_date_str})")
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', required=True, help='config file path')
    parser.add_argument('--name', type=str, action='store', required=True, help='project name')
    parser.add_argument('--validation', type=bool, action='store', required=False, help='config validation')
    parser.add_argument('--model', type=str, action='store', required=False, help='config model date')
    parser.add_argument('--sample', type=str, action='store', required=False, help='config sample date')
    parser.add_argument('--eval_keys', type=str, action='store', required=True, help='eval keys')
    args = parser.parse_args()

    trainer = MsModelTrainFlow(config_path=args.conf)
    try:
        trainer.run_complete_flow(
            validation=args.validation, 
            model_date=args.model, 
            sample_date=args.sample, 
            name=args.name,
            eval_keys=args.eval_keys) 
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    MovasLogger.save_to_local()
