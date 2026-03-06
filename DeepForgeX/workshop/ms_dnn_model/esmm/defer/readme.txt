本部分为延迟建模中主网络训练部分
运行前需要确认辅助dp网络已经训练完成，已经生成所需按曝光时间排序的样本，并在conf/defer.yaml中配置好相关样本以及模型路径aux_model_in_path
check_label用于查看每日正例、延迟正例、立即正例数量
run.sh为训练脚本
validation.sh为测试脚本
defer.py为主要metaspore代码，运行前需要确认dp_dnn目录下sample.py、imp_sample.py、run.sh均已执行完毕，并正确配置相关路径
movas_logger用于log保存
metrics_eval用于计算auc、pcoc指标
