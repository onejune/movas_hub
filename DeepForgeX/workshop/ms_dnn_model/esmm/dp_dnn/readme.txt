本部分为延迟建模中辅助网络训练部分
sample.py用于给现有样本打上dp标签，当label=1，diff_hours>24h时为延迟正例，即dp=1
imp_sample.py用于对样本进行重新排序，按照"part"曝光日期进行排序
训练脚本为run.sh，测试脚本为validation.sh，均调用dnn.py
相关配置文件为conf/config_template.yaml