#
# Copyright 2022 DMetaSoul
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# you should change to your finetuned model, `q_model` is the query model and `p_model` is the passage model
q_model=output/train_de_loss_contrastive_in_batch/2022_05_27_18_53_58/epoch_1/model1
p_model=output/train_de_loss_contrastive_in_batch/2022_05_27_18_53_58/epoch_1/model2
split_name=dev
query_file=./data/dev/dev.q.format
#split_name=train
#query_file=./data/train/train.q.format
passage_data=./data/passage-collection
topk=50

mkdir -p ./logs
nohup sh script/retrieval.sh ${q_model} ${p_model} ${query_file} ${passage_data} ${topk} ${split_name} search > logs/retrieval.log 2>&1 &
