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

import metaspore as ms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import struct

from pyspark.sql.functions import col

class ESMMAgent(ms.PyTorchAgent):
    def __init__(self,
                 ctr_loss_weight=1.0,
                 ctcvr_loss_weight=1.0,
                 step=0,
                 **kwargs):
        super().__init__()
        # self.ctr_loss = nn.BCELoss()
        # self.ctcvr_loss = nn.BCELoss()
        self.ctr_loss = nn.BCELoss()
        self.ctcvr_loss = nn.BCELoss()
        self.ctr_loss_weight = ctr_loss_weight
        self.ctcvr_loss_weight = ctcvr_loss_weight
        self.step = step

    def train_minibatch(self, minibatch):
        self.step += 1
        self.model.train()
        ndarrays, ctcvr_labels, ctr_labels = self.preprocess_minibatch(minibatch)
        ctr_predictions, cvr_predictions = self.model(minibatch)
        ctcvr_predictions = ctr_predictions * cvr_predictions
        ctcvr_labels = torch.from_numpy(ctcvr_labels).reshape(-1, 1)
        ctr_labels = torch.from_numpy(ctr_labels).reshape(-1, 1)

        loss = self.ctcvr_loss(ctcvr_predictions, ctcvr_labels) * self.ctcvr_loss_weight \
               + self.ctr_loss(ctr_predictions, ctr_labels) * self.ctr_loss_weight

        # if self.step % 100 == 0:
        #     print(f"loss: {loss}")
        #     print("==CVR Pred vs Label (only when clicked)==")

        #     mask = (ctr_labels == 1.0).squeeze()  # 变成 1D mask
        #     cvr_pred_sample = cvr_predictions[mask]
        #     cvr_label_sample = (ctcvr_labels[mask])  # 实际就是 CVR 的标签

        #     for i in range(min(10, len(cvr_label_sample))):
        #         pred = cvr_pred_sample[i].item()
        #         label = cvr_label_sample[i].item()
        #         print(f"[{i}] pred: {pred:.6f}, label: {label}")

        #     print("==CTR Pred vs Label==")
        #     for i in range(min(10, len(ctr_labels))):
        #         pred = ctr_predictions[i].item()
        #         label = ctr_labels[i].item()
        #         print(f"[{i}] pred: {pred:.6f}, label: {label}")
        self.trainer.train(loss)
        # print("Training step completed.")
        self.update_progress(ctcvr_predictions, ctcvr_labels, loss)
        return minibatch

    def validate_minibatch(self, minibatch):
        self.model.eval()
        ndarrays, ctcvr_labels, ctr_labels = self.preprocess_minibatch(minibatch)
        ctr_predictions, cvr_predictions = self.model(minibatch)
        ctcvr_predictions = cvr_predictions * ctr_predictions
        ctcvr_labels = torch.from_numpy(ctcvr_labels).reshape(-1, 1)
        ctr_labels = torch.from_numpy(ctr_labels).reshape(-1, 1)
        # print(f"ctcvr_labels shape: {ctcvr_labels.shape}, ctcvr_predictions shape: {ctcvr_predictions.shape}")
        # print(f"ctr_labels shape: {ctr_labels.shape}, ctr_predictions shape: {ctr_predictions.shape}")
        # print(f"ctcvr_labels: {ctcvr_labels[:5]}, ctcvr_predictions: {ctcvr_predictions[:5]}")
        # print(f"ctr_labels: {ctr_labels[:5]}, ctr_predictions: {ctr_predictions[:5]}")
        # print("Val check:")
        # input()
        self.update_progress(ctcvr_predictions, ctcvr_labels, torch.tensor(0.0))
        return self._make_validation_result(minibatch, ctcvr_labels, ctcvr_predictions, ctr_labels, ctr_predictions, cvr_predictions)
        # print("Validation step completed.")
        # input()
        # return ctcvr_predictions.detach().reshape(-1), \
        #        ctr_predictions.detach().reshape(-1), \
        #        cvr_predictions.detach().reshape(-1)
    
    def _make_validation_result(self, minibatch, labels, predictions, ctr_labels, ctr_predictions, cvr_predictions):
        labels = labels.reshape(-1).numpy().astype('double')
        predictions = predictions.detach().reshape(-1).numpy().astype('double')
        minibatch[self.output_label_column_name] = labels
        minibatch[self.output_prediction_column_name] = predictions
        ctr_labels = ctr_labels.reshape(-1).numpy().astype('double')
        ctr_predictions = ctr_predictions.detach().reshape(-1).numpy().astype('double')
        minibatch[self.output_ctr_label_column_name] = ctr_labels
        minibatch[self.output_ctr_prediction_column_name] = ctr_predictions
        cvr_predictions = cvr_predictions.detach().reshape(-1).numpy().astype('double')
        minibatch[self.output_cvr_prediction_column_name] = cvr_predictions
        return minibatch
    
    def _make_validation_result_schema(self, df):
        from pyspark.sql.types import StructType
        fields = []
        reserved = set([self.output_label_column_name, self.output_prediction_column_name, \
                        self.output_ctr_label_column_name, self.output_ctr_prediction_column_name,\
                            self.output_cvr_prediction_column_name])
        for field in df.schema.fields:
            if field.name not in reserved:
                fields.append(field)
        result_schema = StructType(fields)
        result_schema.add(self.output_label_column_name, self.output_label_column_type)
        result_schema.add(self.output_prediction_column_name, self.output_prediction_column_type)
        result_schema.add(self.output_ctr_label_column_name, self.output_label_column_type)
        result_schema.add(self.output_ctr_prediction_column_name, self.output_prediction_column_type)
        result_schema.add(self.output_cvr_prediction_column_name, self.output_prediction_column_type)
        return result_schema

    def preprocess_minibatch(self, minibatch):
        if isinstance(minibatch, tuple):
        # 创建DataFrame，列名应与self.dataset.columns一致
            minibatch = pd.DataFrame({
                col: series for col, series in zip(self.dataset.columns, minibatch)
            })
        
        # 确保minibatch是DataFrame
        if not isinstance(minibatch, pd.DataFrame):
            raise ValueError(f"Expected pandas.DataFrame, got {type(minibatch)}")

        ctcvr_labels = minibatch[self.output_cvr_label_column_name].values.astype(np.float32)
        ctr_labels = minibatch[self.output_ctr_label_column_name].values.astype(np.float32)

        return minibatch, ctcvr_labels, ctr_labels

    def _create_metric(self):
        metric = ESMMMetric()
        return metric

    def update_metric(self, predictions, labels, loss):
        self._metric.accumulate(predictions.data.numpy(), labels.data.numpy(), loss.data.numpy())

    def update_progress(self, predictions, labels, loss):
        self.minibatch_id += 1
        self.update_metric(predictions, labels, loss)
        # print(f"Minibatch ID: {self.minibatch_id}, Metric update interval: {self.metric_update_interval}")
        if self.minibatch_id % self.metric_update_interval == 0:
            # print("Pushing metric...")
            self.push_metric()
            # print("Metric pushed successfully.")

    def handle_request(self, req):
        import json
        body = json.loads(req.body)
        command = body.get('command')
        if command == 'PushMetric':
            states = ()
            for i in range(req.slice_count):
                states += req.get_slice(i),
            accum = self._metric
            delta = ESMMMetric.from_states(states)
            accum.merge(delta)
            from datetime import datetime
            string = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            string += f' -- auc: {accum.compute_auc()}'
            string += f', \u0394auc: {delta.compute_auc()}'
            string += f', loss: {accum.compute_loss()}'
            string += f', \u0394loss: {delta.compute_loss()}'
            string += f', pcoc: {accum.compute_pcoc()}'
            string += f', \u0394pcoc: {delta.compute_pcoc()}'
            string += f', #instance: {accum.instance_count}'
            if accum.threshold > 0.0:
                string += f', accuracy: {accum.compute_accuracy()}'
                string += f', precision: {accum.compute_precision()}'
                string += f', recall: {accum.compute_recall()}'
                string += f', F{accum.beta:g}_score: {accum.compute_f_score()}'
            print(string)
            from metaspore._metaspore import Message
            res = Message()
            self.send_response(req, res)
            return
        super().handle_request(req)

class ESMMMetric(ms.BinaryClassificationModelMetric):
    def __init__(self, buffer_size=1000000, threshold=0.0, beta=1.0):
        super().__init__(buffer_size=buffer_size, threshold=threshold, beta=beta)
        # super().__init__()  # 父类没这些参数
        # self.buffer_size = buffer_size
        # self.threshold = threshold
        # self.beta = beta
        self._loss = 0

    def clear(self):
        super().clear()
        self._loss = 0

    def merge(self, other):
        super().merge(other)
        self._loss += other._loss

    def accumulate(self, predictions, labels, loss):
        super().accumulate(predictions=predictions, labels=labels, batch_size=len(labels), batch_loss=loss.sum() * len(labels))
        self._loss += loss.sum() * len(labels)

    def compute_loss(self):
        if self._instance_num==0:
            return float('nan')
        return self._loss / self._instance_num

    def _get_pack_format(self):
        return 'ddl' + 'l' * 4 + 'd'

    def get_states(self):
        scalars = self._prediction_sum,
        scalars += self._label_sum,
        scalars += self._instance_num,
        scalars += self._true_positive,
        scalars += self._true_negative,
        scalars += self._false_positive,
        scalars += self._false_negative,
        scalars += self._loss,
        scalars = struct.pack(self._get_pack_format(), *scalars)
        scalars = np.array(tuple(scalars), dtype=np.uint8)
        states = scalars,
        states += self._positive_buffer,
        states += self._negative_buffer,
        return states

    @classmethod
    def from_states(cls, states):
        scalars, pos_buf, neg_buf = states
        buffer_size = len(pos_buf)
        inst = cls(buffer_size)
        inst._positive_buffer[:] = pos_buf
        inst._negative_buffer[:] = neg_buf
        pred_sum, lab_sum, inst_num, tp, tn, fp, fn, loss = struct.unpack(inst._get_pack_format(), scalars)
        inst._prediction_sum = pred_sum
        inst._label_sum = lab_sum
        inst._instance_num = inst_num
        inst._true_positive = tp
        inst._true_negative = tn
        inst._false_positive = fp
        inst._false_negative = fn
        inst._loss = loss
        return inst
