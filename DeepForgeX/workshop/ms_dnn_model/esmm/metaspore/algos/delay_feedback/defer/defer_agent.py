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

class DEFERAgent(ms.PyTorchAgent):
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
    
    # def defer_loss(self, outputs, labels, eps=1e-8):
    #     # Clamp probs for numerical stability
    #     probs = outputs[0].clamp(eps, 1 - eps)  # f_theta(x)
    #     dp_prob = outputs[1].clamp(eps, 1 - eps).detach()  # f_dp(x), detached
    #     # dp_prob 0.1采样校正
    #     # dp_prob = 0.1 * dp_prob / (1 - 0.9 * dp_prob)  # 0.1采样校正
    #     # dp截断
    #     dp_prob = torch.minimum(dp_prob, torch.tensor(2.0/3)*probs)
    #     y = labels.float().view(-1, 1)

    #     # Compute coefficients as per Defer loss
    #     clip_min, clip_max = 0.01, 2.0  # clip需要尝试
    #     raw_pos_coeff = probs / (probs - 0.5 * dp_prob + eps)
    #     pos_coeff = torch.where(raw_pos_coeff < 1.0,
    #                             torch.tensor(2.0, device=probs.device),
    #                             raw_pos_coeff)
    #     pos_coeff = pos_coeff.clamp(min=1.0, max=clip_max)
    #     neg_coeff = ((1 - probs) / (1 - probs + 0.5 * dp_prob + eps)).clamp(min=clip_min, max=clip_max)
    #     num_clip_max = (pos_coeff == clip_max).sum()
    #     if self.step % 100 == 0:
    #         print(f"############pos_coeff: {pos_coeff[:5].detach().cpu().numpy()}")
    #         print(f"############neg_coeff: {neg_coeff[:5].detach().cpu().numpy()}")
    #     # Compute the final Defer-style loss
    #     loss = - (y * pos_coeff * torch.log(probs) +
    #             (1 - y) * neg_coeff * torch.log(1 - probs))
    #     return loss.mean(), num_clip_max

    def defer_loss(self, outputs, labels, eps=1e-6):  # 使用更大的eps
        # Clamp probs for numerical stability
        probs = outputs[0].clamp(eps, 1 - eps)  # f_theta(x)
        dp_prob = outputs[1].clamp(eps, 1 - eps).detach()  # f_dp(x), detached
        
        # dp截断 - 确保括号正确
        dp_prob = torch.minimum(dp_prob, (3.0/3)*probs)
        
        y = labels.float().view(-1, 1)

        # 计算分母时添加更大的eps
        denominator_pos = (probs - 0.5 * dp_prob + eps)
        denominator_neg = (1 - probs + 0.5 * dp_prob + eps)
        
        # 确保分母不为零或负值
        denominator_pos = denominator_pos.clamp(min=eps)
        denominator_neg = denominator_neg.clamp(min=eps)
        
        raw_pos_coeff = probs / denominator_pos
        pos_coeff = torch.where(raw_pos_coeff < 1.0,
                            torch.tensor(2.0, device=probs.device),
                            raw_pos_coeff)
        pos_coeff = pos_coeff.clamp(min=1.0, max=2.0)  # 固定clip_max为2.0
        
        neg_coeff = ((1 - probs) / denominator_neg).clamp(min=0.01, max=2.0)
        
        if self.step % 100 == 0:
            print(f"############probs: {probs[:5].detach().cpu().numpy()}")
            print(f"############dp_prob: {dp_prob[:5].detach().cpu().numpy()}")
            print(f"############denominator_pos: {denominator_pos[:5].detach().cpu().numpy()}")
            print(f"############denominator_neg: {denominator_neg[:5].detach().cpu().numpy()}")
            print(f"############pos_coeff: {pos_coeff[:5].detach().cpu().numpy()}")
            print(f"############neg_coeff: {neg_coeff[:5].detach().cpu().numpy()}")

        loss = - (y * pos_coeff * torch.log(probs) +
                (1 - y) * neg_coeff * torch.log(1 - probs))
        return loss.mean(), (pos_coeff == 2.0).sum()

    def train_minibatch(self, minibatch):
        self.step += 1
        self.model.train()
        minibatch, labels = self.preprocess_minibatch(minibatch)
        # ctcvr_predictions, dp_predictions, tn_predictions = self.model(minibatch)
        outputs = self.model(minibatch)
        labels = torch.from_numpy(labels).reshape(-1, 1)
        loss,num_clip_max = self.defer_loss(outputs, labels)

        if self.step % 200 == 0:
            print(f"loss: {loss}")
            print("Outputs check:")
            for i in range(min(10, len(outputs[0]))):
                pred = outputs[0][i].item()
                dp_pred = outputs[1][i].item()
                print(f"[{i}] pred: {pred:.6f}, dp_pred: {dp_pred:.6f}")
                print(f"[{i}] label: {labels[i].item():.6f}")
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
        self.update_progress(outputs[0], labels, num_clip_max)
        return minibatch

    def validate_minibatch(self, minibatch):
        self.step += 1
        self.model.eval()
        ndarrays, labels = self.preprocess_minibatch(minibatch)
        outputs = self.model(minibatch)
        labels = torch.from_numpy(labels).reshape(-1, 1)
        loss,num_clip_max = self.defer_loss(outputs, labels)
        # print(f"ctcvr_labels shape: {ctcvr_labels.shape}, ctcvr_predictions shape: {ctcvr_predictions.shape}")
        # print(f"ctr_labels shape: {ctr_labels.shape}, ctr_predictions shape: {ctr_predictions.shape}")
        # print(f"ctcvr_labels: {ctcvr_labels[:5]}, ctcvr_predictions: {ctcvr_predictions[:5]}")
        # print(f"ctr_labels: {ctr_labels[:5]}, ctr_predictions: {ctr_predictions[:5]}")
        # print("Val check:")
        # input()
        self.update_progress(outputs[0], labels, num_clip_max)
        return self._make_validation_result(minibatch, labels, outputs[0], outputs[1])
        # print("Validation step completed.")
        # input()
        # return ctcvr_predictions.detach().reshape(-1), \
        #        ctr_predictions.detach().reshape(-1), \
        #        cvr_predictions.detach().reshape(-1)
    def _make_validation_result(self, minibatch, labels, predictions,dp_predictions):
        labels = labels.reshape(-1).numpy().astype(self.output_label_column_type)
        predictions = predictions.detach().reshape(-1).numpy().astype(self.output_prediction_column_type)
        dp_predictions = dp_predictions.detach().reshape(-1).numpy().astype(self.output_prediction_column_type)
        minibatch[self.output_label_column_name] = labels
        minibatch[self.output_prediction_column_name] = predictions
        minibatch["rawDP"] = dp_predictions
        return minibatch
    
    def _make_validation_result_schema(self, df):
        from pyspark.sql.types import StructType
        fields = []
        reserved = set([self.output_label_column_name, self.output_prediction_column_name])
        for field in df.schema.fields:
            if field.name not in reserved:
                fields.append(field)
        result_schema = StructType(fields)
        result_schema.add(self.output_label_column_name, self.output_label_column_type)
        result_schema.add(self.output_prediction_column_name, self.output_prediction_column_type)
        result_schema.add("rawDP", self.output_prediction_column_type)
        return result_schema
    
    def _create_metric(self):
        metric = DEFERMetric()
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
            delta = DEFERMetric.from_states(states)
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

class DEFERMetric(ms.BinaryClassificationModelMetric):
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
