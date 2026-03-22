#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dnn_trainFlow.py - DNN 模型训练流程

继承 MsBaseTrainFlow，只实现 _build_model_module (支持 15+ 模型类型)。
公共模块 (Spark管理/数据读取/训练/评估/主流程) 均在基类 base_trainFlow.py 中。
"""
import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(__file__))
print('sys.path:' + str(sys.path))

import metaspore as ms
print("metaspore.__file__ =", ms.__file__)

from metaspore.algos.lr_ftrl_net import *
from metaspore.algos.widedeep_net import *
from metaspore.algos.maskNet import MaskNet
from metaspore.algos.deepfm_net import DeepFM
from metaspore.algos.ffm_net import FFM
from metaspore.algos.dcn_net import DCN
from metaspore.algos.multi_task import PEPNet, PEPNet2
from metaspore.algos.deep_censored_model import DeepCensoredModel
from metaspore.algos.fg_net import FourChannelGateModel
from metaspore.algos.ppnet import PPNet
from metaspore.algos.fwfm_net import FwFM
from metaspore.algos.apg_net import APGNet
from metaspore.algos.pepnet_singleTask import PEPNetSingleTask

from base_trainFlow import BaseTrainFlow
from movas_logger import MovasLogger


class DNNModelTrainFlow(BaseTrainFlow):
    """
    DNN 模型训练流程

    继承 MsBaseTrainFlow，实现 _build_model_module 支持以下模型:
    LRFtrl, LRFtrl2, LRFtrl3, DeepFM, WideDeep, WideDeep2,
    masknet, FourChannelGateModel, FFM, DCN, apg, fwfm, ppnet
    """

    def _build_model_module(self):
        configed_model = self.params.get('model_type', "WideDeep")
        MovasLogger.add_log(content=f"Building model module: {configed_model}")

        if configed_model == "LRFtrl":
            self.model_module = LRFtrl(
                batch_norm=self.batch_norm,
                net_dropout=self.net_dropout,
                wide_embedding_dim=self.embedding_size,
                wide_combine_schema_path=self.wide_combine_schema_path,
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta
            )
        elif configed_model == "LRFtrl2":
            self.model_module = LRFtrl2(
                wide_embedding_dim=self.embedding_size,
                wide_combine_schema_path=self.wide_combine_schema_path,
                adam_learning_rate=self.adam_learning_rate,
            )
        elif configed_model == "LRFtrl3":
            self.model_module = LRFtrl3(
                wide_embedding_dim=self.embedding_size,
                wide_combine_schema_path=self.wide_combine_schema_path,
                learning_rate=self.adam_learning_rate,
            )
        elif configed_model == "DeepFM":
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
        elif configed_model == "WideDeep2":
            self.model_module = WideDeep2(
                use_wide=self.use_wide,
                batch_norm=self.batch_norm,
                net_dropout=self.net_dropout,
                wide_embedding_dim=self.embedding_size,
                deep_embedding_dim=self.embedding_size,
                wide_combine_schema_path=self.wide_combine_schema_path,
                deep_combine_schema_path=self.combine_schema_path,
                dnn_hidden_units=self.dnn_hidden_units,
                dnn_hidden_activations="relu",
                adam_learning_rate=self.adam_learning_rate,
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta
            )
        elif configed_model == "masknet":
            self.model_module = MaskNet(
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
        elif configed_model == "FourChannelGateModel":
            self.model_module = FourChannelGateModel(
                batch_norm=self.batch_norm,
                net_dropout=self.net_dropout,
                use_wide=self.use_wide,
                embedding_dim=self.embedding_size,
                combine_schema_path=self.combine_schema_path,
                wide_combine_schema_path=self.wide_combine_schema_path,
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
        elif configed_model == "apg":
            self.model_module = APGNet(
                use_wide=self.use_wide,
                batch_norm=self.batch_norm,
                net_dropout=self.net_dropout,
                wide_embedding_dim=self.embedding_size,
                deep_embedding_dim=self.embedding_size,
                wide_combine_schema_path=self.wide_combine_schema_path,
                deep_combine_schema_path=self.combine_schema_path,
                apg_hidden_units=self.dnn_hidden_units,
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta,
                sparse_optimizer_type=self.sparse_optimizer_type
            )
        elif configed_model == "fwfm":
            self.model_module = FwFM(
                use_wide=self.use_wide,
                use_dnn=self.use_dnn,
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
        elif configed_model == "ppnet":
            self.model_module = PPNet(
                embedding_dim=self.embedding_size,
                combine_schema_path=self.combine_schema_path,
                gate_combine_schema_path=self.gate_combine_schema_path,
                batch_norm=self.batch_norm,
                net_dropout=self.net_dropout,
                dnn_hidden_units=self.dnn_hidden_units,
                ftrl_l1=self.ftrl_l1,
                ftrl_l2=self.ftrl_l2,
                ftrl_alpha=self.ftrl_alpha,
                ftrl_beta=self.ftrl_beta
            )
        else:
            raise ValueError(
                f"Unsupported model type: {configed_model}. "
                f"Supported: LRFtrl, LRFtrl2, LRFtrl3, DeepFM, WideDeep, WideDeep2, "
                f"masknet, FourChannelGateModel, FFM, DCN, apg, fwfm, ppnet"
            )
        self.configed_model = configed_model


if __name__ == "__main__":
    args = DNNModelTrainFlow.parse_args()
    print(f'DNNModelTrainFlow: debug_args={args}')
    trainer = DNNModelTrainFlow(config_path=args.conf)
    trainer.run_complete_flow(args)
    MovasLogger.save_to_local()
