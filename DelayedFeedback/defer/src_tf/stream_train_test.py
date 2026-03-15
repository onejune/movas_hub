from models import get_model
from loss import get_loss_fn
from utils import get_optimizer, ScalarMovingAverage
from metrics import cal_llloss_with_logits, cal_auc, cal_prauc, cal_llloss_with_prob, ece_score
from data import get_criteo_dataset_stream
from model_utils import load_model_weights
from tqdm import tqdm
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers, Model
import tensorflow as tf
from collections import defaultdict
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")
if len(physical_devices) > 1:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[1], enable=True)
    except RuntimeError as e:
        print(f"GPU 1 memory growth setting failed: {e}")


def test(model, test_data, params):
    all_logits = []
    all_probs = []
    all_labels = []
    all_props = []
    for step, (batch_x, batch_y) in enumerate(tqdm(test_data), 1):
        outputs = model.predict(batch_x)
        
        # 处理返回值为字典的情况
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        all_logits.append(logits if isinstance(logits, np.ndarray) else logits.numpy())
        all_labels.append(batch_y.numpy() if hasattr(batch_y, 'numpy') else batch_y)
         
        if params["method"] == "3class":
            prop = tf.nn.softmax(logits)
            prop_00 = tf.reshape(prop[:, 0], (-1, 1)) #tf.slice(prop, [0, 0],[-1, 1])
            prop_01 = tf.reshape(prop[:, 1], (-1, 1))
            prop_11 = tf.reshape(prop[:, 2], (-1, 1))
            prop_1 = prop_01+prop_11
            all_probs.append(prop_1)
            all_props.append(prop)
        elif params["method"] == "delay_win_time" or params["method"] == "likeli" or params["method"] == "test" or params["method"] == "delay_win_time_iw"\
				or params["method"] == "delay_win_time_sep" or params["method"] == "delay_win_adapt" or params["method"] == "" or params["method"] == "ES-DFM-win" or params["method"] == "ES-DFM-wines":
            all_probs.append(tf.sigmoid(logits[:,0]))
            all_props.append(tf.sigmoid(logits[:,0]))

        elif params["method"] == "ES-DFM" or params["method"] == "ES-DFM-normal" or params["method"] == "ES-DFM10" or params["method"] == "ES-DFM-FULL":
            # MLP_tn_dp: logits shape (-1, 2), col0=tn_logits (cv pred)
            all_probs.append(tf.sigmoid(logits[:, 0]))
            all_props.append(tf.sigmoid(logits[:, 0]))
        else:
            all_probs.append(tf.sigmoid(logits))
            all_props.append(tf.sigmoid(logits))

    if params["method"] == "3class":
        all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1, 3))
        all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1, ))
        all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1, ))
        all_props = np.reshape(np.concatenate(all_props, axis=0), (-1, 3))
    elif params["method"] == "delay_win_time" or params["method"] == "likeli" or params["method"] == "test" or params["method"] == "delay_win_time_sep" or params["method"] == "delay_win_adapt" or params["method"] == "" or params["method"] == "delay_win_time_iw" or params["method"] == "ES-DFM-win" or params["method"] == "ES-DFM-wines":
        all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1, 2))
        all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1, ))
        all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1, ))
        all_props = np.reshape(np.concatenate(all_props, axis=0), (-1, ))
    elif params["method"] =="delay_win_select":
        all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1,))
        all_labels = np.reshape(np.concatenate(all_labels[:,0], axis=0), (-1,))
        all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1,))
        all_props = np.reshape(np.concatenate(all_props, axis=0), (-1,))
    else:
        all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1,))
        all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1,))
        all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1,))
        all_props = np.reshape(np.concatenate(all_props, axis=0), (-1,))


    if params["method"] == "FNC":
        all_probs = all_probs / (1-all_probs+1e-8)
        llloss = cal_llloss_with_prob(all_labels, all_probs)

    elif params["method"] == "FNC10":
        all_probs = all_probs * 2
        llloss = cal_llloss_with_prob(all_labels, all_probs)
    elif params["method"] == "3class":
        llloss = cal_llloss_with_prob(all_labels, all_probs)
    elif params["method"] == "delay_win_time" or params["method"] == "likeli" or params["method"] == "test" or params["method"] == "delay_win_time_sep" or params["method"] == "delay_win_adapt" or params["method"] == "" or params["method"] == "delay_win_time_iw" or params["method"] == "ES-DFM-win" or params["method"] == "ES-DFM-wines":
        cv_prop = tf.reshape(tf.cast(all_probs, tf.float32), (-1, 1))
        cv_label = tf.reshape(tf.cast((all_labels), tf.float32), (-1, 1))

        #loss_cv = -tf.reduce_mean(tf.math.log(cv_prop) * cv_label + tf.math.log(1-cv_prop)* (1-cv_label))
        
        #print "cv_label"
        #print tf.reduce_mean(cv_label)
        #print "pcvr"
        #print tf.reduce_mean(cv_prop)
        #print "test_loss_cv"
        #print loss_cv

        cv_label_numpy = np.reshape(cv_label.numpy(),(-1, 1))
        cv_prop_numpy = np.reshape(cv_prop.numpy(),(-1, 1))
        cv_label_numpy_1 = np.reshape(cv_label.numpy(),(-1))
        cv_prop_numpy_1 = np.reshape(cv_prop.numpy(),(-1))

        # 过滤 NaN（WinAdapt/win_time 模型可能产生 NaN probs）
        import math
        nan_mask = ~np.isnan(cv_prop_numpy_1)
        if nan_mask.sum() < len(cv_prop_numpy_1):
            cv_prop_numpy_1 = cv_prop_numpy_1[nan_mask]
            cv_label_numpy_1 = cv_label_numpy_1[nan_mask]
            cv_prop_numpy = cv_prop_numpy[nan_mask]
            cv_label_numpy = cv_label_numpy[nan_mask]

        llloss = cal_llloss_with_prob(cv_label_numpy_1, cv_prop_numpy_1)

        cv_auc = cal_auc(cv_label_numpy_1, cv_prop_numpy_1)
        prauc = cal_prauc(cv_label_numpy_1, cv_prop_numpy_1)

        ctr = np.mean(cv_label_numpy_1,axis=0)
        pctr = np.mean(cv_prop_numpy_1,axis=0)

        ece = ece_score(cv_label_numpy_1,cv_prop_numpy_1)
        #print "cv_auc"
        #print cv_auc
        return cv_auc, prauc, llloss, ctr, pctr, all_labels, all_probs, ece

    else:
        llloss = cal_llloss_with_logits(all_labels, all_logits)
    batch_size = all_logits.shape[0]

    # 过滤 NaN（ES-DFM 重要性权重可能产生 NaN）
    nan_mask = ~np.isnan(all_probs)
    if nan_mask.sum() < len(all_probs):
        all_probs = all_probs[nan_mask]
        all_labels = all_labels[nan_mask]

    pred = all_probs >= 0.5

    auc = cal_auc(all_labels, all_probs)
    prauc = cal_prauc(all_labels, all_probs)

    ctr = np.mean(all_labels,axis=0)
    pctr = np.mean(all_probs,axis=0)
    prop = np.mean(all_props,axis=0)

    #cv_label_numpy_1 = np.reshape(all_labels.numpy(),(-1))
    #cv_prop_numpy_1 = np.reshape(all_probs.numpy(),(-1))
    ece = ece_score(all_labels,all_probs)
    #ece = ece_score(cv_label_numpy_1,cv_prop_numpy_1)
    #print "test_label:%f" 
    #print ctr
    #print "test_prop:"
    #print prop
    #print "test_pctr:"
    #print pctr

    return auc, prauc, llloss, ctr, pctr, all_labels, all_probs, ece


def train(models, optimizer, train_data, params):
    if params["loss"] == "none_loss":
        return
    loss_fn = get_loss_fn(params["loss"])
    for step, batch in enumerate(tqdm(train_data), 1):
        batch_x = batch[0]
        batch_y = batch[1]
        targets = {"label": batch_y}

        with tf.GradientTape() as g:
            outputs = models["model"](batch_x, training=True)
            if params["method"] == "FSIW":
                logits0 = models["fsiw0"](batch_x, training=False)["logits"]
                logits1 = models["fsiw1"](batch_x, training=False)["logits"]
                outputs = {
                    "logits": outputs["logits"],
                    "logits0": logits0,
                    "logits1": logits1
                }
            elif params["method"] == "ES-DFM" or params["method"] == "ES-DFM-win"  or params["method"] == "ES-DFM-normal" or params["method"] == "ES-DFM10" or params["method"] == "ES-DFM-wines" or params["method"] == "delay_win_select" or params["method"] == "ES-DFM-FULL":
                logitsx = models["esdfm"](batch_x, training=False)
                outputs = {
                    "logits": outputs["logits"],
                    "tn_logits": logitsx["tn_logits"],
                    "dp_logits": logitsx["dp_logits"]
                }
            reg_loss = tf.add_n(models["model"].losses)
            loss_dict = loss_fn(targets, outputs, params)
            loss = loss_dict["loss"] + reg_loss
            print("train_loss:")
            print(loss_dict)
        
        trainable_variables = models["model"].trainable_variables
        gradients = g.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))


def _safe_load(model, ckpt_path):
    """安全加载权重，路径为 None 时跳过"""
    if ckpt_path is None:
        print(f"[WARN] pretrain ckpt path is None, skipping weight load for {model.name}")
        return
    full_path = ckpt_path + "/weights.weights.h5"
    import os
    if not os.path.exists(full_path):
        print(f"[WARN] ckpt not found: {full_path}, skipping")
        return
    load_model_weights(model, full_path)


def stream_run(params):
    #train_stream, test_stream = get_criteo_dataset_stream(params)
    ws_model = None
    if params["method"] == "DFM":
        model = get_model("MLP_EXP_DELAY", params)
        _safe_load(model, params["pretrain_dfm_model_ckpt_path"])
    elif params["method"] == "3class":
        model = get_model("MLP_3class", params)
        _safe_load(model, params["pretrain_3class_model_ckpt_path"])
    elif params["method"] == "likeli" or params["method"] == "delay_win_time" or params["method"] == "delay_win_time_iw" or params["method"] == "ES-DFM-win" or params["method"] == "ES-DFM-wines":
        model = get_model("MLP_likeli", params)
        _safe_load(model, params["pretrain_wintime_model_ckpt_path"])
    elif params["method"] == "delay_win_adapt":
        model = get_model("MLP_winadapt", params)
        _safe_load(model, params["pretrain_winselect_model_ckpt_path"])
    elif params["method"] == "delay_win_time_sep":
        model = get_model("MLP_wintime_sep", params)
        _safe_load(model, params["pretrain_sepwintime_model_ckpt_path"])
    else:
        model = get_model("MLP_SIG", params)
        _safe_load(model, params["pretrain_baseline_model_ckpt_path"])
    models = {"model": model}
    if params["method"] == "FSIW":
        fsiw0_model = get_model("MLP_FSIW", params)
        _safe_load(fsiw0_model, params["pretrain_fsiw0_model_ckpt_path"])
        fsiw1_model = get_model("MLP_FSIW", params)
        _safe_load(fsiw1_model, params["pretrain_fsiw1_model_ckpt_path"])
        models["fsiw0"] = fsiw0_model
        models["fsiw1"] = fsiw1_model
    elif params["method"] == "ES-DFM" or params["method"] == "ES-DFM-win" or params["method"] == "ES-DFM-normal" or params["method"] == "ES-DFM10" or params["method"] == "ES-DFM-wines" or params["method"] == "ES-DFM-FULL":
        esdfm_model = get_model("MLP_tn_dp", params)
        _safe_load(esdfm_model, params["pretrain_esdfm_model_ckpt_path"])
        models["esdfm"] = esdfm_model
    elif params["method"] == "delay_win_select":
        ws_model = get_model("MLP_winadapt", params)
        _safe_load(ws_model, params["pretrain_winselect_model_ckpt_path"])
        models["wsm"] = ws_model
        esdfm_model = get_model("MLP_tn_dp", params)
        _safe_load(esdfm_model, params["pretrain_esdfm_model_ckpt_path"])
        models["esdfm"] = esdfm_model
    elif params["method"] == "delay_win_adapt":
        # delay_win_adapt 需要 winselect 模型用于时间窗口选择
        ws_model = get_model("MLP_winadapt", params)
        _safe_load(ws_model, params["pretrain_winselect_model_ckpt_path"])
        models["wsm"] = ws_model

    elif params["method"] == "DFM":
        dfm_model = get_model("MLP_EXP_DELAY", params)
        _safe_load(dfm_model, params["pretrain_dfm_model_ckpt_path"])
        models["model"] = dfm_model

    train_stream, test_stream, test_stream_nowin = get_criteo_dataset_stream(params, pre_trained_model=ws_model)

    optimizer = get_optimizer(params["optimizer"], params)

    auc_ma = ScalarMovingAverage()
    nll_ma = ScalarMovingAverage()
    prauc_ma = ScalarMovingAverage()
    ctr_ma = ScalarMovingAverage()
    pctr_ma = ScalarMovingAverage()
    ece_ma = ScalarMovingAverage()

    nowin_auc_ma = ScalarMovingAverage()
    nowin_nll_ma = ScalarMovingAverage()
    nowin_prauc_ma = ScalarMovingAverage()
    nowin_ctr_ma = ScalarMovingAverage()
    nowin_pctr_ma = ScalarMovingAverage()
    nowin_ece_ma = ScalarMovingAverage()

    for ep, (train_dataset, test_dataset, test_dataset_nowin) in enumerate(zip(train_stream, test_stream, test_stream_nowin)):
        train_data = tf.data.Dataset.from_tensor_slices(
            (dict(train_dataset["x"]), train_dataset["labels"]))
        train_data = train_data.batch(params["batch_size"]).prefetch(1)
        train(models, optimizer, train_data, params)

        test_batch_size = test_dataset["x"].shape[0]
        test_data = tf.data.Dataset.from_tensor_slices(
            (dict(test_dataset["x"]), test_dataset["labels"]))
        test_data = test_data.batch(params["batch_size"]).prefetch(1)
        auc, prauc, llloss, ctr, pctr, label, probs, ece = test(model, test_data, params)
        print("epoch {}, auc {}, prauc {}, llloss {}, ece {}".format(
            ep, auc, prauc, llloss, ece))
        import math
        if not math.isnan(float(auc)):
            auc_ma.add(auc*test_batch_size, test_batch_size)
        nll_ma.add(llloss*test_batch_size, test_batch_size)
        prauc_ma.add(prauc*test_batch_size, test_batch_size)
        ctr_ma.add(ctr*test_batch_size, test_batch_size)
        pctr_ma.add(pctr*test_batch_size, test_batch_size)
        if not math.isnan(float(ece)):
            ece_ma.add(ece*test_batch_size,test_batch_size)
        print("epoch {}, auc_ma {}, prauc_ma {}, llloss_ma {}, ece_ma {}, ctr_ma {}, pctr_ma {}".format(
            ep, auc_ma.get(), prauc_ma.get(), nll_ma.get(), ece_ma.get(), ctr_ma.get(), pctr_ma.get()))
        #for i in range(0, len(probs)):
        #    print 'epoch%d gcn:%d,%f' %(ep, label[i],probs[i])


        if True: #params["method"] == "FNW":
            test_batch_size = test_dataset_nowin["x"].shape[0]
            test_data_nowin = tf.data.Dataset.from_tensor_slices(
                (dict(test_dataset_nowin["x"]), test_dataset_nowin["labels"]))
            test_data_nowin = test_data_nowin.batch(params["batch_size"]).prefetch(1)
            auc, prauc, llloss, ctr, pctr, label, probs, ece = test(model, test_data_nowin, params)
            print("epoch {}, nowin_auc {}, nowin_prauc {}, nowin_llloss {} nowin_ece {}".format(
                ep, auc, prauc, llloss, ece))
            if not math.isnan(float(auc)):
                nowin_auc_ma.add(auc*test_batch_size, test_batch_size)
            nowin_nll_ma.add(llloss*test_batch_size, test_batch_size)
            nowin_prauc_ma.add(prauc*test_batch_size, test_batch_size)
            nowin_ctr_ma.add(ctr*test_batch_size, test_batch_size)
            nowin_pctr_ma.add(pctr*test_batch_size, test_batch_size)
            if not math.isnan(float(ece)):
                nowin_ece_ma.add(ece*test_batch_size,test_batch_size)
            print("epoch {}, nowin_auc_ma {}, nowin_prauc_ma {}, nowin_llloss_ma {}, nowin_ece_ma {},  nowin_ctr_ma {}, nowin_pctr_ma {}".format(
                ep, nowin_auc_ma.get(), nowin_prauc_ma.get(), nowin_nll_ma.get(), nowin_ece_ma.get(), nowin_ctr_ma.get(), nowin_pctr_ma.get()))
            #for i in range(0, len(probs)):
            #    print 'epoch%d gcn:%d,%f' %(ep, label[i],probs[i])




