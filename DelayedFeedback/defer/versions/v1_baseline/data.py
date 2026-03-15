"""
Defer 数据加载 - PyTorch 版本

数据格式 (TSV):
    click_ts  pay_ts  num_0~num_7  cat_0~cat_8
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import pickle


SECONDS_A_DAY = 60 * 60 * 24
SECONDS_AN_HOUR = 60 * 60


class DeferDataset(Dataset):
    """
    Defer 数据集
    
    Args:
        features: [N, 17] 特征 (8 数值 + 9 类别)
        click_ts: [N] 点击时间戳
        pay_ts: [N] 转化时间戳 (-1 表示未转化)
        labels: [N, ?] 标签 (维度取决于方法)
    """
    def __init__(self, features, click_ts, pay_ts, labels=None, sample_ts=None):
        self.features = features
        self.click_ts = click_ts
        self.pay_ts = pay_ts
        self.sample_ts = sample_ts if sample_ts is not None else click_ts
        
        if labels is not None:
            self.labels = labels
        else:
            # 默认标签: 是否转化
            self.labels = (pay_ts > 0).astype(np.int32)
    
    def __len__(self):
        return len(self.click_ts)
    
    def __getitem__(self, idx):
        # 数值特征: 空 tensor (模型会忽略)
        num_feats = torch.tensor([], dtype=torch.float32)
        # 类别特征 (跳过前 8 列数值特征，取后 9 列)
        cate_feats = torch.tensor(self.features[idx, 8:], dtype=torch.long)
        # 标签
        if isinstance(self.labels, np.ndarray):
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            label = torch.tensor(self.labels, dtype=torch.float32)
        
        return {
            "num_feats": num_feats,
            "cate_feats": cate_feats,
            "label": label,
            "click_ts": self.click_ts[idx],
            "pay_ts": self.pay_ts[idx],
        }


class DataDF:
    """
    数据帧包装类，提供时间切片和标签构造功能
    """
    def __init__(self, features, click_ts, pay_ts, sample_ts=None, labels=None):
        self.features = features
        self.click_ts = click_ts
        self.pay_ts = pay_ts
        self.sample_ts = sample_ts if sample_ts is not None else click_ts.copy()
        
        if labels is not None:
            self.labels = labels
        else:
            self.labels = (pay_ts > 0).astype(np.int32)
    
    def __len__(self):
        return len(self.click_ts)
    
    def sub_days(self, start_day, end_day):
        """按天切片数据"""
        start_ts = start_day * SECONDS_A_DAY
        end_ts = end_day * SECONDS_A_DAY
        mask = (self.sample_ts >= start_ts) & (self.sample_ts < end_ts)
        return DataDF(
            self.features[mask],
            self.click_ts[mask],
            self.pay_ts[mask],
            self.sample_ts[mask],
            self.labels[mask] if len(self.labels.shape) > 0 else self.labels
        )
    
    def sub_hours(self, start_hour, end_hour):
        """按小时切片数据"""
        start_ts = start_hour * SECONDS_AN_HOUR
        end_ts = end_hour * SECONDS_AN_HOUR
        mask = (self.sample_ts >= start_ts) & (self.sample_ts < end_ts)
        return DataDF(
            self.features[mask],
            self.click_ts[mask],
            self.pay_ts[mask],
            self.sample_ts[mask],
            self.labels[mask] if len(self.labels.shape) > 0 else self.labels
        )
    
    def add_fake_neg(self, cut_sec):
        """
        添加假负样本处理
        
        短窗口策略: 窗口内未转化的标记为负，后续转化的复制为正样本
        
        Args:
            cut_sec: 观察窗口大小 (秒)
        
        Returns:
            labels: [batch_size, 2] -> [label, elapsed_time]
        """
        # 窗口内转化的正样本
        pos_in_window = (self.pay_ts > 0) & (self.pay_ts - self.click_ts <= cut_sec)
        # 窗口外转化的样本 (延迟正样本)
        pos_out_window = (self.pay_ts > 0) & (self.pay_ts - self.click_ts > cut_sec)
        # 未转化的样本
        neg = self.pay_ts < 0
        
        # 构造标签: [label, elapsed_time]
        # label: 窗口内正样本 = 1，其他 = 0
        # elapsed_time: 从曝光到当前的时间 (用于 FNW 加权)
        label_col = np.zeros(len(self.click_ts), dtype=np.float32)
        label_col[pos_in_window] = 1
        
        # elapsed_time: 窗口内未转化样本的经过时间 (归一化)
        elapsed_time = np.zeros(len(self.click_ts), dtype=np.float32)
        # 对于窗口外的样本，elapsed_time = cut_sec (已经过了整个窗口)
        elapsed_time[~pos_in_window & ~pos_out_window] = cut_sec
        
        labels = np.stack([label_col, elapsed_time], axis=1)
        
        # 复制延迟正样本
        if pos_out_window.sum() > 0:
            delay_features = self.features[pos_out_window]
            delay_click_ts = self.click_ts[pos_out_window]
            delay_pay_ts = self.pay_ts[pos_out_window]
            delay_sample_ts = self.pay_ts[pos_out_window]  # 用转化时间作为样本时间
            # 延迟正样本的标签: [1, 0] (label=1, elapsed_time=0)
            delay_labels = np.zeros((pos_out_window.sum(), 2), dtype=np.float32)
            delay_labels[:, 0] = 1
            
            # 合并
            features = np.vstack([self.features, delay_features])
            click_ts = np.concatenate([self.click_ts, delay_click_ts])
            pay_ts = np.concatenate([self.pay_ts, delay_pay_ts])
            sample_ts = np.concatenate([self.sample_ts, delay_sample_ts])
            labels = np.vstack([labels, delay_labels])
            
            # 按 sample_ts 排序
            idx = np.argsort(sample_ts)
            return DataDF(features[idx], click_ts[idx], pay_ts[idx], sample_ts[idx], labels[idx])
        
        return DataDF(self.features, self.click_ts, self.pay_ts, self.sample_ts, labels)
    
    def add_oracle_labels(self):
        """
        构造 Oracle 标签 (真正的上界)
        
        使用完整的真实转化标签，不做任何假负处理
        这是理论上界，实际场景中无法获得
        
        标签: [is_converted, 0] (保持 2 维兼容性)
        """
        is_converted = (self.pay_ts > 0).astype(np.float32)
        # 第二维填 0，保持和其他方法兼容
        labels = np.stack([is_converted, np.zeros_like(is_converted)], axis=1)
        
        return DataDF(self.features, self.click_ts, self.pay_ts, self.sample_ts, labels)
    
    def add_esdfm_labels(self, cut_sec, tn_window=24*3600):
        """
        构造 ES-DFM 标签 (原版实现)
        
        ES-DFM 核心思想：
        - 观察窗口内，样本分为三类：
          1. 窗口内转化的正样本 (pos_label=1)
          2. 窗口内未转化，但后来转化的延迟正样本 (dp_label=1)
          3. 真正的负样本 (tn_label=1)
        
        标签格式: [tn_label, dp_label, pos_label]
        - tn_label: 是否为真负样本
        - dp_label: 是否为延迟正样本 (窗口外转化)
        - pos_label: 是否为正样本 (窗口内转化)
        """
        delay = self.pay_ts - self.click_ts
        is_pos = self.pay_ts > 0
        
        # 窗口内转化的正样本
        pos_in_window = is_pos & (delay <= cut_sec)
        # 窗口外转化的延迟正样本
        pos_out_window = is_pos & (delay > cut_sec)
        # 真负样本 (永不转化)
        true_neg = self.pay_ts < 0
        
        labels = np.zeros((len(self.click_ts), 3), dtype=np.float32)
        labels[true_neg, 0] = 1           # tn_label
        labels[pos_out_window, 1] = 1     # dp_label
        labels[pos_in_window, 2] = 1      # pos_label
        
        return DataDF(self.features, self.click_ts, self.pay_ts, self.sample_ts, labels)
    
    def add_dfm_labels(self, cut_sec):
        """
        构造 DFM 标签
        
        标签: [is_converted, delay_time_normalized]
        delay_time 归一化到 [0, 1] 范围，避免数值爆炸
        """
        is_converted = (self.pay_ts > 0).astype(np.float32)
        
        # 计算延迟时间 (秒)
        raw_delay = np.where(
            self.pay_ts > 0,
            self.pay_ts - self.click_ts,
            cut_sec
        ).astype(np.float32)
        
        # 修复负延迟 (数据问题) 和限制最大值
        raw_delay = np.clip(raw_delay, 0, cut_sec * 10)  # 最大 10 倍窗口
        
        # 归一化到 [0, 1]，用窗口大小作为基准
        delay_time = raw_delay / cut_sec
        
        labels = np.stack([is_converted, delay_time], axis=1)
        
        return DataDF(self.features, self.click_ts, self.pay_ts, self.sample_ts, labels)
    
    def add_winadapt_labels(self, cut_sec, win1=6, win2=24, win3=72, bt_max_delay=None):
        """
        构造自适应窗口标签 (Defer 核心) - 对齐 TF 原版 add_delay_winadapt_cut_fake_neg

        TF 原版核心逻辑：
        1. 每个样本有独立的 time_win (per-sample 观察窗口，按 business_type 的最大延迟)
        2. 混合三种样本：
           - 通常样本 (sample_ts = click_ts + time_win)
           - 延迟正样本 label_11 (sample_ts = pay_ts，转化发生时再次出现)
           - 真负样本 label_10 (sample_ts = click_ts + 1day，次日确定为负)
        3. label_01_30/60_mask 基于 time_win 动态计算

        Args:
            cut_sec: 基础观察窗口 (秒)，未使用 bt_max_delay 时的默认值
            win1/win2/win3: 三个子窗口 (小时)
            bt_max_delay: dict {bt_value: max_delay_hours}，按 business_type 动态设置 time_win
                          None 时所有样本使用 win3 作为 time_win

        标签: 14 维 (同 TF 原版)
            0:  label_11      - 延迟正样本 (窗口外转化)
            1:  label_10      - 真负样本
            2:  label_01_15   - win1 内转化
            3:  label_01_30   - win1~win2 内转化 (增量)
            4:  label_01_60   - win2~win3 内转化 (增量)
            5:  label_01_30_sum - win2 内累计转化
            6:  label_01_60_sum - win3 内累计转化
            7:  label_01_30_mask - win2 是否可观察 (time_win >= win2)
            8:  label_01_60_mask - win3 是否可观察 (time_win >= win3)
            9:  label_00      - 窗口内负样本
            10: label_01      - 窗口内正样本 (win3 内所有转化)
            11: label_11_15   - 延迟正且 win1 < delay <= win2
            12: label_11_30   - 延迟正且 win2 < delay <= win3
            13: label_11_60   - 延迟正且 delay > win3
        """
        win1_sec = win1 * SECONDS_AN_HOUR
        win2_sec = win2 * SECONDS_AN_HOUR
        win3_sec = win3 * SECONDS_AN_HOUR

        delay = self.pay_ts - self.click_ts
        is_pos = self.pay_ts > 0

        # ==================== per-sample time_win ====================
        # 按 business_type 动态设置观察窗口 (对齐 TF 原版 adatag=True)
        # bt_col_index = 0 (cate_feat[0] 是 business_type)
        if bt_max_delay is not None:
            bt_values = self.features[:, 8].astype(int)  # cate_feat[0] = col 8
            time_win = np.full(len(self.click_ts), win3_sec, dtype=np.float64)
            for bt_val, max_hours in bt_max_delay.items():
                mask_bt = (bt_values == int(bt_val))
                time_win[mask_bt] = max_hours * SECONDS_AN_HOUR
        else:
            time_win = np.full(len(self.click_ts), win3_sec, dtype=np.float64)

        # ==================== 三类样本构造 ====================
        # mask_spm1: 延迟正样本 (delay > time_win，窗口外转化)
        mask_spm1 = is_pos & (delay > time_win)
        # mask_spm0: 真负样本 (永不转化)
        mask_spm0 = ~is_pos

        # 合并三类样本
        n_orig = len(self.click_ts)
        n_spm1 = mask_spm1.sum()
        n_spm0 = mask_spm0.sum()

        features_all = np.vstack([
            self.features,
            self.features[mask_spm1],
            self.features[mask_spm0]
        ])
        click_ts_all = np.concatenate([
            self.click_ts,
            self.click_ts[mask_spm1],
            self.click_ts[mask_spm0]
        ])
        pay_ts_all = np.concatenate([
            self.pay_ts,
            self.pay_ts[mask_spm1],
            self.pay_ts[mask_spm0]
        ])
        # sample_ts: 通常=click_ts+time_win, 延迟正=pay_ts, 真负=click_ts+1day
        sample_ts_all = np.concatenate([
            self.click_ts + time_win,                              # 通常样本
            self.pay_ts[mask_spm1],                                # 延迟正样本
            self.click_ts[mask_spm0] + SECONDS_A_DAY              # 真负样本
        ])
        time_win_all = np.concatenate([
            time_win,
            time_win[mask_spm1],
            time_win[mask_spm0]
        ])

        # ==================== 14 维标签构造 ====================
        delay_all = pay_ts_all - click_ts_all
        is_pos_all = pay_ts_all > 0

        # 各窗口内转化 (仅通常样本段有意义)
        # 需要同时满足 delay <= time_win (在观察窗口内)
        in_win = delay_all <= time_win_all

        pos_15_part = is_pos_all & (delay_all <= win1_sec)
        pos_30_part = is_pos_all & (delay_all <= win2_sec)
        pos_60_part = is_pos_all & (delay_all <= win3_sec)

        # label_01_15: win1 内转化 且 在观察窗口内
        label_01_15 = (is_pos_all & (delay_all <= win1_sec) & in_win).astype(np.float32)
        # label_01_30: win1~win2 内转化 且 在观察窗口内
        label_01_30 = (is_pos_all & (delay_all > win1_sec) & (delay_all <= win2_sec) & in_win).astype(np.float32)
        # label_01_60: win2~win3 内转化 且 在观察窗口内
        label_01_60 = (is_pos_all & (delay_all > win2_sec) & (delay_all <= win3_sec) & in_win).astype(np.float32)

        label_01_30_sum = np.minimum(label_01_15 + label_01_30, 1)
        label_01_60_sum = np.minimum(label_01_15 + label_01_30 + label_01_60, 1)

        # label_01_30_mask: time_win >= win2 且 尚未在 win2 内转化 → 不确定 → mask=0
        # 对齐 TF: mask = 1 - (time_win < cut_sec2 AND label_01_30_sum_part < 1)
        pos_30_sum_part = np.minimum(pos_15_part.astype(float) + (is_pos_all & (delay_all > win1_sec) & (delay_all <= win2_sec)).astype(float), 1)
        label_01_30_mask = 1 - ((time_win_all < win2_sec - 1) & (pos_30_sum_part < 1)).astype(np.float32)

        pos_60_sum_part = np.minimum(pos_15_part.astype(float) + (is_pos_all & (delay_all > win1_sec) & (delay_all <= win2_sec)).astype(float) + (is_pos_all & (delay_all > win2_sec) & (delay_all <= win3_sec)).astype(float), 1)
        label_01_60_mask = 1 - ((time_win_all < win3_sec - 1) & (pos_60_sum_part < 1)).astype(np.float32)

        # label_11: 延迟正样本段 (第二段全为1，其余为0)
        label_11 = np.zeros(len(click_ts_all), dtype=np.float32)
        label_11[n_orig:n_orig + n_spm1] = 1

        # label_10: 真负样本段 (第三段全为1，其余为0)
        label_10 = np.zeros(len(click_ts_all), dtype=np.float32)
        label_10[n_orig + n_spm1:] = 1

        # label_00: 通常样本中，窗口内未转化 (delay > time_win 或 pay_ts <= 0)
        label_00 = (~is_pos_all | (delay_all > time_win_all)).astype(np.float32)
        label_00[n_orig:] = 0  # 延迟正和真负段不算 label_00

        # label_01: 通常样本中，窗口内转化
        label_01 = (is_pos_all & in_win).astype(np.float32)
        label_01[n_orig:] = 0

        # label_11_15/30/60: 延迟正样本段的细分
        # 对齐 TF: label_11_15 = delay in (win1, win2] AND delay > time_win
        label_11_15 = np.zeros(len(click_ts_all), dtype=np.float32)
        label_11_30 = np.zeros(len(click_ts_all), dtype=np.float32)
        label_11_60 = np.zeros(len(click_ts_all), dtype=np.float32)
        if n_spm1 > 0:
            d_spm1 = delay_all[n_orig:n_orig + n_spm1]
            tw_spm1 = time_win_all[n_orig:n_orig + n_spm1]
            label_11_15[n_orig:n_orig + n_spm1] = ((d_spm1 > win1_sec) & (d_spm1 <= win2_sec) & (d_spm1 > tw_spm1)).astype(np.float32)
            label_11_30[n_orig:n_orig + n_spm1] = ((d_spm1 > win2_sec) & (d_spm1 <= win3_sec) & (d_spm1 > tw_spm1)).astype(np.float32)
            label_11_60[n_orig:n_orig + n_spm1] = ((d_spm1 > win3_sec) & (d_spm1 > tw_spm1)).astype(np.float32)

        labels = np.stack([
            label_11, label_10, label_01_15, label_01_30, label_01_60,
            label_01_30_sum, label_01_60_sum, label_01_30_mask, label_01_60_mask,
            label_00, label_01, label_11_15, label_11_30, label_11_60
        ], axis=1)

        # 按 sample_ts 排序
        idx = np.argsort(sample_ts_all, kind='stable')
        return DataDF(features_all[idx], click_ts_all[idx], pay_ts_all[idx], sample_ts_all[idx], labels[idx])
    
    def to_dataset(self):
        """转换为 PyTorch Dataset"""
        return DeferDataset(
            self.features,
            self.click_ts,
            self.pay_ts,
            self.labels,
            self.sample_ts
        )
    
    def to_dataloader(self, batch_size=1024, shuffle=False, num_workers=2):
        """转换为 PyTorch DataLoader (内存优化版)"""
        dataset = self.to_dataset()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,  # 关闭 pin_memory 减少内存占用
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=False  # 关闭持久化 worker，每次评估后释放内存
        )


def load_data(data_path, cache_path=None):
    """
    加载数据
    
    Args:
        data_path: 数据文件路径 (TSV 格式)
        cache_path: 缓存路径 (可选)
    
    Returns:
        DataDF 对象
    """
    # 尝试从缓存加载
    if cache_path and Path(cache_path).exists():
        print(f"从缓存加载: {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return DataDF(
            data['features'],
            data['click_ts'],
            data['pay_ts'],
            data.get('sample_ts')
        )
    
    print(f"加载数据: {data_path}")
    
    # 读取 TSV 文件
    df = pd.read_csv(data_path, sep='\t', header=None)
    
    click_ts = df.iloc[:, 0].values.astype(np.int64)
    pay_ts = df.iloc[:, 1].values.astype(np.float64)
    
    # 特征: 第 2-18 列 (8 数值 + 9 类别)
    features = df.iloc[:, 2:].values.astype(np.float32)
    
    print(f"  样本数: {len(click_ts)}")
    print(f"  转化数: {(pay_ts > 0).sum()}")
    print(f"  转化率: {(pay_ts > 0).mean():.4f}")
    print(f"  时间范围: {click_ts.min()} ~ {click_ts.max()}")
    print(f"  天数: {(click_ts.max() - click_ts.min()) / SECONDS_A_DAY:.1f}")
    
    # 保存缓存
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'features': features,
                'click_ts': click_ts,
                'pay_ts': pay_ts,
            }, f)
        print(f"  缓存已保存: {cache_path}")
    
    return DataDF(features, click_ts, pay_ts)


def get_stream_data(data, method, params):
    """
    获取流式训练数据
    
    Args:
        data: DataDF 对象
        method: 方法名
        params: 参数字典
    
    Returns:
        (train_stream, test_stream): 按小时分割的数据流
    """
    start = params.get('stream_start', 30)
    mid = params.get('stream_mid', 30)
    end = params.get('stream_end', 60)
    cut_sec = params.get('C', 6) * SECONDS_AN_HOUR
    
    # 加载 business_type 最大延迟配置
    import json, os
    bt_max_delay = None
    bt_groups_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'business_type_groups.json')
    if os.path.exists(bt_groups_path):
        with open(bt_groups_path) as f:
            bt_cfg = json.load(f)
        bt_max_delay = {int(k): float(v) for k, v in bt_cfg.get('bt_max_delay', {}).items()}

    # 根据方法构造标签
    if method == "Oracle":
        train_data = data.sub_days(0, end).add_oracle_labels()
    elif method == "Vanilla":
        train_data = data.sub_days(0, end).add_fake_neg(cut_sec)
    elif method == "ES-DFM":
        train_data = data.sub_days(0, end).add_esdfm_labels(cut_sec)
    elif method == "DFM":
        train_data = data.sub_days(0, end).add_dfm_labels(cut_sec)
    elif method == "delay_win_adapt":
        win1 = params.get('win1', 6)
        win2 = params.get('win2', 24)
        win3 = params.get('win3', 72)
        # 对齐 TF 原版：传入 bt_max_delay 实现 per-sample time_win
        train_data = data.sub_days(0, end).add_winadapt_labels(cut_sec, win1, win2, win3, bt_max_delay=bt_max_delay)
    else:
        train_data = data.sub_days(0, end).add_fake_neg(cut_sec)

    # 测试数据 (标准)：统一用 Oracle 标签评估
    test_data = data.sub_days(mid, end).add_oracle_labels()

    # nowin 测试数据：包含延迟正样本 (对齐 TF 原版 test_stream_nowin)
    # 延迟正样本 = 转化发生在最大窗口之外的样本，用真实标签评估
    test_data_nowin = data.sub_days(mid, end).add_oracle_labels()

    # 按小时分割
    train_stream = []
    test_stream = []
    test_stream_nowin = []

    for hour in range(mid * 24, (end - 1) * 24 + 23):
        hour_data = train_data.sub_hours(hour, hour + 1)
        if len(hour_data) > 10:
            train_stream.append(hour_data)

    for hour in range(mid * 24 + 1, end * 24):
        hour_data = test_data.sub_hours(hour, hour + 1)
        if len(hour_data) > 10:
            test_stream.append(hour_data)
        nowin_data = test_data_nowin.sub_hours(hour, hour + 1)
        if len(nowin_data) > 10:
            test_stream_nowin.append(nowin_data)

    return train_stream, test_stream, test_stream_nowin


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 测试数据加载
    data_path = "../data/business_data.txt"
    
    if Path(data_path).exists():
        data = load_data(data_path)
        print(f"\n数据加载成功: {len(data)} 条")
        
        # 测试时间切片
        day0 = data.sub_days(0, 1)
        print(f"第 0 天: {len(day0)} 条")
        
        # 测试标签构造
        with_labels = data.add_fake_neg(900)
        print(f"添加假负样本后: {len(with_labels)} 条")
        
        # 测试 DataLoader
        loader = day0.to_dataloader(batch_size=32)
        batch = next(iter(loader))
        print(f"Batch keys: {batch.keys()}")
        print(f"num_feats shape: {batch['num_feats'].shape}")
        print(f"cate_feats shape: {batch['cate_feats'].shape}")
    else:
        print(f"数据文件不存在: {data_path}")
