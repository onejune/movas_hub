# metrics_eval.py

from typing import List, Tuple

def compute_auc_pcoc(label_prediction_list):
    """
    根据 label 和 raw prediction 列表计算 AUC 和 PCOC

    参数:
        label_prediction_list: List of (label, prediction_score)

    返回:
        auc, pcoc
    """
    auc_dict = {}
    for label, score in label_prediction_list:
        score_str = str(score)
        auc_dict.setdefault(score_str, [0, 0])  # [负样本数, 正样本数]
        if float(label) != 1:
            auc_dict[score_str][0] += 1
        else:
            auc_dict[score_str][1] += 1

    predicted_ctr = []
    num_impressions = []
    num_clicks = []
    for score_str in auc_dict:
        predicted_ctr.append(float(score_str))
        num_impressions.append(auc_dict[score_str][0])
        num_clicks.append(auc_dict[score_str][1])

    auc, _ = scoreClickAUC(num_clicks, num_impressions, predicted_ctr)
    pcoc = PCOC(num_clicks, num_impressions, predicted_ctr)
    return auc, pcoc


def PCOC(num_clicks: List[int], num_impressions: List[int], predicted_ctr: List[float]) -> float:
    """
    计算 PCOC（预测点击率的加权平均 / 实际点击数）

    参数:
        num_clicks: 每组的点击数
        num_impressions: 每组的曝光数（负样本数）
        predicted_ctr: 每组的预测 CTR

    返回:
        pcoc 值
    """
    sum_pctr = 0.0
    for i in range(len(num_clicks)):
        sum_pctr += predicted_ctr[i] * (num_impressions[i] + num_clicks[i])
    sum_clicks = sum(num_clicks)
    return round(sum_pctr / sum_clicks, 4) if sum_clicks != 0 else 0.0


def scoreClickAUC(num_clicks: List[int], num_impressions: List[int], predicted_ctr: List[float]) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    计算 click-based AUC

    参数:
        num_clicks: 每个预测分数 bucket 中的点击数
        num_impressions: 每个 bucket 中的曝光数
        predicted_ctr: 每个 bucket 的预测CTR

    返回:
        auc 值 和 bucket 信息
    """
    i_sorted = sorted(range(len(predicted_ctr)), key=lambda i: predicted_ctr[i], reverse=True)

    auc_temp = 0.0
    click_sum = 0.0
    old_click_sum = 0.0
    no_click = 0.0
    no_click_sum = 0.0

    last_ctr = predicted_ctr[i_sorted[0]] + 1.0
    bucket = []

    for i in range(len(predicted_ctr)):
        if last_ctr != predicted_ctr[i_sorted[i]]:
            bucket.append((click_sum, no_click_sum, last_ctr))
            auc_temp += (click_sum + old_click_sum) * no_click / 2.0
            old_click_sum = click_sum
            no_click = 0.0
            last_ctr = predicted_ctr[i_sorted[i]]
        no_click += num_impressions[i_sorted[i]] - num_clicks[i_sorted[i]]
        no_click_sum += num_impressions[i_sorted[i]] - num_clicks[i_sorted[i]]
        click_sum += num_clicks[i_sorted[i]]

    auc_temp += (click_sum + old_click_sum) * no_click / 2.0

    if click_sum == 0 or no_click_sum == 0:
        auc = 1.0
    else:
        auc = round(auc_temp / (click_sum * no_click_sum), 5)

    bucket.append((click_sum, no_click_sum, last_ctr))
    return auc, bucket
