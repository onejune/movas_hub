"""
Scoring Metrics for KDD Cup 2012, Track 2

Reads in a solution/subission files

Scores on the following three metrics:
-NWMAE
-WRMSE
-AUC

Author: Ben Hamner (kdd2012@benhamner.com)
"""

import os
import sys
from tabulate import tabulate

metric_result = []

def scoreElementwiseMetric(num_clicks, num_impressions, predicted_ctr, elementwise_metric):
    """
    Calculates an elementwise error metric

    Parameters
    ----------
    num_clicks : a list containing the number of clicks

    num_impressions : a list containing the number of impressions

    predicted_ctr : a list containing the predicted click-through rates

    elementwise_metric : a function such as MSE that evaluates the error on a single instance, given the clicks, impressions, and p_ctr

    Returns
    -------
    score : the error on the elementwise metric over the set
    """
    score = 0.0
    weight_sum = 0.0

    for clicks, impressions, p_ctr in zip(num_clicks, num_impressions, predicted_ctr):
        score += elementwise_metric(clicks, impressions, p_ctr)*impressions
        weight_sum += impressions
    score = score / weight_sum
    return score

def scoreWRMSE(num_clicks, num_impressions, predicted_ctr):
    """
    Calculates the Weighted Root Mean Squared Error (WRMSE)

    Parameters
    ----------
    num_clicks : a list containing the number of clicks

    num_impressions : a list containing the number of impressions

    predicted_ctr : a list containing the predicted click-through rates

    Returns
    -------
    wrmse : the weighted root mean squared error
    """
    import math

    mse = lambda clicks, impressions, p_ctr: math.pow(clicks/impressions-p_ctr,2.0)
    wmse = scoreElementwiseMetric(num_clicks, num_impressions, predicted_ctr, mse)
    wrmse = math.sqrt(wmse)
    return wrmse

def scoreNWMAE(num_clicks, num_impressions, predicted_ctr):
    """
    Calculates the normalized weighted mean absolute error

    Parameters
    ----------
    num_clicks : a list containing the number of clicks

    num_impressions : a list containing the number of impressions

    predicted_ctr : a list containing the predicted click-through rates

    Returns
    -------
    nwmae : the normalized weighted mean absolute error
    """
    mae = lambda clicks, impressions, p_ctr: abs(clicks/impressions-p_ctr)
    nwmae = scoreElementwiseMetric(num_clicks, num_impressions, predicted_ctr, mae)
    return nwmae

def PCOC(num_clicks, num_impressions, predicted_ctr):
    sum_pctr = 0
    for i in range(len(num_clicks)):
        sum_pctr += predicted_ctr[i] * num_impressions[i]
    sum_clicks = sum(num_clicks)
    return sum_pctr / sum_clicks

def scoreClickAUC(num_clicks, num_impressions, predicted_ctr):
    """
    Calculates the area under the ROC curve (AUC) for click rates

    Parameters
    ----------
    num_clicks : a list containing the number of clicks

    num_impressions : a list containing the number of impressions

    predicted_ctr : a list containing the predicted click-through rates

    Returns
    -------
    auc : the area under the ROC curve (AUC) for click rates
    """
    i_sorted = sorted(range(len(predicted_ctr)),key=lambda i: predicted_ctr[i],reverse=True)
    auc_temp = 0.0
    click_sum = 0.0
    old_click_sum = 0.0
    no_click = 0.0
    no_click_sum = 0.0


    # treat all instances with the same predicted_ctr as coming from the
    # same bucket
    last_ctr = predicted_ctr[i_sorted[0]] + 1.0

    bucket = []
    for i in range(len(predicted_ctr)):
        if last_ctr != predicted_ctr[i_sorted[i]]:
            bucket.append((click_sum, no_click_sum, last_ctr))
            auc_temp += (click_sum+old_click_sum) * no_click / 2.0
            old_click_sum = click_sum
            no_click = 0.0
            last_ctr = predicted_ctr[i_sorted[i]]
        no_click += num_impressions[i_sorted[i]] - num_clicks[i_sorted[i]]
        no_click_sum += num_impressions[i_sorted[i]] - num_clicks[i_sorted[i]]
        click_sum += num_clicks[i_sorted[i]]
    auc_temp += (click_sum+old_click_sum) * no_click / 2.0
    if click_sum==0 or no_click_sum==0:
        auc=1
    else:
        auc = auc_temp / (click_sum * no_click_sum)
    bucket.append((click_sum, no_click_sum, last_ctr))
    return auc, bucket

def saveROCCoordinates(bucket, roc_file):
    max_click_sum = bucket[-1][0]
    max_no_click_sum = bucket[-1][1]

    fout = open(roc_file, 'w')
    for i in range(len(bucket)):
        fp_rate = bucket[i][1] * 1.0 / max_no_click_sum
        tp_rate = bucket[i][0] * 1.0 / max_click_sum
        fout.write('%f\t%f\n' % (fp_rate, tp_rate))
    fout.close()

def bucket_predictions(num_clicks, num_impressions, predicted_ctr, num_digits=4):
    predicted_ctr_buckets = {}

    for clicks, impressions, p_ctr in zip(num_clicks, num_impressions, predicted_ctr):
        p_ctr = round(p_ctr, num_digits)
        if p_ctr not in predicted_ctr_buckets:
            predicted_ctr_buckets[p_ctr] = [0,0]
        predicted_ctr_buckets[p_ctr][0] += clicks
        predicted_ctr_buckets[p_ctr][1] += impressions
    predicted_ctr_b = sorted(predicted_ctr_buckets.keys())
    num_clicks_b = []
    num_impressions_b = []
    for p_ctr in predicted_ctr_b:
        num_clicks_b.append(predicted_ctr_buckets[p_ctr][0])
        num_impressions_b.append(predicted_ctr_buckets[p_ctr][1])
    return (num_clicks_b, num_impressions_b, predicted_ctr_b)

def bucket_predictions_quantiles(num_clicks, num_impressions, predicted_ctr, num_quantiles=50):
    i_sorted = sorted(range(len(predicted_ctr)),key=lambda i: predicted_ctr[i],
                      reverse=True)
    num_clicks_q = []
    num_impressions_q = []
    predicted_ctr_q = []

    clicks = 0
    impressions = 0
    p_clicks = 0

    for i in range(len(i_sorted)):
        clicks += num_clicks[i_sorted[i]]
        impressions += num_impressions[i_sorted[i]]
        p_clicks += predicted_ctr[i_sorted[i]] * num_impressions[i_sorted[i]]

        if i % int(len(num_clicks)/num_quantiles) == 0 or i == len(i_sorted)-1:
            num_clicks_q.append(clicks)
            num_impressions_q.append(impressions)
            predicted_ctr_q.append(p_clicks / impressions)

            clicks = 0
            impressions = 0
            p_clicks = 0

    return (num_clicks_q, num_impressions_q, predicted_ctr_q)

def read_solution_file(f_sol_name):
    """
    Reads in a solution file

    Parameters
    ----------
    f_sol_name : submission file name

    Returns
    -------
    num_clicks : a list of clicks
    num_impressions : a list of impressions
    """
    f_sol = open(f_sol_name)

    num_clicks = []
    num_impressions = []

    for line in f_sol:
        line = line.strip().split("\001")
        num_clicks.append(int(line[1]))
        num_impressions.append(int(line[0]))
    return (num_clicks, num_impressions)

def read_submission_file(f_sub_name):
    """
    Reads in a submission file

    Parameters
    ----------
    f_sub_name : submission file name

    Returns
    -------
    predicted_ctr : a list of predicted click-through rates
    """
    f_sub = open(f_sub_name)

    predicted_ctr = []

    for line in f_sub:
        line = line.strip().split("\001")
        p = round(float(line[2]), 6)
        predicted_ctr.append(p)

    return predicted_ctr

def _read_file(input_file):
    num_clicks = []
    num_impressions = []
    predicted_ctr = []

    fin = open(input_file)
    for line in fin:
        fields = line.rstrip(os.linesep).split('\001')
        num_impressions.append(int(fields[0]))
        num_clicks.append(int(fields[1]))
        p = round(float(fields[2]), 6)
        predicted_ctr.append(p)

    return num_clicks, num_impressions, predicted_ctr

class Leaf(object):
    def __init__(self):
        self.num_clicks = []
        self.num_impressions = []
        self.predicted_ctr = []

def _merge_leaves(leaves):
    prob_dict = {}
    for leaf in leaves:
        for i in range(len(leaf.predicted_ctr)):
            counter = prob_dict.setdefault(leaf.predicted_ctr[i],
                                           [0, 0])
            counter[0] += leaf.num_impressions[i]
            counter[1] += leaf.num_clicks[i]

    predicted_ctr = []
    num_impressions = []
    num_clicks = []

    for k, v in prob_dict.items():
        predicted_ctr.append(k)
        num_impressions.append(v[0])
        num_clicks.append(v[1])

    return predicted_ctr, num_impressions, num_clicks

class Node(object):
    def __init__(self, key, path):
        self.key = key
        self.path = path
        self.children = {}
        self.leaf = None

def _insert_node(node, key, path):
    return node.children.setdefault(key, Node(key, path))

def _build_auc_tree(input_file):
    root = Node('', 'all')
    fin = open(input_file)
    for line in fin:
        fields = line.rstrip(os.linesep).split('\t')
        levels = fields[0].split('|')[1].split('_') # 按照下划线划分维度
        node = root
        for i in range(len(levels)):
            key = levels[i]
            path = '_'.join(levels[:i + 1])
            node = _insert_node(node, key, path)
        if node.leaf == None:
            node.leaf = Leaf()

        p = round(float(fields[0].split('|')[0]), 6)
        node.leaf.predicted_ctr.append(p)
        node.leaf.num_impressions.append(int(fields[1]))
        node.leaf.num_clicks.append(int(fields[2].split(".")[0]))

    return root

def _get_all_leaves(node):
    leaves = []

    def _get_aux(node):
        if len(node.children) == 0 and node.leaf is not None:
            leaves.append(node.leaf)
        else:
            for k, v in node.children.items():
                _get_aux(v)

    _get_aux(node)
    return leaves

def _walk_tree(node, visit, output):
    visit(node, output)
    for k, child in node.children.items():
        _walk_tree(child, visit, output)

def _evaluation(node, output):
    predicted_ctr, num_impressions, num_clicks = _merge_leaves(_get_all_leaves(node))
    if sum(num_clicks)<10:
        if sum(num_impressions)>10000:
            ivr = round(sum(num_clicks) / sum(num_impressions), 6)
            metric_result.append([node.path, 0, 0, sum(num_clicks), sum(num_impressions), ivr])
        return

    auc, bucket = scoreClickAUC(num_clicks, num_impressions, predicted_ctr)
    pcoc = PCOC(num_clicks, num_impressions, predicted_ctr)
    ivr = round(sum(num_clicks) / sum(num_impressions), 6)
    #saveROCCoordinates(bucket, roc_file)
    #output.write("-" * 50 + "\n")
    #output.write("%s|AUC:%.4f PCOC:%.4f CLICK:%d IMPRESSION:%d\n" % (node.path, auc, pcoc, sum(num_clicks), sum(num_impressions)))
    metric_result.append([node.path, auc, pcoc, sum(num_clicks), sum(num_impressions), ivr])
    
    #output.write("%s|PCOC:%.4f\n" % (node.path, pcoc))
    nwmae = scoreNWMAE(num_clicks, num_impressions, predicted_ctr)
    #output.write("%s|NWMAE:%f\n" % (node.path, nwmae))
    wrmse = scoreWRMSE(num_clicks, num_impressions, predicted_ctr)
    #output.write("%s|WRMSE:%f\n" % (node.path, wrmse))
    #output.write("%s|CLICK:%d\n" % (node.path, sum(num_clicks)))
    #output.write("%s|IMPRESSION:%d\n" % (node.path, sum(num_impressions)))

def _print_path(node, output):
    print >>output, node.path

def main(input_file, output_file):
    dtm = 'date'
    try:
        dtm = input_file.split('_')[-4]
    except:
        pass
    print('dtm:', dtm)
    root = _build_auc_tree(input_file)
    _walk_tree(root, _evaluation, None)

    # 表头
    headers = ["DATE", "Key", "AUC", "PCOC", "PURCHASE", "IMPRESSION", "IVR"]
    new_result = []
    for ele in metric_result:
        if ele[3] < 200:
            continue
        ele[0] = 'metric-' + ele[0]
        ele = [dtm] + ele
        new_result.append(ele)
        
    new_result.sort(key = lambda d: d[5], reverse = True)
    # 使用 tabulate 生成表格: simple/plain/grid
    table = tabulate(new_result, headers, tablefmt="grid")
    # 将表格写入文件
    with open(output_file, 'w') as file:
        file.write(table)
    print("metrics 已成功写入%s文件" % (output_file))

if __name__=="__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python score_kdd.py input_file, output_file")
        sys.exit(-1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
