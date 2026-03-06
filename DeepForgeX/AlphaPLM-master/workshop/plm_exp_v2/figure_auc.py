import os, sys

def PCOC(num_clicks, num_impressions, predicted_ctr):
    sum_pctr = 0
    for i in range(len(num_clicks)):
        sum_pctr += predicted_ctr[i] * (num_impressions[i] + num_clicks[i])
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

auc_dict = {}
for line in sys.stdin:
    label, score = line.strip().split(' ')
    auc_dict.setdefault(score, [0, 0])
    if float(label) != 1:
        auc_dict[score][0] += 1
    else:
        auc_dict[score][1] += 1

predicted_ctr = []
num_impressions = []
num_clicks = []
for score in auc_dict:
    predicted_ctr.append(float(score))
    num_impressions.append(auc_dict[score][0])
    num_clicks.append(auc_dict[score][1])

auc, bucket = scoreClickAUC(num_clicks, num_impressions, predicted_ctr)
pcoc = PCOC(num_clicks, num_impressions, predicted_ctr)

print('*' * 20, 'validation result', '*' * 20)
print('pos num:', sum(num_clicks), 'neg num:', sum(num_impressions))
print('AUC: %.4f, PCOC: %.4f' % (auc, pcoc))
print('*' * 60)

