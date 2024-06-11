from sklearn import metrics
import numpy as np

def roc_aucs(labels, preds, average=None, multi_class="raise"):
    return metrics.roc_auc_score(labels, preds, average=average, multi_class=multi_class)

def balanced_accs(labels, preds):
    return metrics.balanced_accuracy_score(labels, preds)

def pr_aucs(labels, preds, average=None):
    return metrics.average_precision_score(labels, preds, average=average)

def f1_score(labels, preds, average=None):
    preds = np.around(preds)
    if average:
        return metrics.f1_score(labels, preds, average=average)
    else:
        return metrics.f1_score(labels, preds, average=average)[1]

def stats_calculator(path, metrics=['Test roc-auc','Test pr-auc', 'Test balanced accuracy'], folds=4):
    """
    Calculates mean and std of metrics from logs of a cross-validation experiment. The structure should be path/fold_i/logs.txt
    Args:
        path: path to the folder containing the logs
        metrics: list of metrics to calculate mean and std
        folds: number of folds
    """
    results_dict = {key: [] for key in metrics}
    for i in range(folds):
        logs = open(path + '/fold_'+str(i)+'/logs.txt', 'r').readlines()[-1]
        for j in metrics:
            assert j in logs, 'Metric ' + j+ ' not found in logs'
            results_dict[j].append(float(logs.split(j+': ')[1].split('.. ')[0]))
    for key in results_dict.keys():
        temp = np.array(results_dict[key])
        results_dict[key] = key+" Mean: "+str(np.mean(temp))+" Std: "+str(np.std(temp))
    with open(path + '/results.txt', 'w') as f:
        for key in results_dict.keys():
            f.write(results_dict[key]+'\n')