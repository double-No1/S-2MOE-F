import numpy as np
from sklearn.metrics import precision_recall_fscore_support
def compute_pre_recall_f1(target, score):
    normal_ratio = (target == 0).sum() / len(target)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = np.zeros(len(score))
    pred[score > threshold] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average='binary')

    # precision, recall, thresholds = precision_recall_curve(target, score)
    # numerator = 2 * recall * precision
    # denom = recall + precision
    # f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    # f1 = np.max(f1_scores)
    return f1