# coding=utf-8

import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

one_hot_decode = lambda x: np.argmax(x, axis=1) if (len(x.shape) > 1 and x.shape[-1] > 1) else x.flatten()
one_hot_decode_thresh = lambda x, t: x[:,-1]>=t if (len(x.shape) > 1 and x.shape[-1] > 1) else x.flatten()
select_one_class = lambda x: x[:,-1] if (len(x.shape) > 1 and x.shape[-1] > 1) else x.flatten()
colors = ['b','r','c','g','y','m','k']

def print_report(y_true, y_pred, names):
    for yt, yp, name in zip(y_true, y_pred, names):
        print("Classification Report for {}".format(name))
        print( classification_report(yt, one_hot_decode(yp)) )

    print("Confusion Matrix [TN, FP, FN, TP]")
    for yt, yp in zip(y_true, y_pred):
        print( confusion_matrix(yt, one_hot_decode(yp)).ravel() )

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        fpr, tpr, thresh = roc_curve(yt, select_one_class(yp))
        plt.plot(fpr, tpr, colors[i])

        auc = roc_auc_score(yt, select_one_class(yp))

        acc_5fpr = {'fpr': 0, 'acc': 0}
        fpr_max_acc = {'fpr': 0, 'acc': 0}
        for t in thresh:
            tn, fp, fn, tp = confusion_matrix(yt, one_hot_decode_thresh(yp, t)).ravel()
            fpr = fp/(fp+tn)
            acc = (tp+tn)/(tp+tn+fp+fn)

            if acc > fpr_max_acc['acc']:
                fpr_max_acc['acc'] = acc
                fpr_max_acc['fpr'] = fpr

            if fpr > acc_5fpr['fpr'] and fpr <= 0.05:
                acc_5fpr['acc'] = acc
                acc_5fpr['fpr'] = fpr

        print("Max Acc: {}".format(fpr_max_acc))
        print("FPR5: {}".format(acc_5fpr))
        print("AUC: {}".format(auc))
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("results_paths", type=str, nargs="+", help="Path to the file containing prediciton results")
parser.add_argument("--y_pred", default="y_pred", help="Dataset identifier for the predicted results")
parser.add_argument("--y_true", default="y_true", help="Dataset identifier for the ground truth labels")
args = parser.parse_args()

results = [sio.loadmat(path) for path in args.results_paths]
names = ["{}_{}".format(i, path) for i, path in enumerate(args.results_paths)]
y_pred = [res[args.y_pred] for res in results]
y_true = [one_hot_decode(res[args.y_true]) for res in results]

print_report(y_true, y_pred, names)
