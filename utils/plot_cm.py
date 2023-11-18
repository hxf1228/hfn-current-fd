# Created by Xufeng Huang on 2022-07-01
# Email: xufenghuang1228@gmail.com


import numpy as np
from matplotlib import pyplot as plt
import itertools
import pandas as pd


def plot_confusion_matrix(cm, classes, normalize=False, shrink=1, rotation=0):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    size = 15  #
    cmap = plt.cm.Blues

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.imshow(cm_perc, interpolation='nearest', cmap=cmap)
    cbar = plt.colorbar(shrink=shrink, ticks=[0, 20, 40, 60, 80, 100])
    tick_marks = np.arange(len(classes))
    classes_index = []
    for i in classes.values():
        classes_index.append(i)
    plt.xticks(tick_marks, classes_index, rotation=rotation, size=size)
    plt.yticks(tick_marks, classes_index, size=size)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     c = cm[i, j]
    #     p = cm_perc[i, j]
    #     if i == j:
    #         s = cm_sum[i]
    #         plt.text(j, i, "%.2f%%" % p,
    #                  va="center",
    #                  ha="center",
    #                  color="white" if cm[i, j] > thresh else "black", size=size)
    #     elif c == 0:
    #         plt.text(j, i, " ",
    #                  va="center",
    #                  ha="center",
    #                  color="white" if cm[i, j] > thresh else "black", size=size)
    #     else:
    #         plt.text(j, i, "%.2f%%" % p,
    #                  va="center",
    #                  ha="center",
    #                  color="white" if cm[i, j] > thresh else "black", size=size)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        c = cm[i, j]
        p = cm_perc[i, j]
        if i == j:
            s = cm_sum[i]
            plt.text(j, i, "%.2f%%\n %d/%d" % (p, c, s),
                     va="center",
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black", size=size)
        elif c == 0:
            plt.text(j, i, " ",
                     va="center",
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black", size=size)
        else:
            plt.text(j, i, "%.2f%%\n %d" % (p, c),
                     va="center",
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black", size=size)

    plt.ylabel('True label', size=size)
    plt.xlabel('Predicted label', size=size)
    plt.tight_layout()
