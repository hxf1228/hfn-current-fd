import os
import numpy as np
import scipy.io as scio
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils.plot_cm import plot_confusion_matrix

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_correct_num = 0

    seed = 1000
    mat_name = "mvmd_1"
    mat_path = rf"data/matlabcm/{mat_name}_cm.mat"
    data_mat = scio.loadmat(mat_path)["cm"]

    pic_name = f'{mat_name}_{seed}_cm.tif'
    pic_path = os.path.join("results", "figures", pic_name)

    labels_test = data_mat[:, 0]
    pred_test = data_mat[:, 1]

    test_correct_num += (pred_test == labels_test).sum().item()

    print('Seed: {}, Test Acc: {:.2f} %'.format(seed,
                                                100 * test_correct_num / 400))

    labels_test_all = labels_test
    out_labels_test_all = pred_test
    vals, idx_start, count = np.unique(labels_test_all, return_counts=True, return_index=True)
    print(count)

    # confusion matrix
    classes_name = {0: 'NO', 1: 'IF', 2: 'OF', 3: 'CO'}

    plt.rc('font', family='Times New Roman')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(labels_test_all, out_labels_test_all)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(5, 4))  # 1inch=2.5cm  8.4cm=3.36inch
    plt.rcParams.update({'font.size': 15})
    shrink_value = 0.93
    plot_confusion_matrix(cnf_matrix,
                          classes=classes_name,
                          normalize=True,
                          shrink=shrink_value)
    tick_marks = np.array(range(len(classes_name))) + 0.5

    # offset the tick
    plt.clim(0, 100)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linewidth=0.8, linestyle='-')
    plt.subplots_adjust(top=0.95, bottom=0.12, right=0.95, left=0.18, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(pic_path, dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.show()
