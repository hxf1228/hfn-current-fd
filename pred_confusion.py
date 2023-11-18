from config import get_cfg_defaults
from utils.dataloader import DataKAT
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
from models.cnn3d import Cnn3dPlain, Cnn3dFuse, Cnn3dFuseSmu
from models.matrixcnn import MatrixCNN
from models.rescnn import ResCNN
from models.graycnn import GrayCNN
from utils.reproduce import set_seed
from utils.plot_cm import plot_confusion_matrix

import os
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config_file = rf'configs/exp1.yaml'
    model_name = "cnnfusesmu"  # "cnnfusesmu" "cnn" "cnnfuse" "matrixcnn" "rescnn"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.freeze()

    gen_train = DataKAT(cfg)

    condition = 1
    # seed_list = [0, 9, 666, 700, 800, 1000, 2000, 2023, 2028, 5000]
    seed_list = [1000, ]
    acc_list = []

    for iSeed in seed_list:
        seed = iSeed
        set_seed(seed)

        x_data, y_label = gen_train.get_data(condition)

        x_tr = torch.tensor(x_data)
        if model_name == "matrixcnn":
            x_tr = torch.tensor(x_data)
            x_tr = x_tr.permute(2, 1, 0)
            x_tr = x_tr.view(-1, 1, 2, 4096)
        elif model_name == "graycnn":
            x_tr = torch.tensor(x_data)
            x_tr = x_tr.permute(3, 2, 0, 1) / 255.
        elif model_name == "rescnnp1" or model_name == "rescnnp2":
            x_tr = torch.tensor(x_data)
            x_tr = x_tr.permute(3, 2, 0, 1) / 255.
        else:
            x_tr = torch.tensor(x_data)
            x_tr = x_tr.permute(4, 2, 0, 1, 3)
        y_tr = torch.LongTensor(y_label)
        dataset = torch.utils.data.TensorDataset(x_tr, y_tr)

        dataset_size = len(dataset)
        shuffle_dataset = True
        train_ratio = cfg.data.train_ratio
        test_ratio = 1 - train_ratio
        val_ratio = test_ratio
        train_num = int(np.floor(train_ratio * dataset_size))
        test_num = int(np.floor(test_ratio * dataset_size))
        indices = list(range(dataset_size))
        if shuffle_dataset:
            set_seed(seed)
            np.random.shuffle(indices)
        train_indices = indices[0:train_num]
        test_indices = indices[train_num:]

        # Creating data samplers and loaders:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset,
                                  batch_size=cfg.params.batch_size,
                                  sampler=train_sampler, )

        test_loader = DataLoader(dataset,
                                 batch_size=cfg.params.batch_size,
                                 sampler=test_sampler,
                                 )

        criteria = nn.CrossEntropyLoss()
        if model_name == "cnnfuse":
            classifier = Cnn3dFuse(4).cuda()
        elif model_name == "cnn":
            classifier = Cnn3dPlain(4).cuda()
        elif model_name == "cnnfusesmu":
            classifier = Cnn3dFuseSmu(4).cuda()
        elif model_name == "matrixcnn":
            classifier = MatrixCNN(4).cuda()
        elif model_name == "graycnn":
            classifier = GrayCNN(4).cuda()
        elif model_name == "rescnnp1" or model_name == "rescnnp2":
            classifier = ResCNN(4).cuda()

        checkpoint = torch.load(f'checkpoints/{model_name}_best_{seed}_{condition}_'
                                f'{train_ratio}.pt')

        classifier.load_state_dict(checkpoint)
        classifier.eval()

        with torch.no_grad():
            test_correct_num = 0
            total = 0
            labels_test_all = []
            out_labels_test_all = []
            for iTest, (inputs_test, labels_test) in enumerate(test_loader):
                labels_test_all.extend(labels_test)
                inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                outputs_test = classifier(inputs_test)

                _, pred_test = torch.max(outputs_test, 1)
                out_labels_test_all.extend(pred_test.cpu().numpy())
                total += labels_test.size(0)
                test_correct_num += (pred_test == labels_test).sum().item()

            print('Seed: {}, Test Acc: {:.2f} %'.format(seed,
                                                        100 * test_correct_num / total))
            acc_i = 100 * test_correct_num / total
            acc_list.append(acc_i)

        labels_test_all = np.stack(labels_test_all, axis=0)
        out_labels_test_all = np.stack(out_labels_test_all, axis=0)
        vals, idx_start, count = np.unique(labels_test_all, return_counts=True, return_index=True)
        print(count)

        # confusion matrix
        classes_name = {0: 'NO', 1: 'IF', 2: 'OF', 3: 'CO'}
        # 设置画图属性
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

        pic_name = model_name + f'_{seed}_cm.tif'
        pic_path = os.path.join("results", "figures", pic_name)

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

    print('Mean: {:.2f}, Std: {:.2f}'.format(np.mean(acc_list), np.std(acc_list)))
