from config import get_cfg_defaults
from utils.dataloader import DataKAT
import torch.utils.data
import numpy as np
from utils.reproduce import set_seed
from scipy.io import savemat

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config_file = rf'configs/exp1.yaml'
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.freeze()

    gen_train = DataKAT(cfg)

    condition = 1

    train_list = []
    test_list = []
    seed_list = [0, 9, 666, 700, 800, 1000, 2000, 2023, 2028, 5000]

    for iSeed in seed_list:
        seed = iSeed
        set_seed(seed)

        x_data, y_label = gen_train.get_data(condition)

        x_tr = torch.tensor(x_data)
        x_tr = x_tr.permute(4, 2, 0, 1, 3)
        y_tr = torch.LongTensor(y_label)
        dataset = torch.utils.data.TensorDataset(x_tr, y_tr)

        dataset_size = len(dataset)
        shuffle_dataset = True
        train_ratio = cfg.data.train_ratio
        test_ratio = 1 - train_ratio
        train_num = int(np.floor(train_ratio * dataset_size))
        test_num = int(np.floor(test_ratio * dataset_size))
        indices = list(range(dataset_size))
        if shuffle_dataset:
            set_seed(seed)
            np.random.shuffle(indices)
        train_indices = indices[0:train_num]
        test_indices = indices[train_num:]
        train_list.append(train_indices)
        test_list.append(test_indices)
    train_list_arr = np.asarray(train_list)
    test_list_arr = np.asarray(test_list)
    savemat("random.mat", {'train_index': train_list_arr, 'test_index': test_list_arr})
