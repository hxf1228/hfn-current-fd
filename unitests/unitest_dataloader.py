from utils.dataloader import DataKAT
from config import get_cfg_defaults
import torch

if __name__ == '__main__':
    config_file = rf'../configs/matrixcnn.yaml'

    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    # cfg.merge_from_list(["data.train_ratio", 0.8,)
    cfg.freeze()

    gen_train = DataKAT(cfg)

    condition = 1

    x_data, y_label = gen_train.get_data(condition)

    x_tr = torch.tensor(x_data)
    y_tr = torch.LongTensor(y_label)
    dataset = torch.utils.data.TensorDataset(x_tr, y_tr)

