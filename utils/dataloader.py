import numpy as np
import h5py


class DataKAT:
    def __init__(self, cfg):
        self._cfg = cfg
        self.condiname = ["K1", "K2", "K3", "K4"]
        self.labelname = ["K1Label", "K2Label", "K3Label", "K4Label"]

    def get_data(self, condition):
        data_h5 = h5py.File(self._cfg.path.h5path, 'r')
        data_raw = data_h5[self.condiname[condition - 1]]
        labels_raw = data_h5[self.labelname[condition - 1]]

        data = np.asarray(data_raw, dtype=np.float32)
        label = np.asarray(labels_raw, dtype=np.int32)

        data_h5.close()

        label = label.reshape(-1) - 1

        return data, label
