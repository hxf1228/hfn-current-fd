from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.version = 2

_C.path = CN()
_C.path.root_dir = "/home/hxf/fault-diagnosis/current-fd"
_C.path.matpath = "data/mvmdf_1.mat"
_C.path.h5path = "data/mvmd.h5"
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.data = CN()
_C.data.train_ratio = 0.9

_C.params = CN()
_C.params.batch_size = 256


def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    return _C.clone()
