import random
import numpy as np
import torch
from torch.backends import cudnn


def set_seed(seed: int = 0):
    """
    Sets the seed for generating random numbers. Using identical seeds for reproducibility.

    Args:
        seed: The desired seed.

    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    # cudnn.enabled = True
