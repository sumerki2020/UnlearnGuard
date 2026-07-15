"""Global determinism. torch is imported lazily so CPU-only tooling
(dataset generation, config checks) works without it installed."""

import os
import random

import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
