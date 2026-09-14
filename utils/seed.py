"""随机种子设置，保证实验可复现（论文要求）。"""

import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    """固定 random / numpy / torch / cuda 的随机源。

    注意：确定性模式（cudnn.deterministic=True）会牺牲少量性能，
    但对论文实验的可复现性是必要的。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
