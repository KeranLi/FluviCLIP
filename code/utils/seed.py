# seed.py
import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    固定随机种子以确保结果的可重复性。

    Args:
        seed (int): 随机种子值，默认为42。
    """
    random.seed(seed)  # Python 随机种子
    np.random.seed(seed)  # NumPy 随机种子
    torch.manual_seed(seed)  # PyTorch 随机种子
    torch.cuda.manual_seed(seed)  # GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 多 GPU 随机种子
    torch.backends.cudnn.deterministic = True  # 确保每次卷积的结果一致
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 的优化（影响速度但可重复）
    print(f"Random seed set to: {seed}")
