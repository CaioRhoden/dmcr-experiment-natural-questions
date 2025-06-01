import torch
import numpy as np
import random
import polars as pl

def set_random_seed(seed: int) -> None:
    """
    Set all relevant seeds for deterministic behavior.

    This function sets the seeds for PyTorch (on CPU and all GPUs), NumPy,
    Python's random module, and Polars. Additionally, it sets PyTorch's
    CuDNN backend to be deterministic and disables benchmarking.

    Note that this function does not affect any other libraries or frameworks
    that may be in use.

    Args:
        seed: The seed to use for all random number generators.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    pl.set_random_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
