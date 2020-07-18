"""Implementation of Maximum A Posteriori."""
from typing import Dict, List, Union

import torch
from tqdm.auto import tqdm

from src.mm import vis


def run_map(
    prior_model: torch.nn.Module,
    likelihood_model: torch.nn.Module,
    data: List[Union[int, List[int]]],
    num_epochs: int,
    lr: float,
    verbose=False,
) -> Dict[str, Union[List[float], Dict[str, List[float]]]]:
    """Implements Maximum Likelihood Estimation using gradient descent."""
    pass
