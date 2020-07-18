"""Function for dataset generation."""
import numpy as np
from sklearn import model_selection as sk_ms
import torch


def generate_dataset(num_total, train_size):
    """Generates dataset with `num_total` samples and given train size."""
    # Data generator parameters
    means = {0: 0.99, 1: 0.75}
    stds = {0: 10, 1: 5}

    # Dataset
    ds = {'train': {}, 'test': {}}

    speed_limits = np.random.uniform(low=30, high=240, size=(num_total,))

    for is_cold in [0, 1]:
        measurements = [
            torch.distributions.Normal(
                means[is_cold] * sl,
                stds[is_cold]
            ).sample().item()
            for sl in speed_limits
        ]

        xtr, xte, ytr, yte = sk_ms.train_test_split(
            speed_limits.tolist(), measurements,
            train_size=train_size,
        )

        ds['train'][is_cold] = {
            'speed_limit': torch.tensor(xtr),
            'measurement': torch.tensor(ytr),
        }
        ds['test'][is_cold] = {
            'speed_limit': torch.tensor(xte),
            'measurement': torch.tensor(yte),
        }

    return ds
