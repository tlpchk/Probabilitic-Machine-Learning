"""Utility function for plotting."""
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_losses_over_time(loss_values: List[float]):
    """Plots loss curve."""
    x = list(range(len(loss_values)))
    plt.figure(figsize=(15, 5))
    plt.plot(x, loss_values, color='red', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')


def plot_param_value_over_time(
    param_values: List[List[float]],
    true_param_value: List[float],
    x_vals: List[int] = None,
    xlabel: str = 'Epoch',
    title: str = 'Parameters probability estimates over epochs',
):
    """Plots changes of parameter value during training process."""
    n_params = len(true_param_value)

    fig, axs = plt.subplots(nrows=n_params, ncols=1, figsize=(15, 4 * n_params))

    if x_vals is None:
        x_vals = list(range(len(param_values)))
    y = np.array(param_values)

    for idx, ax in enumerate(axs.ravel()):
        ax.plot(
            x_vals, y[:, idx],
            marker='x', linestyle='',
            color='green', label=f'[{idx}] MLE',
        )
        ax.plot(
            x_vals, y[:, idx],
            marker='', linestyle='--', alpha=0.2,
        )

        ax.axhline(
            true_param_value[idx],
            linestyle='--',
            color='blue',
            label=f'[{idx}] True probability',
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Probability')
        ax.set_ylim((0, 1))
        ax.legend()

    fig.suptitle(title)
