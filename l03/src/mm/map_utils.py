"""Utility functions for MAP."""
from tqdm.auto import tqdm

from src.mm import map_
from src.mm import vis


def plot_map_estimate_given_sample_size(
        prior_model_cls,
        lk_model_cls,
        data,
        true_prob,
        step,
        epochs_per_step,
):
    x = []
    maps = []

    for size in tqdm(range(1, len(data), step)):
        pr_model = prior_model_cls()
        lk_model = lk_model_cls()

        map_.run_map(
            prior_model=pr_model,
            likelihood_model=lk_model,
            data=data[:size],
            num_epochs=epochs_per_step,
            lr=1e-1,
        )

        x.append(size)
        mp = lk_model.probs
        maps.append(mp.item() if len(mp) == 1 else mp.tolist())

    vis.plot_param_value_over_time(
        maps,
        true_prob,
        x_vals=x,
        xlabel='Data sample size',
        title='MAPs for different sample sizes',
    )
