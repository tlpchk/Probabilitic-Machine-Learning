"""Utility functions for MLE."""
from tqdm.auto import tqdm

from src.mm import mle
from src.mm import vis


def plot_mle_estimate_given_sample_size(
    model_cls,
    data,
    true_prob,
    step,
    epochs_per_step
):
    """Runs MLE for given data sample sizes and plots results."""
    x = []
    mles = []

    for size in tqdm(range(1, len(data), step)):
        model = model_cls()

        mle.run_mle(
            likelihood_model=model,
            data=data[:size],
            num_epochs=epochs_per_step,
            lr=1e-1,
        )

        x.append(size)
        mp = model.probs
        mles.append(mp.item() if len(mp) == 1 else mp.tolist())

    vis.plot_param_value_over_time(
        mles,
        true_prob,
        x_vals=x,
        xlabel='Data sample size',
        title='MLEs for different sample sizes',
    )
