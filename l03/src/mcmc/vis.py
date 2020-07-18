"""Functions for visualization of MCMC results."""
import matplotlib.pyplot as plt
import torch


def summary(samples, qs=(0.05, 0.95)):
    """Computes mean, std and percentiles for all samples."""
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            **{
                f'{q * 100:.0f}%': v.kthvalue(int(len(v) * q), dim=0)[0]
                for q in qs
            }
        }
    return site_stats


def plot_predictions(ds_test, predictors):
    """Plots predictions of given MCMC predictors."""
    fig, axs = plt.subplots(
        nrows=len(predictors), ncols=2,
        figsize=(15, 5 * len(predictors)),
        sharey=True,
        squeeze=False,
    )

    for ax, (predictor_name, predictors) in zip(axs, predictors.items()):
        for is_cold in (0, 1):
            sl = ds_test[is_cold]['speed_limit']
            m_true = ds_test[is_cold]['measurement']

            samples = predictors[is_cold](sl)
            pred = summary(samples)['measurement']

            x, ym, ylb, yub, y_true = list(zip(*sorted(
                zip(
                    sl.tolist(),
                    pred['mean'].tolist(),
                    pred['5%'].tolist(),
                    pred['95%'].tolist(),
                    m_true,
                ),
                key=lambda r: r[0]
            )))

            ax[is_cold].plot(x, ym)
            ax[is_cold].fill_between(x, ylb, yub, alpha=0.5)

            ax[is_cold].plot(
                x, y_true,
                marker='o', ms=2,
                linestyle='', alpha=0.5
            )
            ax[is_cold].set(
                xlabel='Speed limit',
                ylabel='Measurement',
                title='Cold' if is_cold == 1 else 'Warm',
            )

        ax[0].set_ylabel(f"{predictor_name}\nMeasurement")

    plt.show()
