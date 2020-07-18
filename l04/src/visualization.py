import math

import torch
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from pyro.infer import Predictive
import pyro
from scipy.stats import norm


from src.utils import summary, run_svi, SviPredictive


# Export MCMC utils and SVI utils


def plot_single_prediction(x, y_true, predictor):
    assert type(y_true) == float
    sns.kdeplot(
        predictor(x)['obs'].numpy().ravel(),
        label = 'Bayes Posterior Prediction'
    )
    plt.axvline(x = y_true, label = 'True value',
          c = 'red', linestyle='--')
    plt.legend(loc='upper right')
    plt.xlabel('Model output', size = 18)
    plt.ylabel('Probability Density', size = 18)


def plot_mpd(sites, sites_dist, samples, svi_samples):
    nrows = math.ceil(len(sites) / 2)
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 4 * nrows))
    fig.suptitle(
        "Marginal Posterior density of the Regression Coefficients",
        fontsize=16
    )
    ax = axs.ravel()
    for i, site in enumerate(sites):
        sns.distplot(
            svi_samples[site],
            ax=ax[i],
            label="SVI (Empirical)",
            hist_kws={'alpha':0.2}
        )
        for k,v in samples.items():
            sns.distplot(v[site], ax=ax[i], label=k, hist_kws={'alpha': 0.2})
        x = np.linspace(*ax[i].get_xlim(), num=1000)
        ax[i].plot(x, sites_dist[site].pdf(x), label="SVI (Analitical)")
        ax[i].set_title(site)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()


def plot_data(df, properties=dict()):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 6),
        sharey=True,
        sharex=True
    )
    positive_category = df[df[properties.get('category', 'category')] == 1]
    negative_category = df[df[properties.get('category', 'category')] == 0]
    sns.scatterplot(negative_category[properties.get('x', 'x')],
                negative_category[properties.get('y', 'y')],
                ax=ax[0])
    cat_name = properties.get('category_labels', dict())
    ax[0].set(xlabel=properties.get('x_label', 'x'),
              ylabel=properties.get('y_label', 'y'),
              title=cat_name.get(0, ''))
    sns.scatterplot(positive_category[properties.get('x', 'x')],
                    positive_category[properties.get('y', 'y')],
                    ax=ax[1])
    ax[1].set(xlabel=properties.get('x_label', 'x'),
              ylabel=properties.get('y_label', 'y'),
              title=cat_name.get(1, ''))


def plot_ols(data, properties=dict()):
    """Plots OLS."""
    fig, axs = plt.subplots(
        nrows=len(data), ncols=2,
        figsize=(15, 6 * len(data)),
        sharey=True,
        sharex=True,
        squeeze=False,
    )

    x_col = properties.get('x', None)
    x_label = properties.get('x_label', 'x')
    y_label = properties.get('y_label', 'y')

    cat_col = properties.get('category', None)
    cat_name = properties.get('category_labels', dict())
    positive_cat = cat_name.get(1, 'Positive')
    negative_cat = cat_name.get(0, 'Negative')

    for ax, (predictor_name, predictor) in zip(axs, data.items()):
        x = predictor['x']
        y = predictor['y']
        ols = predictor['ols']
        for category in (0, 1):
            category_idx = x[:, cat_col] == category
            x_data = x[category_idx, x_col]
            y_data = y[category_idx]
            y_mean = ols[0][category_idx]
            y_bottom_ci = ols[1][category_idx, 0]
            y_top_ci = ols[1][category_idx, 1]
            y_bottom_pi = ols[2][category_idx, 0]
            y_top_pi = ols[2][category_idx, 1]

            xplot, ym, ylbci, yubci, ylbpi, yubpi, y_true = list(zip(*sorted(
                zip(
                    x_data.tolist(),
                    y_mean,
                    y_bottom_ci,
                    y_top_ci,
                    y_bottom_pi,
                    y_top_pi,
                    y_data,
                ),
                key=lambda r: r[0]
            )))

            ax[category].plot(
                xplot,
                ym,
                color="red",
                label="Mean output"
            )
            ax[category].fill_between(
                xplot,
                ylbpi,
                yubpi,
                color='cornflowerblue',
                alpha=0.5,
                label=f"Prediction Interval"
            )
            ax[category].fill_between(
                xplot,
                ylbci,
                yubci,
                color='orange',
                alpha=0.5,
                label=f"Confidence Interval"
            )

            ax[category].plot(
                xplot, y_true,
                marker='o', ms=4,
                linestyle='', alpha=1,
                color='green',
                label="True values"
            )
            ax[category].set(
                xlabel=x_label,
                ylabel=y_label,
                title=positive_cat if category == 1 else negative_cat,
            )

        ax[0].set_ylabel(f"{predictor_name}\n{y_label}")

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()


def _plot_predictions_linear(
        data,
        predictors,
        properties=dict(),
        obs_site_name='obs'
):
    """Plots predictions of given pyro predictors."""
    fig, axs = plt.subplots(
        nrows=len(predictors), ncols=2,
        figsize=(15, 6 * len(predictors)),
        sharey=True,
        sharex=True,
        squeeze=False,
    )

    x_col = properties.get('x', None)
    x_label = properties.get('x_label', 'x')
    y_label = properties.get('y_label', 'y')

    cat_col = properties.get('category', None)
    cat_name = properties.get('category_labels', dict())
    positive_cat = cat_name.get(1, 'Positive')
    negative_cat = cat_name.get(0, 'Negative')

    x = data['x']
    y = data['y']

    for ax, (predictor_name, predictor) in zip(axs, predictors.items()):
        for category in (0, 1):
            category_idx = x[:, cat_col] == category
            x_data = x[category_idx, x_col]
            y_data = y[category_idx]

            samples = predictor(x)
            pred_summary = summary(samples)
            y_pred = pred_summary[obs_site_name]
            mu = pred_summary["_RETURN"]

            y_pred = {k: v if len(v.shape) == 1 else v.squeeze(0)
                for k, v in y_pred.items()
            }

            mu = {k: v if len(v.shape) == 1 else v.squeeze(0)
                for k, v in mu.items()
            }

            xplot, mum, mu5, mu95, ym, y5, y95, y_true = list(zip(*sorted(
                zip(
                    x_data,
                    mu["mean"],
                    mu["5%"],
                    mu["95%"],
                    y_pred["mean"],
                    y_pred["5%"],
                    y_pred["95%"],
                    y_data
                ),
                key=lambda r: r[0]
            )))
            ax[category].fill_between(
                xplot,
                y5,
                y95,
                color='orange',
                alpha=0.5,
                label="Posterior predictive distribution with 90% CI"
            )
            ax[category].fill_between(
                xplot,
                mu5,
                mu95,
                color='cornflowerblue',
                alpha=0.8,
                label="Regression line 90% CI"
            )
            ax[category].plot(
                xplot,
                mum,
                color="red",
                label="Mean output"
            )

            ax[category].plot(
                xplot, y_true,
                marker='o', ms=4,
                linestyle='', alpha=1,
                color='green',
                label="True values"
            )
            ax[category].set(
                xlabel=x_label,
                ylabel=y_label,
                title=positive_cat if category == 1 else negative_cat,
            )

        ax[0].set_ylabel(f"{predictor_name}\n{y_label}")

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()


def _plot_predictions_logistic(
        data,
        predictors,
        properties=dict(),
        obs_site_name='obs'
):
    """Plots predictions of given pyro predictors."""
    fig, axs = plt.subplots(
        nrows=len(predictors), ncols=2,
        figsize=(15, 6 * len(predictors)),
        sharey=True,
        sharex=True,
        squeeze=False,
    )

    x_col = properties.get('x', None)
    x_label = properties.get('x_label', 'x')
    y_label = properties.get('y_label', 'y')

    y_labels = properties.get('y_labels', dict())
    positive_y = y_labels.get(1, 'Positive class')
    negative_y = y_labels.get(0, 'Negative class')

    cat_col = properties.get('category', None)
    cat_name = properties.get('category_labels', dict())
    positive_cat = cat_name.get(1, 'Positive')
    negative_cat = cat_name.get(0, 'Negative')

    x = data['x']
    y = data['y']

    for ax, (predictor_name, predictor) in zip(axs, predictors.items()):
        for category in (0, 1):
            category_idx = x[:, cat_col] == category
            x_data = x[category_idx, x_col]
            y_data = y[category_idx]

            samples = predictor(x)
            pred_summary = summary(samples)
            y_pred = pred_summary[obs_site_name]

            xplot, ym, y_true = list(zip(*sorted(
                zip(
                    x_data,
                    y_pred["mean"],
                    y_data
                ),
                key=lambda r: r[0]
            )))

            y_positive_idx = (torch.Tensor(y_true) == 1.).numpy().astype('bool')
            ax[category].plot(
                np.array(xplot)[y_positive_idx], np.array(ym)[y_positive_idx],
                marker='o', ms=10,
                linestyle='', alpha=1,
                color='green',
                label=positive_y
            )
            ax[category].plot(
                np.array(xplot)[~y_positive_idx], np.array(ym)[~y_positive_idx],
                marker='o', ms=10,
                linestyle='', alpha=1,
                color='red',
                label=negative_y
            )
            ax[category].set(
                xlabel=x_label,
                ylabel=y_label,
                title=positive_cat if category == 1 else negative_cat,
            )

        ax[0].set_ylabel(f"{predictor_name}\n{y_label}")

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()


def plot_predictions(
        data,
        predictors,
        properties=dict(),
        regression='linear',
        obs_site_name='obs'
):
    if regression == 'linear':
        _plot_predictions_linear(data, predictors, properties, obs_site_name)
    elif regression == 'logistic':
        _plot_predictions_logistic(data, predictors, properties, obs_site_name)


def make_interavtive(nd, ns, data, samples, model, guide, sites):
    x_train, y_train, x_test, y_test = (
        data['x_train'][:nd, :],
        data['y_train'][:nd],
        data['x_test'],
        data['y_test']
    )
    run_svi(model, guide, x_train, y_train, verbose=False, num_iters=1000)
    svi_samples = {k: v.reshape(ns).detach().cpu()
                   for k, v in
                   Predictive(
                       model, guide=guide, num_samples=ns
                   )(x_train).items()
                   if k != "obs"}
    param_store = dict(pyro.get_param_store())
    sites_dist = {
        "intercept": norm(
            loc=param_store['loc'].detach()[7].item(),
            scale=param_store['scale'].detach()[7].item()
        ),
        "b_GRE": norm(
            loc=param_store['loc'].detach()[0].item(),
            scale=param_store['scale'].detach()[0].item()
        ),
        "b_TEOFL": norm(
            loc=param_store['loc'].detach()[1].item(),
            scale=param_store['scale'].detach()[1].item()
        ),
        "b_university": norm(
            loc=param_store['loc'].detach()[2].item(),
            scale=param_store['scale'].detach()[2].item()
        ),
        "b_SOP": norm(
            loc=param_store['loc'].detach()[3].item(),
            scale=param_store['scale'].detach()[3].item()
        ),
        "b_LOP": norm(
            loc=param_store['loc'].detach()[4].item(),
            scale=param_store['scale'].detach()[4].item()
        ),
        "b_CGPA": norm(
            loc=param_store['loc'].detach()[5].item(),
            scale=param_store['scale'].detach()[5].item()
        ),
        "b_research": norm(
            loc=param_store['loc'].detach()[6].item(),
            scale=param_store['scale'].detach()[6].item()
        ),
        'sigma': norm(loc=param_store['sigma_mean'].detach().item(),
                      scale=0.05),
    }

    plot_mpd(sites, sites_dist, samples, svi_samples)

    svi_predictive = SviPredictive(model, guide=guide, num_samples=ns,
                                   return_sites=('obs', '_RETURN'))
    properties = {
        'x': 0,
        'x_label': "GRE Score",
        'y_label': "Chance of Admit",
        'category': 6,
        'category_labels': {
            0: "No exp. in research",
            1: "Exp. in research",
        }
    }
    predictors = {
        'SVI': svi_predictive
    }
    data = {
        'x': x_test,
        'y': y_test
    }
    plot_predictions(data, predictors, properties)
