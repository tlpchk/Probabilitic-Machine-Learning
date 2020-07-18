from copy import deepcopy as copy
from scipy.stats import t as tstudent
import torch
from IPython.display import Code, display
import inspect
import pyro
from pyro import optim
import logging
from pyro.infer import MCMC, Predictive, SVI, Trace_ELBO


def make_ols(x_train, y_train, x_test, alpha=0.1):
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    p = x_train.shape[1] + 1
    x_train_intercept = torch.ones(n_train, p)
    x_train_intercept[:, 1:] = x_train
    x_test_intercept = torch.ones(n_test, p)
    x_test_intercept[:, 1:] = x_test

    gramian_matrix = torch.matmul(x_train_intercept.t(), x_train_intercept)
    inv_xx = torch.inverse(gramian_matrix)

    beta = torch.matmul(
        torch.matmul(inv_xx, x_train_intercept.t()),
        y_train
    )

    y_hat_train = torch.matmul(beta, x_train_intercept.t())
    y_hat_test = torch.matmul(beta, x_test_intercept.t())

    sigma = torch.sqrt(
        torch.sum((y_train - y_hat_train) ** 2) / (y_train.shape[0] - 1)
    ).item()

    q_bottom = tstudent(n_train - p).ppf(alpha / 2)
    q_up = tstudent(n_train - p).ppf(1 - alpha / 2)

    ci_train = []
    ci_test = []
    pi_train = []
    pi_test = []

    for idx, row in enumerate(x_train_intercept):
        var_mean = torch.matmul(
            torch.matmul(
                row,
                inv_xx
            ),
            row.t()
        )
        ci_train.append((
            y_hat_train[idx].item() + q_bottom * sigma * torch.sqrt(
                var_mean).item(),
            y_hat_train[idx].item() + q_up * sigma * torch.sqrt(var_mean).item()
        ))
        pi_train.append((
            y_hat_train[idx].item() + q_bottom * sigma * torch.sqrt(
                var_mean + 1).item(),
            y_hat_train[idx].item() + q_up * sigma * torch.sqrt(
                var_mean + 1).item()
        ))

    for idx, row in enumerate(x_test_intercept):
        var_mean = torch.matmul(
            torch.matmul(
                row,
                inv_xx
            ),
            row.t()
        )
        ci_test.append((
            y_hat_test[idx].item() + q_bottom * sigma * torch.sqrt(
                var_mean).item(),
            y_hat_test[idx].item() + q_up * sigma * torch.sqrt(var_mean).item()
        ))
        pi_test.append((
            y_hat_test[idx].item() + q_bottom * sigma * torch.sqrt(
                var_mean + 1).item(),
            y_hat_test[idx].item() + q_up * sigma * torch.sqrt(
                var_mean + 1).item()
        ))

    return (
        (y_hat_train, torch.tensor(ci_train), torch.tensor(pi_train)),
        (y_hat_test, torch.tensor(ci_test), torch.tensor(pi_test)),
        beta,
    )


def display_sourcecode(fun):
    display(Code(inspect.getsource(fun), language='python3'))


def run_mcmc(kernel, model_args, num_samples, warmup_steps=None, num_chains=1):
    """Executes MCMC using given `kernel` and `data`."""
    pyro.clear_param_store()

    mcmc_posterior = pyro.infer.MCMC(
        kernel=kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains
    )

    mcmc_posterior.run(*model_args)
    return mcmc_posterior


def sample_mcmc(mcmc_posterior):
    return {
        k: v.detach().cpu()
        for k, v in mcmc_posterior.get_samples().items()
    }


def summary(samples, qs = (0.05, 0.95)):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            **{
                f'{q*100:.0f}%': v.kthvalue(int(len(v) * q), dim=0)[0]
                for q in qs
            }
        }
    return site_stats


def run_svi(model, guide, x, y, optimizer=pyro.optim.Adam({"lr": 0.03}),
            loss=Trace_ELBO(), num_iters=5000, verbose=True):
    pyro.clear_param_store()
    svi = SVI(model,
          guide,
          optimizer,
          loss)
    elbos = []
    params = []
    for i in range(num_iters):
        elbo = svi.step(x, y)
        if verbose and i % 500 == 0:
            logging.info("Elbo loss: {}".format(elbo))
        elbos.append(elbo)
        params.append(dict(copy(pyro.get_param_store())))

    return elbos, params


class SviPredictive(Predictive):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)

    def forward(self, *args, **kwargs):
        return {k: v.detach() for k, v in
                super().forward(*args, **kwargs).items()}


def print_summary(model_summary):
    for site, values in model_summary.items():
        print("Site: {}".format(site))
        print({k: v.item() for k,v in values.items()})


def sample_svi(model, guide, num_samples, x):
    return {
        k: v.reshape(num_samples).detach().cpu()
        for k, v in Predictive(
            model,
            guide=guide,
            num_samples=num_samples
        )(x).items() if k != "obs"
    }


def generator2d(gen_model, num_samples, num_datasets=1, **kwargs):
    data = Predictive(
        gen_model,
        num_samples=num_datasets,
        return_sites=('x', 'y')
    )(num_samples=num_samples, **kwargs)
    return torch.cat([data['x'], data['y']]).t()
