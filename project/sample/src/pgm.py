from pyro import distributions as dist
from torch.distributions import constraints

from src.utils import *

pyro.enable_validation(True)


def model(x, verbose=False):
    # Parameters
    theta_D = pyro.param(
        'theta_D',
        torch.tensor(0.5),
        constraint=constraints.interval(0, 1),
    )
    theta_I = pyro.param(
        'theta_I',
        torch.tensor(0.5),
        constraint=constraints.interval(0, 1),
    )

    theta_S = pyro.param(
        'theta_S',
        torch.tensor([0.5, 0.5]),
        constraint=constraints.interval(0, 1),
    )

    theta_G = pyro.param(
        'theta_G',
        torch.ones(2, 2, 3).div(3),
        constraint=constraints.simplex,
    )

    theta_L = pyro.param(
        'theta_L',
        torch.tensor([0.5, 0.5, 0.5]),
        constraint=constraints.interval(0, 1),
    )

    # Forward
    with pyro.plate('data', x.shape[0]):
        d = pyro.sample('Difficulty', dist.Bernoulli(probs=theta_D), obs=x.d).long()
        i = pyro.sample('Intelligence', dist.Bernoulli(probs=theta_I), obs=x.i).long()

        s = pyro.sample('SAT', dist.Bernoulli(probs=theta_S[i]), obs=x.s).long()
        # Grade not observed but enumerated
        g = pyro.sample(
            'Grade',
            dist.Categorical(probs=theta_G[i, d]),
            infer={"enumerate": "parallel"}
        ).long()

        l = pyro.sample('Letter', dist.Bernoulli(probs=theta_L[g]), obs=x.l).long()


def guide(x, verbose=False):
    pass