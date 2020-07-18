"""Implementation of speedometer model."""
import torch
import pyro
import pyro.distributions as dist


def speedometer_model(speed_limit, is_cold, measurement=None):
    """Model for measuring vehicle speed given `speed_limit` and `is_cold`."""
    #     c_means = {0: torch.tensor(0.99), 1: torch.tensor(0.75)}
    #     measurement_stds = {0: torch.tensor(10.), 1: torch.tensor(5.)}
    c_means = {0: torch.tensor(0.2), 1: torch.tensor(0.1)}
    measurement_stds = {0: torch.tensor(1.), 1: torch.tensor(2.)}

    c = pyro.sample('c', dist.Normal(c_means[is_cold], 0.01))
    std = pyro.sample('std', dist.Normal(measurement_stds[is_cold], 2.))

    with pyro.plate("data", len(speed_limit)):
        return pyro.sample(
            "measurement",
            dist.Normal(c * speed_limit, std),
            obs=measurement,
        )
