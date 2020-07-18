from __future__ import annotations

import abc
import typing as t

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import config_enumerate, SVI, TraceEnum_ELBO, infer_discrete
from pyro.infer.autoguide import AutoDelta

pyro.enable_validation(True)


def to_covariance_matrix(scales):
    return torch.stack(list(map(lambda t: torch.diag(t), scales)))


class MixtureModel(abc.ABC):
    def __init__(self, num_components: int):
        self.num_components = num_components
        self.history = {
            "loss": []
        }  # TODO: accumulate during training

    @abc.abstractmethod
    def fit(self, x: torch.Tensor) -> MixtureModel:
        """Fit posterior of k Gaussians and mixing components.

        :param x: N x D matrix of the input data points.
        :return: self
        """

    @abc.abstractmethod
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict assignment probabilities for the input points.

        :param x: N x D matrix of the input data points.
        :return: N x K matrix of probabilities where K is the number of mixing
            components of gaussians.
        """

    @abc.abstractmethod
    def log_likelihood(self, x: torch.Tensor) -> t.Union[float, torch.Tensor]:
        """Calculate log likelihood of determined posterior.

        :param x: N x D matrix of the input data points.
        :return: A scalar containing log likelihood.
        """

    @abc.abstractmethod
    def score(self, x: torch.Tensor):
        """Calculate negative log likelihood for each sample and component.

        :param x: N x D matrix of the input data points.
        :return: N x K matrix containing negative log likelihood of a sample
            being produced by each of the K components in the mixture model.
        """


class GaussianMixtureModel(MixtureModel):
    def __init__(self, num_components: int, optim_steps: int):
        super().__init__(num_components)
        self.optim_steps = optim_steps
        self.guide = None

    @config_enumerate
    def model(self, data):
        pyro.clear_param_store()
        K = self.num_components

        weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))

        with pyro.plate('components', K):
            locs = pyro.sample('locs', dist.Normal(torch.zeros(2), 2.).independent(1))
            scales = pyro.sample('scales', dist.LogNormal(torch.zeros(2), 3.).independent(1))
        scales = to_covariance_matrix(scales)

        with pyro.plate('data', data.shape[0]):
            assignment = pyro.sample('assignment', dist.Categorical(weights))
            obs = pyro.sample('obs',
                              dist.MultivariateNormal(locs[assignment], scales[assignment]),
                              obs=data)
        return obs

    def fit(self, x: torch.Tensor) -> MixtureModel:
        def init_loc_fn(site):
            K = self.num_components
            if site["name"] == "weights":
                return torch.ones(K) / K
            if site["name"] == "scales":
                return torch.tensor([[(x.var() / 2).sqrt()] * 2] * K)
            if site["name"] == "locs":
                return x[torch.multinomial(torch.ones(x.shape[0]) / x.shape[0], K), :]
            raise ValueError(site["name"])

        self.guide = AutoDelta(poutine.block(self.model, expose=['weights', 'locs', 'scales']),
                               init_loc_fn=init_loc_fn)

        optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
        loss = TraceEnum_ELBO(max_plate_nesting=1)

        svi = SVI(self.model, self.guide, optim, loss=loss)

        for i in range(self.optim_steps):
            elbo = svi.step(x)
            self.history["loss"].append(elbo)
        return self

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        K = self.num_components
        trace = self.get_infered_model_trace(x)
        probs = torch.zeros((x.shape[0], K))
        for i, v in enumerate(trace.nodes["assignment"]["value"]):
            probs[i, v] = 1
        return probs

    def log_likelihood(self, x: torch.Tensor) -> t.Union[float, torch.Tensor]:
        return self.get_infered_model_trace(x).log_prob_sum()

    def get_infered_model_trace(self, x):
        guide_trace = poutine.trace(self.guide).get_trace(x)
        trained_model = poutine.replay(self.model, trace=guide_trace)
        inferred_model = infer_discrete(trained_model, temperature=0, first_available_dim=-2)
        return poutine.trace(inferred_model).get_trace(x)

    def score(self, x: torch.Tensor):
        map_estimates = self.guide(x)

        locs = map_estimates['locs']
        scales = to_covariance_matrix(map_estimates['scales'])

        N, D = x.shape
        K = self.num_components
        result = torch.zeros((N, K))

        for k in range(K):
            loc = locs[k]
            scale = scales[k]
            result[:, k] = torch.distributions.MultivariateNormal(loc, scale).log_prob(x)

        return result * (-1)
