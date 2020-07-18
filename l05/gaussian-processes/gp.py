from __future__ import annotations

import abc
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions
from kernels import Kernel


class Regressor(abc.ABC):
    @abc.abstractmethod
    def fit_train(self, x: torch.Tensor, y: torch.Tensor) -> Regressor:
        """Fit to the training data.

        :param x: N x D matrix of the input data points.
        :param y: N x K matrix of targets for K predicted values.
        :return: self
        """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> t.Any:
        """Predict value for each point in the input data.

        It also should save parameters of the multivariate distribution for
        given points so it can be used `sample_fun` method.

        :param x: N x D matrix of the input data points.
        :return: Anything that you need.
        """

    @abc.abstractmethod
    def confidence_interval(self, x: torch.Tensor) -> torch.Tensor:
        """For each sampling point return its upper and lower confidences.

        It should be realised .95 confidence interval.
        :param x: Input data points of N' x D dimensions.
        :return: Confidence intervals of N' x 2 (lower and upper bounds).
        """

    @abc.abstractmethod
    def log_likelihood(self) -> t.Union[float, torch.Tensor]:
        """Calculate log probability of the fitted model.

        :return: scalar with the log likelihood value.
        """

    @abc.abstractmethod
    def sample_fun(self):
        """Samples single function using test data.

        Test data should be used in `forward` method before calling this
        method.
        :return: M x 1 function values where M is number of data points used
            in the `forward` method.
        """


class GPRegressor(Regressor):
    def __init__(self, kernel: Kernel, noise: float, jitter: float):
        self.x_train = None
        self.y_train = None
        self.mu = None
        self.cov = None

        self.kernel = kernel
        self.noise = noise
        self.jitter = jitter

    def fit_train(self, x: torch.Tensor, y: torch.Tensor) -> Regressor:
        self.mu = torch.zeros(x.shape)
        self.cov = self.kernel.apply(x)

        self.x_train = x
        self.y_train = y

        return self

    def forward(self, x: torch.Tensor) -> t.Any:
        K = self.kernel.apply(self.x_train) + self.jitter * torch.eye(len(self.x_train))
        K_s = self.kernel.apply(x, self.x_train)
        K_ss = self.kernel.apply(x)
        K_inv = torch.inverse(K)

        # L = K.cholesky()  # cholesky decomposition
        # partial = self.y_train.triangular_solve(L, upper=False)[0]
        # alpha = partial.triangular_solve(L, upper=False, transpose=True)[0]  # final results

        self.mu = K_s @ K_inv @ self.y_train
        self.cov = K_ss - (K_s @ K_inv @ K_s.t())

    def confidence_interval(self, x: torch.Tensor) -> torch.Tensor:
        self.forward(x)
        uncertainty = 1.96 * torch.sqrt(torch.diag(self.cov)) # TODO: kwantyl ?
        intervals = torch.stack([self.mu + uncertainty, self.mu - uncertainty]).t()
        return intervals

    def log_likelihood(self, *args) -> t.Union[float, torch.Tensor]:
        cov = self.cov + self.noise * torch.eye(len(self.cov))
        return torch.distributions.MultivariateNormal(self.mu, cov).log_prob(self.y_train).sum()

    def sample_fun(self):
        cov = self.cov + torch.eye(len(self.cov)) * self.jitter
        return torch.distributions.MultivariateNormal(self.mu, cov).sample()


def visualize_data_and_intervals(
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        data_points_to_visualize: torch.Tensor,
        model: GPRegressor,
) -> plt.Axes:
    intervals = model.confidence_interval(data_points_to_visualize)
    mean = intervals.mean(dim=1)
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.plot(data_points_to_visualize.squeeze(), mean, color="blue", lw=2)
    ax.scatter(x_train, y_train.squeeze(), color="green", label="Train")
    ax.scatter(x_test, y_test.squeeze(), color="orange", label="Test")
    ax.fill_between(
        data_points_to_visualize.squeeze(),
        intervals[:, 0],
        intervals[:, 1],
        alpha=0.3,
    )
    ax.legend()
    return ax


def visualize_data_samplings(
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        data_points_to_visualize: torch.Tensor,
        model: GPRegressor,
        num_iterations: int,
) -> plt.Axes:
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    model.forward(data_points_to_visualize)
    for _ in range(num_iterations):
        ax.plot(
            data_points_to_visualize, model.sample_fun(), color="blue", lw=2
        )

    intervals = model.confidence_interval(data_points_to_visualize)
    ax.scatter(x_train, y_train, color="green", label="Train")
    ax.scatter(x_test, y_test, color="orange", label="Test")
    ax.fill_between(
        data_points_to_visualize.squeeze(),
        intervals[:, 0],
        intervals[:, 1],
        alpha=0.3,
    )
    ax.legend()
    return ax
