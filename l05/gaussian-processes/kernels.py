import abc
import math
import typing as t

import matplotlib.pyplot as plt
import torch


class Kernel(abc.ABC):
    @abc.abstractmethod
    def apply(
            self,
            points: torch.Tensor,
            other_points: t.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transforms input data to obtain covariance matrix.

        Covariance matrix is used in the gaussian distribution over functions.
        :param points: Tensor of shape N x D, where N is the number of
            samples and D is the number of dimensions.
        :param other_points: Tensor of shape N' x D, where N' is the number of
            samples and D is the number of dimensions.
        :return: Matrix of size N x N'. If `other_points` is `None`, then the
            method should produce matrix N x N.
        """

    def visualize(self, num_points: int) -> plt.Axes:
        """Visualize kernel calculation.

        Example results are shown in
        https://mlss2011.comp.nus.edu.sg/uploads/Site/lect1gp.pdf.
        :param num_points: Number of points to generate.
        :return: Plot figure with kernel working
        """
        x = torch.arange(num_points).unsqueeze(-1).float()
        num_samples = x.shape[0]
        results = (
            self.apply(x)
                .reshape((num_samples, num_samples))
                .detach()
                .cpu()
                .numpy()
        )

        fig, ax = plt.subplots(1, 1)
        plt.imshow(results)

        return fig


class RBFKernel(Kernel):
    def __init__(self, variance: float, lengthscale: float):
        self.lengthscale = lengthscale
        self.variance = variance

    def apply(
            self,
            points: torch.Tensor,
            other_points: t.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if other_points is None:
            other_points = points
        result = torch.zeros((points.shape[0], other_points.shape[0]))
        for i, point in enumerate(points):
            diff = point - other_points
            norm = torch.norm(diff, dim=-1)
            result[i, :] = self.variance ** 2 * torch.exp(norm.pow(2) / (-2 * self.lengthscale ** 2))
        return result


class PeriodicKernel(Kernel):
    def __init__(
            self, lengthscale: float, periodicty: float, deviation: float
    ):
        self.lengthscale = lengthscale
        self.periodicity = periodicty
        self.deviation = deviation

    def apply(
            self,
            points: torch.Tensor,
            other_points: t.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if other_points is None:
            other_points = points
        result = torch.zeros((points.shape[0], other_points.shape[0]))
        for i, point in enumerate(points):
            diff = point - other_points
            norm = torch.norm(diff, dim=-1)
            numerator = 2 * torch.sin(math.pi * norm / self.periodicity).pow(2)
            denominator = self.lengthscale ** 2
            result[i, :] = self.deviation ** 2 * torch.exp(-numerator / denominator)
        return result


class KernelCombiner(Kernel):
    def __init__(self, *kernels: Kernel):
        self.kernels = kernels

    def apply(
            self,
            points: torch.Tensor,
            other_points: t.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if other_points is None:
            other_points = points

        result = torch.ones((points.shape[0], other_points.shape[0]))
        for kernel in self.kernels:
            result = result * kernel.apply(points,other_points)
        return result
