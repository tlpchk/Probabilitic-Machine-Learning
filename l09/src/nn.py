import torch
import torch.nn as nn


# implementation of own activation sigmoid
class OwnSigmoid(nn.Module):
    @classmethod
    def forward(cls, x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + (-x).exp())


# implementation of softmax
class OwnSoftmax(nn.Module):
    @classmethod
    def forward(cls, x: torch.Tensor) -> torch.Tensor:
        numerator = x.exp()
        denominator = numerator.sum(dim=1, keepdim=True)
        # keepdim is important to keep the same dimensions as numerator
        # otherwise the division wouldn't work
        return numerator / denominator


# implementation of mse loss
class MSELoss(nn.Module):
    @classmethod
    def forward(
        cls, y_predicted: torch.Tensor, y_ground_truth: torch.Tensor
    ) -> torch.Tensor:
        squared_differences = (y_predicted - y_ground_truth).pow(2)
        sum_over_classes = squared_differences.sum(dim=-1)
        average_over_samples = sum_over_classes.mean(dim=0)
        return average_over_samples
