import pyro
import torch

from src.vae import VariationalAutoEncoder


class BetaVariationalAutoEncoder(VariationalAutoEncoder):
    """beta-VAE model."""

    def __init__(self, beta: float, **kwargs):
        """ Implement beta Variational Autoencoder model.

        .. note: try to reuse as much as you can from VariationalAutoEncoder class

        :param n_data_features: number of input and output features (28 x 28 = 784 for MNIST)
        :param n_encoder_hidden_features: number of neurons in encoder's hidden layer
        :param n_decoder_hidden_features: number of neurons in decoder's hid den layer
        :param n_latent_features: number of latent features
        :param beta: regularization coefficient
        """
        self.beta = beta
        super().__init__(**kwargs)

    def encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_forward = pyro.poutine.scale(super().encoder_forward, scale=self.beta)
        return scaled_forward(x)
