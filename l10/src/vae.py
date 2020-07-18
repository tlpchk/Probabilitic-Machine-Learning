from typing import Tuple

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn

from src.ae import BaseAutoEncoder

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)


class VEncoder(nn.Module):
    """Encoder for VAE."""

    def __init__(
            self, n_input_features: int, n_hidden_neurons: int, n_latent_features: int,
    ):
        """ Implement Encoder neural network with given params.

        :param n_input_features: number of input features (28 x 28 = 784 for MNIST)
        :param n_hidden_neurons: number of neurons in hidden FC layer
        :param n_latent_features: size of the latent vector
        """
        super().__init__()

        self.layer_input_to_hidden = nn.Linear(n_input_features, n_hidden_neurons)
        self.loc_layer_hidden_to_latent = nn.Linear(n_hidden_neurons, n_latent_features)
        self.scale_layer_hidden_to_latent = nn.Linear(n_hidden_neurons, n_latent_features)
        self.nonlinear = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implement encoding data to gaussian distribution params."""
        hidden = self.layer_input_to_hidden(x)
        hidden = self.nonlinear(hidden)
        z_loc = self.loc_layer_hidden_to_latent(hidden)
        z_scale = torch.exp(self.scale_layer_hidden_to_latent(hidden))
        return z_loc, z_scale


class VDecoder(nn.Module):
    """Decoder for VAE."""

    def __init__(
            self, n_latent_features: int, n_hidden_neurons: int, n_output_features: int,
    ):
        """ Implement Decoder neural network with given params.

        :param n_latent_features: number of latent features (same as in Encoder)
        :param n_hidden_neurons: number of neurons in hidden FC layer
        :param n_output_features: size of the output vector (28 x 28 = 784 for MNIST)
        """
        super().__init__()

        self.layer_latent_to_hidden = nn.Linear(n_latent_features, n_hidden_neurons)
        self.hidden_nonlinear = nn.Tanh()
        self.layer_hidden_to_output = nn.Linear(n_hidden_neurons, n_output_features)
        self.output_nonlinear = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Implement decoding latent vector to image."""
        hidden = self.layer_latent_to_hidden(z)
        hidden = self.hidden_nonlinear(hidden)
        loc = self.layer_hidden_to_output(hidden)
        loc = self.output_nonlinear(loc)
        return loc


class VariationalAutoEncoder(BaseAutoEncoder):
    """Variational Auto Encoder model."""

    def __init__(
            self,
            n_data_features: int,
            n_encoder_hidden_features: int,
            n_decoder_hidden_features: int,
            n_latent_features: int,
    ):
        """ Implement Variational Autoencoder with Pyro tools.

        :param n_data_features: number of input and output features (28 x 28 = 784 for MNIST)
        :param n_encoder_hidden_features: number of neurons in encoder's hidden layer
        :param n_decoder_hidden_features: number of neurons in decoder's hidden layer
        :param n_latent_features: number of latent features
        """
        encoder = VEncoder(
            n_input_features=n_data_features,
            n_hidden_neurons=n_encoder_hidden_features,
            n_latent_features=n_latent_features,
        )
        decoder = VDecoder(
            n_latent_features=n_latent_features,
            n_hidden_neurons=n_decoder_hidden_features,
            n_output_features=n_data_features,
        )
        super().__init__(
            encoder=encoder, decoder=decoder, n_latent_features=n_latent_features
        )

    def model(self, x: torch.Tensor):
        """Implement Pyro model for VAE; p(x|z)p(z)."""
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = torch.zeros((x.shape[0], self.n_latent_features))
            z_scale = torch.ones((x.shape[0], self.n_latent_features))

            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
            loc_img = self.decoder_forward(z)

            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x)

    def guide(self, x: torch.Tensor):
        """Implement Pyro guide for VAE; q(z|x)"""
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            self.encoder_forward(x)

    def encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Implement function to perform forward pass through encoder network.

        takes: tensor of shape [batch_size x input_flattened_size] (flattened input)
        returns: tensor of shape [batch_size x latent_feature_size] (latent vector)
        """
        z_loc, z_scale = self.encoder.forward(x)
        return pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

    def decoder_forward(self, z: torch.Tensor) -> torch.Tensor:
        """ Implement unction to perform forward pass through decoder network.

        takes: tensor of shape [batch_size x latent_feature_size] (latent vector)
        returns: tensor of shape [batch_size x output_flattened_size] (flattened output)
        """
        return self.decoder.forward(z)
