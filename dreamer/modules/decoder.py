import torch.nn as nn
import torch
from dreamer.utils.utils import (
    initialize_weights,
    horizontal_forward,
    create_normal_dist,
    ImgChLayerNorm
)

'''

The decoder model takes the state and the new action and uses it to reconstruct
the scene. It is a convolution transpose model. 

The dimensions are hard coded so that it correctly resembles the input dim; the one 
provided in the SimpleDreamer code does not give a correct dimension.

Our contribution here was adding normalization layers and LeakyReLU activation.
The normalization layers are there to stabilize the training of the model.
The LeakyReLU layer ensures that we do not have to deal with the vanishing gradient
problem during training, the activation function is also there to learn features of the 
environment that are encoded with negative numbers. 

Also, the standard deviation of the output is now a learned single parameter rather than
the output of a layer, simiplifying the design.

'''

class Decoder(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.parameters.dreamer.decoder
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        activation = getattr(nn, self.config.activation)()
        self.observation_shape = observation_shape

        self.network = nn.Sequential(
            nn.Unflatten(dim = 1, unflattened_size=(-1, 1, 1)),
            nn.ConvTranspose2d(
                1024,
                self.config.depth * 4,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            ImgChLayerNorm(self.config.depth * 4),
            nn.ConvTranspose2d(
                self.config.depth * 4,
                self.config.depth * 2,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            ImgChLayerNorm(self.config.depth * 2),

            nn.ConvTranspose2d(
                self.config.depth * 2,
                self.config.depth * 1,
                self.config.kernel_size+1,
                self.config.stride,
                output_padding=(1, 1)
            ),
            activation,
            ImgChLayerNorm(self.config.depth * 1),

            nn.ConvTranspose2d(
                self.config.depth * 1,
                self.observation_shape[0],
                self.config.kernel_size+1,
                self.config.stride+1,
            ),
            )
        self.linear = nn.Sequential(nn.Linear(config.parameters.dreamer.deterministic_size + config.parameters.dreamer.stochastic_size, 1024),
                                    activation,
                                    nn.LayerNorm(1024))
        self.network.apply(initialize_weights)

    def forward(self, posterior, deterministic, seq=0):
        if seq:
            seq_len = posterior.shape[0]
            batch_size = posterior.shape[1]
            posterior = posterior.reshape(-1, posterior.shape[-1])
            deterministic = deterministic.reshape(-1, deterministic.shape[-1])

        x = torch.cat([posterior, deterministic], -1)
        x = self.linear(x)
        x = self.network(x)
        if seq:
            x = x.reshape(seq_len, batch_size, *self.observation_shape)
        dist = create_normal_dist(x, std=0.3, event_shape=len(self.observation_shape))
        return dist
