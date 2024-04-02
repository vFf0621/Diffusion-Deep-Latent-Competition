import torch
import torch.nn as nn

from dreamer.utils.utils import (
    initialize_weights,
    horizontal_forward,
    ImgChLayerNorm
)

'''

The encoder is used to encode the observations into a latent space 
then fed into the representation model. It is a CNN model.

Our contribution was the addition of batch normalization layers to 
ensure stability in training.

'''

class Encoder(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.parameters.dreamer.encoder

        activation = getattr(nn, self.config.activation)()
        self.observation_shape = observation_shape
        self.activation = activation
        self.network = nn.Sequential(
            ImgChLayerNorm(observation_shape[0]),

            nn.Conv2d(
                self.observation_shape[0],
                self.config.depth * 1,
                self.config.kernel_size,
                self.config.stride+1,
                bias=False

            ),
            ImgChLayerNorm(self.config.depth * 1),
            activation,

            nn.Conv2d(
                self.config.depth * 1,
                self.config.depth * 2,
                self.config.kernel_size,
                self.config.stride,
                bias=False

            ),
            ImgChLayerNorm(self.config.depth * 2),
            activation,

            nn.Conv2d(
                self.config.depth * 2,
                self.config.depth * 4,
                self.config.kernel_size,
                self.config.stride,
                bias=False
            ),
            ImgChLayerNorm(self.config.depth * 4),

            activation,

            nn.Conv2d(
                self.config.depth * 4,
                self.config.depth * 8,
                self.config.kernel_size,
                self.config.stride,
                bias=False
            ),
            ImgChLayerNorm(self.config.depth * 8),
            activation,

        )
        z_shape=config.parameters.dreamer.embedded_state_size
        self.bn = nn.LayerNorm(z_shape)
        self.fc = nn.Linear(1024, z_shape)
        self.network.apply(initialize_weights)

    def forward(self, x, seq=0):
        if seq:
            seq_len = x.shape[0]
            batch_size = x.shape[1]
            x = x.reshape(-1, *self.observation_shape)
        elif len(x.shape) < 4:
            x = x.unsqueeze(0)
        y = torch.flatten(self.network(x)).view(-1)

        y = self.fc(y.reshape(-1, 1024)).squeeze(0)
        y = self.bn(y)

        y = self.activation(y)

        if seq:
            y = y.reshape(seq_len, batch_size, -1)
        return y
    


class ImagEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()  
        self.config = config.parameters.dreamer.encoder

        self.activation = nn.SiLU()
        self.z_shape = z_shape=config.parameters.dreamer.embedded_state_size
        self.bn = nn.LayerNorm(z_shape // 2)
        self.bn1 = nn.LayerNorm(z_shape)
        self.bn2 = nn.LayerNorm(z_shape)

        self.fc = nn.Linear(z_shape, z_shape//2, bias=False)
        self.fc2 = nn.Linear(z_shape//2, z_shape)
        self.fc.apply(initialize_weights)
        self.fc2.apply(initialize_weights)

    def forward(self, x, seq=0):
        batch_size = x.shape[0]
        y0 = self.fc(x.reshape(-1, self.z_shape)).squeeze(0)
        y= self.bn(y0)

        y = self.activation(y)
        y = self.fc2(y)+ x
        y = self.bn1(y)

        y = self.activation(y) 
        if seq:
            y = y.reshape(batch_size, -1)
        return y