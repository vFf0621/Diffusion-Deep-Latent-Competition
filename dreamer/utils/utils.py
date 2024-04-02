
def attrdict_monkeypatch_fix():
    import collections
    import collections.abc
    for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))
attrdict_monkeypatch_fix()
import numpy as np

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
from attrdict import AttrDict
class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def horizontal_forward(network, x, y=None, input_shape=(-1,), output_shape=(-1,)):
    batch_with_horizon_shape = x.shape[: -len(input_shape)]
    if not batch_with_horizon_shape:
        batch_with_horizon_shape = (1,)
    if y is not None:
        x = torch.cat((x, y), -1)
        input_shape = (x.shape[-1],)  #
    x = x.reshape(-1, *input_shape)
    x = network(x)

    x = x.reshape(*batch_with_horizon_shape, *output_shape)
    return x

def get_gaussian_kernel(kernel_size=3, sigma=0.3, channels=3):
    """
    Generate a 2D Gaussian kernel array.
    """
    # Create a coordinate grid
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    
    mean = (kernel_size - 1) / 2.
    variance = sigma**2.
    
    # Calculate the 2-dimensional gaussian kernel
    gaussian_kernel = (1./(2.*np.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )
    # Normalize the kernel
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
    # Add channels
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    
    gaussian_kernel.requires_grad = False
    
    return gaussian_kernel

class GaussianFilterLayer(nn.Module):
    def __init__(self, kernel_size=3, sigma=0.21, channels=3):
        super(GaussianFilterLayer, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        self.padding = kernel_size // 2

        # Get the Gaussian kernel
        self.gaussian_kernel = get_gaussian_kernel(kernel_size, sigma, channels)

        # Make sure it's not trainable
        self.weight = nn.Parameter(self.gaussian_kernel, requires_grad=False)
    
    def forward(self, x):
        """
        Apply the Gaussian filter to the input tensor.
        """
        return torch.nn.functional.conv2d(x, self.weight, padding=self.padding, groups=self.channels)


class Disc(nn.Module):
    def __init__(self, input_size):
        super(Disc, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )
        
    
    def forward(self, x):
        logits=self.network(x)
        return logits


def build_network(input_size, hidden_size, num_layers, activation, output_size):
    assert num_layers >= 2, "num_layers must be at least 2"
    activation = getattr(nn, activation)()
    layers = []
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(nn.LayerNorm(hidden_size))
    layers.append(activation)

    for i in range(num_layers - 2):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(activation)

    layers.append(nn.Linear(hidden_size, output_size))

    network = nn.Sequential(*layers)
    network.apply(initialize_weights)
    return network


def initialize_weights(m):
    given_scale = 1
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = given_scale / denoms
        limit = np.sqrt(3 * scale)
        nn.init.uniform_(m.weight.data, a=-limit, b=limit)
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = given_scale / denoms
        limit = np.sqrt(3 * scale)
        nn.init.uniform_(m.weight.data, a=-limit, b=limit)
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
           m.bias.data.fill_(0.0)
    


def create_normal_dist(
    x,
    std=None,
    mean_scale=1,
    init_std=1,
    min_std=0.1,
    activation=nn.Identity(),
    event_shape=None,
):
    if std == None:
        mean, std = torch.chunk(x, 2, -1)
        mean = mean / mean_scale
        if activation:
            mean = activation(mean)
        mean = mean_scale * mean
        std = init_std - F.softplus(init_std - std)
        std = min_std + F.softplus(std - min_std)
    else:
        mean = x
    dist = torch.distributions.Normal(mean, std)
    if event_shape:
        dist = torch.distributions.Independent(dist, event_shape)
    return dist


def compute_lambda_values(rewards, values, continues, horizon_length, device, log_probs, lambda_, alpha = 1):
    """
    rewards : (batch_size, time_step, hidden_size)
    values : (batch_size, time_step, hidden_size)
    continue flag will be added
    """
    rewards = rewards[:, :-1]
    continues = continues[:, :-1]
    next_values = values[:, 1:]
    last = next_values[:, -1]
    log_probs = log_probs[:, :-1]
    inputs = rewards + continues * next_values * (1 - lambda_)

    outputs = []
    # single step
    for index in reversed(range(horizon_length - 1)):
        last = inputs[:, index] + continues[:, index] * lambda_ * (last-alpha*log_probs[:, index])
        outputs.append(last)
    returns = torch.stack(list(reversed(outputs)), dim=-1).to(device)
    return returns


class DynamicInfos:
    def __init__(self, device):
        self.device = device
        self.data = {}

    def append(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def get_stacked(self, time_axis=1):
        copy = {key:self.data[key] for key in self.data}
        for key in copy:
            if isinstance(copy[key][0], torch.Tensor):
                copy[key] = torch.stack(copy[key], axis=time_axis).to(self.device)
        stacked_data = AttrDict(
            copy
        )
        self.clear()
        return stacked_data

    def clear(self):
        self.data = {}


def find_file(file_name):
    cur_dir = os.getcwd()

    for root, dirs, files in os.walk(cur_dir):
        if file_name in files:
            return os.path.join(root, file_name)

    raise FileNotFoundError(
        f"File '{file_name}' not found in subdirectories of {cur_dir}"
    )
    
def find_dir(file_name):
    cur_dir = os.getcwd()

    for root, dirs, files in os.walk(cur_dir):
        if file_name in dirs:
            return os.path.join(root, file_name)

    raise FileNotFoundError(
        f"File '{file_name}' not found in subdirectories of {cur_dir}"
    )


def get_base_directory():
    return "/".join(find_file("main.py").split("/")[:-1])


def load_config(config_path):
    if not config_path.endswith(".yml"):
        config_path += ".yml"
    config_path = find_file(config_path)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return AttrDict(config)
