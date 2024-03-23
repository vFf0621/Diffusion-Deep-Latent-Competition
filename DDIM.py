# Note: The code for training the diffusion model in this project (TrainingConfig, TrainingLoop, this file) is based on the code from the following tutorial (also cited in the paper):
# https://huggingface.co/docs/diffusers/v0.23.1/tutorials/basic_training

import torch
from TrainingConfig import TrainingConfig
from diffusers import UNet2DModel, DDIMScheduler
from accelerate import notebook_launcher
from torch.utils.data import DataLoader
from TrainingLoop import train_loop, evaluate
from diffusers.optimization import get_cosine_schedule_with_warmup


def train_diffusion_model():
    """
    Create a UNet2D model for image diffusion and trains it on the car simulation data using the DDIMScheduler.
    """
    num_train_timesteps = 1000
    
    config = TrainingConfig()
    
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 256,512),  # the number of output channels for each UNet block
        attention_head_dim=16,
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    noise_gen = torch.manual_seed(config.seed)
    
    train_dataset = torch.load("pretrained_parameters/ds_1")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    noise_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, noise_gen, num_train_timesteps)
    
    notebook_launcher(train_loop, args, num_processes=1)


if __name__ == '__main__':
    train_diffusion_model()
    
    