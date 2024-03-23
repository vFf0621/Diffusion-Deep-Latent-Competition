
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """ Training configuration for the DDIM model.
    """
    image_size = 96  # the generated image resolution
    train_batch_size = 32
    eval_batch_size = 4  # how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-3
    lr_warmup_steps = 200
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddim"  # the model name locally and on the HF Hub
    time_embedding_type = 'fourier'
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0