# Standard libraries
import os

# PyTorch
import torch

# Declare project constants
AVAIL_GPUS = min(1, torch.cuda.device_count())  # Number of GPUs to use
BATCH_SIZE = 64 if AVAIL_GPUS else 1  # Batch size for data loaders
RAND_SEED = 42  # Random seed for reproducibility
DATASET_PATH = '~/data/'  # Path to the folder where the datasets are/should be downloaded
CHECKPOINT_BASE_PATH = '~/savedmodels/'  # Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.path.join(CHECKPOINT_BASE_PATH, "NNs/")  # Where to save checkpoints
NUM_CLASSES = 10  # Number of classes in the MNIST dataset
