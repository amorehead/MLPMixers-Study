# Standard libraries
import os

# PyTorch Lightning
import pytorch_lightning as pl
# PyTorch
import torch
# PL callbacks
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST

# Project utilities
from constants import RAND_SEED, DATASET_PATH, CHECKPOINT_BASE_PATH, CHECKPOINT_PATH, BATCH_SIZE, AVAIL_GPUS
from modules import LitMLPMixer, LitMLP, LitCNN

"""Declare system hyperparameters"""
# Set the seed
pl.seed_everything(RAND_SEED)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create checkpoint path if it doesn't exist yet
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_BASE_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Define requested model and dataset
model_name = 'MLPMixer'  # Choices are ['MLP', 'MLPMixer', 'CNN']
dataset_name = 'MNIST'  # Choices are ['MNIST', 'FASHION-MNIST']

# Init requested dataset
dataset = FashionMNIST if 'fashion' in dataset_name.strip().lower() else MNIST
train_ds = dataset(DATASET_PATH, train=True, download=True, transform=transforms.ToTensor())
val_ds = dataset(DATASET_PATH, train=False, download=True, transform=transforms.ToTensor())
test_ds = dataset(DATASET_PATH, train=False, download=True, transform=transforms.ToTensor())


def train_image_classifier(model_name, dataset_name, train_dataset, val_dataset, test_dataset,
                           c_hidden, num_layers, dp_rate, log_with_wandb):
    pl.seed_everything(RAND_SEED)

    # Init DataLoader from Dataset
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1)
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=1)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=1)

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(CHECKPOINT_PATH, "ImageClassification" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_f1", save_top_k=3, save_last=True),
                   EarlyStopping(monitor='val_f1', min_delta=3e-5, patience=50)],
        gpus=AVAIL_GPUS,
        max_epochs=50,
        progress_bar_refresh_rate=0
    )

    # Set up logger for Lightning
    if log_with_wandb:
        trainer.logger = pl.loggers.WandbLogger(
            project='MLPMixers-Study',
            name="ImageClassification" + model_name
        )

    # Initialize new model
    pl.seed_everything(seed=RAND_SEED)
    if model_name.lower() == 'mlp':
        model = LitMLP(c_out=10, c_hidden=c_hidden, dp_rate=dp_rate, dataset_name=dataset_name)
    elif model_name.lower() == 'mlpmixer':
        model = LitMLPMixer(
            c_in=1, c_out=10, c_hidden=c_hidden, num_layers=num_layers, dp_rate=dp_rate, dataset_name=dataset_name
        )
    elif model_name.lower() == 'cnn':
        model = LitCNN(c_in=1, c_out=10, c_hidden=c_hidden, dp_rate=dp_rate, dataset_name=dataset_name)
    else:
        raise NotImplementedError(f'The model {model_name} is not currently implemented')

    # Train new model
    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)

    # Test best model on the test set
    model = model.__class__.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(model, test_dataloaders=test_data_loader)

    # Return best model
    return model


if __name__ == '__main__':
    image_nn_model = train_image_classifier(
        model_name=model_name, dataset_name=dataset_name, train_dataset=train_ds, val_dataset=val_ds,
        test_dataset=test_ds, c_hidden=256, num_layers=8, dp_rate=0.2, log_with_wandb=True
    )
