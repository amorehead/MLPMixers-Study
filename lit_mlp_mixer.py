# Standard libraries
import os

# PyTorch Lightning
import pytorch_lightning as pl
# PyTorch
import torch
# PL callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# Declare system hyperparameters
from modules import LitMLPMixer

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 1 if AVAIL_GPUS else 1
DATASET_PATH = '~/data/'  # Path to the folder where the datasets are/should be downloaded
CHECKPOINT_BASE_PATH = '~/savedmodels/'  # Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.path.join(CHECKPOINT_BASE_PATH, "NNs/")
RAND_SEED = 42

# Set the seed
pl.seed_everything(RAND_SEED)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

# Create checkpoint path if it doesn't exist yet
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_BASE_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Init MNIST dataset
train_ds = MNIST(DATASET_PATH, train=True, download=True, transform=transforms.ToTensor())
val_ds = MNIST(DATASET_PATH, train=False, download=True, transform=transforms.ToTensor())
test_ds = MNIST(DATASET_PATH, train=False, download=True, transform=transforms.ToTensor())


def train_image_classifier(model_name, train_dataset, val_dataset, test_dataset, c_hidden, num_layers, dp_rate):
    pl.seed_everything(42)

    # Init DataLoader from MNIST Dataset
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1)
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=1)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=1)

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(CHECKPOINT_PATH, "ImageClassification" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
        gpus=AVAIL_GPUS,
        max_epochs=200,
        progress_bar_refresh_rate=0,
    )  # 0 because epoch size is 1

    # Set up logger for Lightning
    trainer.logger = pl.loggers.WandbLogger(
        project='MLPMixers-Study'
    )

    # Initialize new model
    pl.seed_everything()
    model = LitMLPMixer(
        c_in=1, c_out=10, c_hidden=c_hidden, num_layers=num_layers, dp_rate=dp_rate
    )
    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)
    model = LitMLPMixer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, test_dataloaders=test_data_loader, verbose=False)
    batch = next(iter(test_data_loader))
    batch[0] = batch[0].to(model.device)
    batch[1] = batch[1].to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
    return model, result


# Small function for printing the test scores
def print_results(result_dict):
    if "train" in result_dict:
        print("Train accuracy: %4.2f%%" % (100.0 * result_dict["train"]))
    if "val" in result_dict:
        print("Val accuracy:   %4.2f%%" % (100.0 * result_dict["val"]))
    print("Test accuracy:  %4.2f%%" % (100.0 * result_dict["test"]))


image_nn_model, image_nn_result = train_image_classifier(
    model_name='MLPMixer', train_dataset=train_ds, val_dataset=val_ds,
    test_dataset=test_ds, c_hidden=16, num_layers=2, dp_rate=0.1
)
print_results(image_nn_result)
