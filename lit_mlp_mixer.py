# Standard libraries
import os
# Functional tools
from functools import partial

# PyTorch Lightning
import pytorch_lightning as pl
# PyTorch
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
# PL callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = '~/data/'
# Path to the folder where the pretrained models are saved
CHECKPOINT_BASE_PATH = '~/savedmodels/'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_BASE_PATH, "NNs/")

# Setting the seed
pl.seed_everything(42)

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


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor=4, dropout=0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )


class LitMLPMixer(pl.LightningModule):
    def __init__(self, c_in, c_out, c_hidden, num_layers, dp_rate):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.model = MLPMixer(
            image_size=700,
            channels=c_in,
            patch_size=4,
            dim=c_hidden,
            depth=num_layers,
            num_classes=c_out
        )
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data):
        images, labels = data[0], data[1]
        x = self.model(images)
        loss = self.loss_module(x, labels)
        acc = (x.argmax(dim=-1) == labels).sum().float() / len(labels)
        return loss, acc

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch)
        self.log("test_acc", acc)


def train_image_classifier(model_name, train_dataset, val_dataset, test_dataset, c_hidden, num_layers, dp_rate):
    pl.seed_everything(42)

    # Init DataLoader from MNIST Dataset
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

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
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

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
    batch = batch.to(model.device)
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
