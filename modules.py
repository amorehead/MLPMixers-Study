# Functional tools
from functools import partial

# PyTorch Lightning
import pytorch_lightning as pl
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
# Einstein operations
from einops.layers.torch import Rearrange, Reduce
# PyTorch Lightning
from pytorch_lightning.utilities.types import EPOCH_OUTPUT


class PreNormResidual(nn.Module):
    """A pre-normalization layer for residual connections."""

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    """A standard MLP inside the MLP-Mixer architecture."""
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor=4, dropout=0.):
    """An MLP-Mixer model."""
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


class LitMLP(pl.LightningModule):
    """An MLP training module via PyTorch Lightning."""

    def __init__(self, c_out, c_hidden, dp_rate):
        super().__init__()
        # Define MLP layers
        self.layers = nn.ModuleList([
            # MLP Block 1
            nn.Linear(28 * 28, c_hidden),
            nn.ReLU(),
            nn.Dropout(p=dp_rate, inplace=True),

            # MLP Block 2
            nn.Linear(c_hidden, c_hidden * 2),
            nn.ReLU(),
            nn.Dropout(p=dp_rate, inplace=True),

            # MLP Block 3
            nn.Linear(c_hidden * 2, c_out)
        ])

        # Establish softmax, loss, and metric functions
        self.softmax_fn = torch.log_softmax
        self.loss_fn = F.nll_loss
        self.acc = tm.Accuracy()

        # Save hyperparameters within LightningModule
        self.save_hyperparameters()

    def shared_step(self, data):
        # Data retrieval
        x, labels = data[0], data[1]

        # (b, 1, 28, 28) -> (b, 1 * 28 * 28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # Forward pass for an image batch
        for layer in self.layers:
            x = layer(x)

        # Loss and accuracy calculation
        x = self.softmax_fn(x, dim=1)
        loss = self.loss_fn(x, labels)
        acc = self.acc(x, labels)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.acc.reset()

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.acc.reset()

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer


class LitMLPMixer(pl.LightningModule):
    """An MLP-Mixer training module via PyTorch Lightning."""

    def __init__(self, c_in, c_out, c_hidden, num_layers, dp_rate):
        super().__init__()
        # Define MLP-Mixer layers
        self.model = MLPMixer(
            image_size=28,
            channels=c_in,
            patch_size=4,
            dim=c_hidden,
            depth=num_layers,
            num_classes=c_out,
            dropout=dp_rate
        )

        # Establish softmax, loss, and metric functions
        self.softmax_fn = torch.log_softmax
        self.loss_fn = F.nll_loss
        self.acc = tm.Accuracy()

        # Save hyperparameters within LightningModule
        self.save_hyperparameters()

    def shared_step(self, data):
        # Data retrieval
        images, labels = data[0], data[1]
        x = self.model(images)

        # Loss and accuracy calculation
        x = self.softmax_fn(x, dim=1)
        loss = self.loss_fn(x, labels)
        acc = self.acc(x, labels)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.acc.reset()

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.acc.reset()

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer


class LitCNN(pl.LightningModule):
    """A CNN training module via PyTorch Lightning."""

    def __init__(self, c_in, c_out, c_hidden, dp_rate):
        super().__init__()
        # Define CNN layers
        self.layers = nn.ModuleList([
            # CNN Block 1
            nn.Conv2d(c_in, c_hidden, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.Dropout(p=dp_rate, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # CNN Block 2
            nn.Conv2d(c_hidden, c_hidden * 2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.Dropout(p=dp_rate, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        ])

        # Define classification MLP
        self.clsf_mlp = nn.Linear(c_hidden * 2 * (7 * 7), c_out)

        # Establish softmax, loss, and metric functions
        self.softmax_fn = torch.log_softmax
        self.loss_fn = F.nll_loss
        self.acc = tm.Accuracy()

        # Save hyperparameters within LightningModule
        self.save_hyperparameters()

    def shared_step(self, data):
        # Data retrieval
        x, labels = data[0], data[1]

        # Forward pass
        for layer in self.layers:
            x = layer(x)

        # Apply classification MLP
        x = x.view(x.size(0), -1)  # Flatten output shape of the last CNN layer to (batch_size, 32 * 7 * 7)
        x = self.clsf_mlp(x)

        # Loss and accuracy calculation
        x = self.softmax_fn(x, dim=1)
        loss = self.loss_fn(x, labels)
        acc = self.acc(x, labels)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.acc.reset()

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.acc.reset()

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer
