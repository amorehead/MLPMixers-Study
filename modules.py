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
from pytorch_lightning.utilities.types import EPOCH_OUTPUT


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
            image_size=28,
            channels=c_in,
            patch_size=4,
            dim=c_hidden,
            depth=num_layers,
            num_classes=c_out
        )
        self.loss_module = F.nll_loss
        self.acc = tm.Accuracy()

    def forward(self, data):
        images, labels = data[0], data[1]
        x = self.model(images)
        x = torch.log_softmax(x, dim=1)
        loss = self.loss_module(x, labels)
        acc = self.acc(x, labels)
        return loss, acc

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.acc.reset()

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch)
        self.log("val_acc", acc)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.acc.reset()

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch)
        self.log("test_acc", acc)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.acc.reset()
