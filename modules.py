# Functional tools
from functools import partial
# PyTorch Lightning
from typing import Optional

import pytorch_lightning as pl
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from einops.layers.torch import Rearrange, Reduce

# Einstein operations
import wandb
from constants import MNIST_CLASS_NAMES, FASHION_MNIST_CLASS_NAMES


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

    def __init__(self, c_out, c_hidden, num_layers, dp_rate, lr, dataset_name):
        super().__init__()
        # MLP initializer
        self.init_module = nn.ModuleList([
            nn.Linear(28 * 28, c_hidden),
            nn.ReLU(),
            nn.Dropout(p=dp_rate, inplace=True),
        ])
        # MLP Blocks 1 through N
        blocks = []
        for _ in range(num_layers):
            blocks.extend(self.get_block(c_hidden, dp_rate))
        self.layers = nn.ModuleList(blocks)
        # MLP finalizer
        self.final_linear = nn.Linear(c_hidden, c_out)

        # Declare shared function(s)
        self.loss_fn = F.nll_loss
        self.softmax_fn = torch.log_softmax

        # Define step-specific metrics
        # Train #
        self.train_acc = tm.Accuracy(num_classes=10, average='macro')
        # Val #
        self.val_prec = tm.Precision(num_classes=10, average='macro')
        self.val_recall = tm.Recall(num_classes=10, average='macro')
        self.val_f1 = tm.F1(num_classes=10, average='macro')
        self.val_auroc = tm.AUROC(num_classes=10, average='macro')
        # Test #
        self.test_prec = tm.Precision(num_classes=10, average='macro')
        self.test_recall = tm.Recall(num_classes=10, average='macro')
        self.test_f1 = tm.F1(num_classes=10, average='macro')
        self.test_auroc = tm.AUROC(num_classes=10, average='macro')

        # Capture name of dataset being used
        self.dataset_name = dataset_name.strip().lower()

        # Save hyperparameters within LightningModule
        self.lr = lr
        self.save_hyperparameters()

    @staticmethod
    def get_block(c_hidden: int, dp_rate: float):
        return [
            # MLP Block
            nn.Linear(c_hidden, c_hidden * 2),
            nn.ReLU(),
            nn.Dropout(p=dp_rate, inplace=True),

            nn.Linear(c_hidden * 2, c_hidden),
            nn.ReLU(),
            nn.Dropout(p=dp_rate, inplace=True),
        ]

    def shared_step(self, data):
        # Data retrieval
        x, labels = data[0], data[1]

        # (b, 1, 28, 28) -> (b, 1 * 28 * 28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # Forward pass for an image batch
        for layer in self.init_module:
            x = layer(x)
        res_x = x
        for layer_idx, layer in enumerate(self.layers):
            if (layer_idx > 0 and 5 + layer_idx % 6) == 5:
                x += res_x
            x = layer(x)
        x += res_x
        x = self.final_linear(x)

        # Loss and accuracy calculation
        loss = self.loss_fn(self.softmax_fn(x, dim=1), labels)

        return torch.softmax(x, dim=1), labels, loss

    def training_step(self, *args, **kwargs) -> pl.utilities.types.STEP_OUTPUT:
        x, labels, loss = self.shared_step(args[0])
        class_preds = torch.argmax(x, dim=1)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc(class_preds, labels), on_step=False, on_epoch=True)

        return {
            'loss': loss
        }

    def training_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        # Reset training metrics
        self.train_acc.reset()

    def validation_step(self, *args, **kwargs) -> Optional[pl.utilities.types.STEP_OUTPUT]:
        x, labels, loss = self.shared_step(args[0])
        class_preds = torch.argmax(x, dim=1)
        self.log('val_loss', loss)
        self.log("val_prec", self.val_prec(class_preds, labels))
        self.log("val_recall", self.val_recall(class_preds, labels))
        self.log("val_f1", self.val_f1(class_preds, labels))
        self.log("val_auroc", self.val_auroc(x, labels))

        return {
            'loss': loss,
            'x': x,
            'labels': labels
        }

    def validation_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        # Reset validation metrics
        self.val_prec.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def test_step(self, *args, **kwargs) -> Optional[pl.utilities.types.STEP_OUTPUT]:
        x, labels, loss = self.shared_step(args[0])
        class_preds = torch.argmax(x, dim=1)
        self.log('test_loss', loss)
        self.log("test_prec", self.test_prec(class_preds, labels))
        self.log("test_recall", self.test_recall(class_preds, labels))
        self.log("test_f1", self.test_f1(class_preds, labels))
        self.log("test_auroc", self.test_auroc(x, labels))

        return {
            'loss': loss,
            'x': x,
            'labels': labels
        }

    def test_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        # Reset test metrics
        self.test_prec.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        # Plot total test confusion matrix
        xs = torch.cat([output['x'] for output in outputs])
        labels = torch.cat([output['labels'] for output in outputs])
        class_names = FASHION_MNIST_CLASS_NAMES if 'fashion' in self.dataset_name else MNIST_CLASS_NAMES
        title = 'Fashion MNIST Confusion Matrix' if 'fashion' in self.dataset_name else 'MNIST Confusion Matrix'
        total_test_conf_mat = wandb.plot.confusion_matrix(
            y_true=labels.cpu().numpy(),
            probs=xs.cpu().numpy(),
            class_names=class_names,
            title=title
        )
        self.trainer.logger.experiment.log({'test_conf_mat': total_test_conf_mat})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer


class LitMLPMixer(pl.LightningModule):
    """An MLP-Mixer training module via PyTorch Lightning."""

    def __init__(self, c_in, c_out, c_hidden, num_layers, dp_rate, lr, dataset_name):
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

        # Declare shared function(s)
        self.loss_fn = F.nll_loss
        self.softmax_fn = torch.log_softmax

        # Define step-specific metrics
        # Train #
        self.train_acc = tm.Accuracy(num_classes=10, average='macro')
        # Val #
        self.val_prec = tm.Precision(num_classes=10, average='macro')
        self.val_recall = tm.Recall(num_classes=10, average='macro')
        self.val_f1 = tm.F1(num_classes=10, average='macro')
        self.val_auroc = tm.AUROC(num_classes=10, average='macro')
        # Test #
        self.test_prec = tm.Precision(num_classes=10, average='macro')
        self.test_recall = tm.Recall(num_classes=10, average='macro')
        self.test_f1 = tm.F1(num_classes=10, average='macro')
        self.test_auroc = tm.AUROC(num_classes=10, average='macro')

        # Capture name of dataset being used
        self.dataset_name = dataset_name.strip().lower()

        # Save hyperparameters within LightningModule
        self.lr = lr
        self.save_hyperparameters()

    def shared_step(self, data):
        # Data retrieval
        images, labels = data[0], data[1]

        # Forward pass for an image batch
        x = self.model(images)

        # Loss calculation
        loss = self.loss_fn(self.softmax_fn(x, dim=1), labels)

        return torch.softmax(x, dim=1), labels, loss

    def training_step(self, *args, **kwargs) -> pl.utilities.types.STEP_OUTPUT:
        x, labels, loss = self.shared_step(args[0])
        class_preds = torch.argmax(x, dim=1)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc(class_preds, labels), on_step=False, on_epoch=True)

        return {
            'loss': loss
        }

    def training_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        # Reset training metrics
        self.train_acc.reset()

    def validation_step(self, *args, **kwargs) -> Optional[pl.utilities.types.STEP_OUTPUT]:
        x, labels, loss = self.shared_step(args[0])
        class_preds = torch.argmax(x, dim=1)
        self.log('val_loss', loss)
        self.log("val_prec", self.val_prec(class_preds, labels))
        self.log("val_recall", self.val_recall(class_preds, labels))
        self.log("val_f1", self.val_f1(class_preds, labels))
        self.log("val_auroc", self.val_auroc(x, labels))

        return {
            'loss': loss,
            'x': x,
            'labels': labels
        }

    def validation_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        # Reset validation metrics
        self.val_prec.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def test_step(self, *args, **kwargs) -> Optional[pl.utilities.types.STEP_OUTPUT]:
        x, labels, loss = self.shared_step(args[0])
        class_preds = torch.argmax(x, dim=1)
        self.log('test_loss', loss)
        self.log("test_prec", self.test_prec(class_preds, labels))
        self.log("test_recall", self.test_recall(class_preds, labels))
        self.log("test_f1", self.test_f1(class_preds, labels))
        self.log("test_auroc", self.test_auroc(x, labels))

        return {
            'loss': loss,
            'x': x,
            'labels': labels
        }

    def test_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        # Reset test metrics
        self.test_prec.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        # Plot total test confusion matrix
        xs = torch.cat([output['x'] for output in outputs])
        labels = torch.cat([output['labels'] for output in outputs])
        class_names = FASHION_MNIST_CLASS_NAMES if 'fashion' in self.dataset_name else MNIST_CLASS_NAMES
        title = 'Fashion MNIST Confusion Matrix' if 'fashion' in self.dataset_name else 'MNIST Confusion Matrix'
        total_test_conf_mat = wandb.plot.confusion_matrix(
            y_true=labels.cpu().numpy(),
            probs=xs.cpu().numpy(),
            class_names=class_names,
            title=title
        )
        self.trainer.logger.experiment.log({'test_conf_mat': total_test_conf_mat})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer


class LitCNN(pl.LightningModule):
    """A CNN training module via PyTorch Lightning."""

    def __init__(self, c_in, c_out, c_hidden, num_layers, dp_rate, lr, dataset_name):
        super().__init__()
        # CNN initializer
        self.init_module = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p=dp_rate, inplace=True),
        )
        # CNN Blocks 1 through N
        blocks = []
        for _ in range(num_layers):
            blocks.extend(self.get_block(c_hidden, dp_rate))
        self.layers = nn.ModuleList(blocks)
        # CNN finalizer
        self.final_module = nn.Sequential(
            # Final Block 1
            nn.Conv2d(c_hidden, c_hidden // 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p=dp_rate, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # Final Block 2
            nn.Conv2d(c_hidden // 4, c_hidden, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p=dp_rate, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.final_linear = nn.Linear(c_hidden * (7 * 7), c_out)

        # Declare shared function(s)
        self.loss_fn = F.nll_loss
        self.softmax_fn = torch.log_softmax

        # Define step-specific metrics
        # Train #
        self.train_acc = tm.Accuracy(num_classes=10, average='macro')
        # Val #
        self.val_prec = tm.Precision(num_classes=10, average='macro')
        self.val_recall = tm.Recall(num_classes=10, average='macro')
        self.val_f1 = tm.F1(num_classes=10, average='macro')
        self.val_auroc = tm.AUROC(num_classes=10, average='macro')
        # Test #
        self.test_prec = tm.Precision(num_classes=10, average='macro')
        self.test_recall = tm.Recall(num_classes=10, average='macro')
        self.test_f1 = tm.F1(num_classes=10, average='macro')
        self.test_auroc = tm.AUROC(num_classes=10, average='macro')

        # Capture name of dataset being used
        self.dataset_name = dataset_name.strip().lower()

        # Save hyperparameters within LightningModule
        self.lr = lr
        self.save_hyperparameters()

    @staticmethod
    def get_block(c_hidden: int, dp_rate: float):
        return [
            # Middle Block 1
            nn.Conv2d(c_hidden, c_hidden, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p=dp_rate, inplace=True),
        ]

    def shared_step(self, data):
        # Retrieve data
        x, labels = data[0], data[1]

        # Forward pass for an image batch
        x = self.init_module(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_module(x)
        x = x.squeeze()  # Wring out extraneous dimensions
        x = x.view(x.size(0), -1)  # Flatten output shape of the last CNN layer to (batch_size, 32 * 7 * 7)
        x = self.final_linear(x)

        # Make loss and confusion matrix calculations
        loss = self.loss_fn(self.softmax_fn(x, dim=1), labels)

        return torch.softmax(x, dim=1), labels, loss

    def training_step(self, *args, **kwargs) -> pl.utilities.types.STEP_OUTPUT:
        x, labels, loss = self.shared_step(args[0])
        class_preds = torch.argmax(x, dim=1)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc(class_preds, labels), on_step=False, on_epoch=True)

        return {
            'loss': loss
        }

    def training_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        # Reset training metrics
        self.train_acc.reset()

    def validation_step(self, *args, **kwargs) -> Optional[pl.utilities.types.STEP_OUTPUT]:
        x, labels, loss = self.shared_step(args[0])
        class_preds = torch.argmax(x, dim=1)
        self.log('val_loss', loss)
        self.log("val_prec", self.val_prec(class_preds, labels))
        self.log("val_recall", self.val_recall(class_preds, labels))
        self.log("val_f1", self.val_f1(class_preds, labels))
        self.log("val_auroc", self.val_auroc(x, labels))

        return {
            'loss': loss,
            'x': x,
            'labels': labels
        }

    def validation_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        # Reset validation metrics
        self.val_prec.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def test_step(self, *args, **kwargs) -> Optional[pl.utilities.types.STEP_OUTPUT]:
        x, labels, loss = self.shared_step(args[0])
        class_preds = torch.argmax(x, dim=1)
        self.log('test_loss', loss)
        self.log("test_prec", self.test_prec(class_preds, labels))
        self.log("test_recall", self.test_recall(class_preds, labels))
        self.log("test_f1", self.test_f1(class_preds, labels))
        self.log("test_auroc", self.test_auroc(x, labels))

        return {
            'loss': loss,
            'x': x,
            'labels': labels
        }

    def test_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        # Reset test metrics
        self.test_prec.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        # Plot total test confusion matrix
        xs = torch.cat([output['x'] for output in outputs])
        labels = torch.cat([output['labels'] for output in outputs])
        class_names = FASHION_MNIST_CLASS_NAMES if 'fashion' in self.dataset_name else MNIST_CLASS_NAMES
        title = 'Fashion MNIST Confusion Matrix' if 'fashion' in self.dataset_name else 'MNIST Confusion Matrix'
        total_test_conf_mat = wandb.plot.confusion_matrix(
            y_true=labels.cpu().numpy(),
            probs=xs.cpu().numpy(),
            class_names=class_names,
            title=title
        )
        self.trainer.logger.experiment.log({'test_conf_mat': total_test_conf_mat})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer
