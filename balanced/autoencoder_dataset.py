import os
import pandas as pd
import numpy as np
# from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt


class AutoencoderDataset(Dataset):
    """Implement dataset for autoencoder 3d to 2d"""
    X: np.ndarray
    idx: int
    length: int

    # def __init__(self, X, y, transform=None, target_transform=None):
    def __init__(
            self,
            X: np.ndarray,
    ) -> None:
        """Create dataset from user X and y"""

        X_len = len(X)

        self.X = X
        self.length = X_len
        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32)


# https://colab.research.google.com/drive/1krRZ-VVfpXUsHk3JFCjBtYUiUOwL8vW0?usp=sharing#scrollTo=TBuoZ1PgK01f

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(
            self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor

        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print()
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print()
                print('INFO: Early stopping')
                self.early_stop = True
