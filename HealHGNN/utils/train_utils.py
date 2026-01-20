import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os


def ActivateModule(act_str: str):
    if act_str == 'relu':
        return nn.ReLU()
    elif act_str == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif act_str == 'tanh':
        return nn.Tanh()
    elif act_str == 'elu':
        return nn.ELU()
    elif act_str == 'gelu':
        return nn.GELU()
    elif act_str == 'softplus':
        return nn.Softplus()
    elif act_str == 'sigmoid':
        return nn.Sigmoid()
    elif act_str is None:
        return nn.Identity()
    elif act_str == 'telu':
        return TeLU()
    else:
        raise NotImplementedError

class TeLU(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))

def NormModule(norm_str: str, dim: int):
    if norm_str == "ln":
        return nn.LayerNorm(dim)
    elif norm_str == "bn":
        return nn.BatchNorm1d(dim)
    else:
        return nn.Identity()


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, dir, file, save=True):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if save:
                self.save_checkpoint(val_loss, model, dir, file)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if save:
                self.save_checkpoint(val_loss, model, dir, file)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, dir, file):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists(dir):
            os.mkdir(dir)
        torch.save(model.state_dict(), os.path.join(dir, file))
        self.val_loss_min = val_loss