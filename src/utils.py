# coding: utf-8


import os
import shutil
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision


# sns.set(font_scale=2)
sns.set()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ######################### Data Creater ###########################


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min())


def calc_filter(img: np.ndarray, filter: np.ndarray) -> np.ndarray:
    return normalize(img.dot(filter)[:, :, ::-1])


# ######################### Data Argumentation ###########################


# ######################### Callback Layers ###########################


class ModelCheckPoint(object):

    def __init__(self, checkpoint_path: str, model_name: str, mkdir: bool=False,
                 partience: int=1, verbose: bool=True, *args, **kwargs) -> None:
        self.checkpoint_path = os.path.join(checkpoint_path, model_name)
        self.model_name = model_name
        self.partience = partience
        self.verbose = verbose
        self.all_loss = []
        self.all_val_loss = []
        if mkdir is True:
            if os.path.exists(self.checkpoint_path):
                shutil.rmtree(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.colab2drive_idx = 0
        if 'colab2drive' in kwargs.keys():
            self.colab2drive = kwargs['colab2drive']
            self.colab2drive_path = kwargs['colab2drive_path']
            self.colab2drive_flag = True
        else:
            self.colab2drive_flag = False

    def callback(self, model: str, epoch: int, *args, **kwargs) -> None:
        if 'loss' not in kwargs and 'val_loss' not in kwargs:
            assert 'None Loss'
        else:
            loss = kwargs['loss']
            val_loss = kwargs['val_loss']
        loss = np.mean(loss)
        val_loss = np.mean(val_loss)
        self.all_loss.append(loss)
        self.all_val_loss.append(val_loss)
        save_file = self.model_name + f'_epoch_{epoch:05d}_loss_{loss:.7f}_valloss_{val_loss:.7f}.tar'
        checkpoint_name = os.path.join(self.checkpoint_path, save_file)

        epoch += 1
        if epoch % self.partience == 0:
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'loss': self.all_loss, 'val_loss': self.all_val_loss,
                        'optim': kwargs['optim'].state_dict()}, checkpoint_name)
            if self.verbose is True:
                print(f'CheckPoint Saved by {checkpoint_name}')
        if self.colab2drive_flag is True and epoch == self.colab2drive[self.colab2drive_idx]:
            colab2drive_path = os.path.join(self.colab2drive_path, save_file)
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'loss': self.all_loss, 'val_loss': self.all_val_loss,
                        'optim': kwargs['optim'].state_dict()}, colab2drive_path)
            self.colab2drive_idx += 1
        return self

