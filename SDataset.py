import torch.utils.data as Data
import random
import os
import numpy as np
from io import StringIO
import pandas as pd


class MyDataset(Data.Dataset):
    def __init__(self, path):
        self.path = path
        self.file_length = {}
        # Load file in as a nmap
        d = np.load(os.path.join(self.path), mmap_mode='r')
        self.file_length[f] = len(d['y'])

    def __len__(self):
        return None
        # raise NotImplementedException()

    def __getitem__(self, idx):
        # Find the file where idx belongs to
        count = 0
        f_key = ''
        local_idx = 0
        for k in self.file_length:
            if count < idx < count + self.file_length[k]:
                f_key = k
                local_idx = idx - count
                break
            else:
                count += self.file_length[k]
        # Open file as numpy.memmap
        d = np.load(os.path.join(self.path, f_key), mmap_mode='r')
        # Actually fetch the data
        X = np.expand_dims(d['X'][local_idx], axis=1)
        y = np.expand_dims((d['y'][local_idx] == 2).astype(np.float32), axis=1)
        return X, y