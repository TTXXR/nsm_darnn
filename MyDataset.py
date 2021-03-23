import torch.utils.data as Data
import random
import os
import numpy as np
from io import StringIO
import pandas as pd

"""
class MyDataset(Data.Dataset):
    def __init__(self, file_path, label_path, nraws, shuffle=False):

        file_raws = 0
        # get the count of all samples
        with open(file_path, 'r') as f:
            for _ in f:
                file_raws += 1
        self.file_path = file_path
        self.label_path = label_path
        self.file_raws = file_raws-1
        self.nraws = nraws
        self.shuffle = shuffle

    def initial(self):
        self.finput = open(self.file_path, 'r')
        self.finlabel = open(self.label_path, 'r')
        self.samples = list()

        # put nraw samples into memory
        for _ in range(self.nraws):
            data = self.finput.readline()  # data contains the feature and label
            data = data.strip('\n')
            label_data = self.finlabel.readline()
            label_data = label_data.strip('\n')
            # all_data = data+","+label_data
            # item = pd.read_csv(StringIO(all_data), sep=",")
            if data:
                self.samples.append(data+label_data)
            else:
                break
        self.current_sample_num = len(self.samples)
        self.index = list(range(self.current_sample_num))
        if self.shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return self.file_raws

    def __getitem__(self, item):
        idx = self.index[0]
        data = self.samples[idx]
        self.index = self.index[1:]
        self.current_sample_num -= 1

        if self.current_sample_num <= 0:
            # all the samples in the memory have been used, need to get the new samples
            for _ in range(self.nraws):
                data = self.finput.readline()  # data contains the feature and label
                label_data = self.finlabel.readline()
                if data:
                    self.samples.append([data,label_data])
                else:
                    break
            self.current_sample_num = len(self.samples)
            self.index = list(range(self.current_sample_num))
            if self.shuffle:
                random.shuffle(self.samples)

        return data
"""

class MyDataset(Data.Dataset):
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(self.path)
        self.file_length = {}
        for f in self.files:
            # Load file in as a nmap
            d = np.load(os.path.join(self.path, f), mmap_mode='r')
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