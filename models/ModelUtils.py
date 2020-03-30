from sklearn import metrics
import numpy as np
from math import sqrt
import gzip
import os
import pickle
import urllib
import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets
from . import ModelUtils

class CityDataset(data.Dataset):
    """City Dataset.

    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
    """

    def __init__(self, filepath, tIndex = 0, train=True):
        self.root = os.path.expanduser(filepath)
        self.train = train
        self.dataset_size = None
        self.tIndex = tIndex
        self.x, self.y = self.loadDatasets()

    def setYIndex(self,index):
        self.tIndex = index
        
    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index
        Returns:
            tuple: (x, y) where target is index of the target class.
        """
        return self.x[index, ::], self.y[index,0,self.tIndex]

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def loadDatasets(self):
        """Load from np files."""
        if self.train:
            print("loading "+ os.path.join(self.root, "x.npy"))
            x = np.load(os.path.join(self.root, "x.npy"),allow_pickle=True)
            y = np.load(os.path.join(self.root, "y.npy"),allow_pickle=True)
            x = x.astype(np.float32)
            y = y.astype(np.float32)
            self.dataset_size = x.shape[0]
        else:
            print("loading "+ os.path.join(self.root, "tx.npy"))
            x = np.load(os.path.join(self.root, "tx.npy"),allow_pickle=True)
            y = np.load(os.path.join(self.root, "ty.npy"),allow_pickle=True)
            x = x.astype(np.float32)            
            y = y.astype(np.float32)
            self.dataset_size = x.shape[0]
        return x, y


def loadDatasets(path,src,tar):
    return {"srcTrain":CityDataset(path+src,train = True),
            "srcTest":CityDataset(path+src,train = False),
            "tarTrain":CityDataset(path+tar,train = True),
            "tarTest":CityDataset(path+tar,train = False)}
            
def getDataSrcTarLoaders(datasets,batch_size,tIndex):
    return {
        "srcTrain": torch.utils.data.DataLoader(
            dataset=datasets["srcTrain"],
            batch_size=batch_size,
            shuffle=True),
        "srcTest": torch.utils.data.DataLoader(
            dataset=datasets["srcTest"],
            batch_size=batch_size,
            shuffle=True),
        "tarTrain": torch.utils.data.DataLoader(
            dataset=datasets["tarTrain"],
            batch_size=batch_size,
            shuffle=True),
        "tarTest": torch.utils.data.DataLoader(
            dataset=datasets["tarTest"],
            batch_size=batch_size,
            shuffle=True)
    }

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def root_mean_square_error(y_true, y_pred): 
    return sqrt(metrics.mean_squared_error(y_true, y_pred))


def getMetrics(y_true, y_pred):
    #acc = metrics.accuracy_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    #mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = root_mean_square_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    return {"mae":mae,"rmse":rmse,"r2":r2}