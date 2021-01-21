# @Time     : Jul. 10, 2020 19:45
# @Author   : Zhen Zhang
# @Email    : david.zhen.zhang@gmail.com
# @FileName : ModelUtils.py
# @Version  : 1.0
# @IDE      : VSCode

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

    def __init__(self, filepath, tIndex = 0, train=True ,trainWithParkingData = True):
        self.root = os.path.expanduser(filepath)
        self.train = train
        self.dataset_size = None
        self.tIndex = tIndex
        self.baseUnit = 6
        self.featureNum = 0
        self.trainWithParkingData = trainWithParkingData
        self.x, self.y = self.loadDatasets()
        print("CityDataset dataset_size {}".format(self.dataset_size))

    def getBaseUnit(self):
        return self.baseUnit

    def getFeatureNum(self):
        return self.featureNum

    def getY(self):
        return self.y

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
            
            if self.trainWithParkingData == False:
                x = x[:,:,1:x.shape[2]]

            if os.path.exists(os.path.join(self.root, "trainIndex.npy")) == True:
                trIndex = np.load(os.path.join(self.root, "trainIndex.npy"),allow_pickle=True)
                x = x[trIndex]
                y = y[trIndex]
            
            x = x.astype(np.float32)
            y = y.astype(np.float32)
            self.dataset_size = x.shape[0]
            self.baseUnit = x.shape[1]
            self.featureNum = x.shape[2]

        else:
            print("loading "+ os.path.join(self.root, "tx.npy"))
            x = None
            y = None
            
            if os.path.exists(os.path.join(self.root, "testIndex.npy")) == True:
                x = np.load(os.path.join(self.root, "x.npy"),allow_pickle=True)
                y = np.load(os.path.join(self.root, "y.npy"),allow_pickle=True)

                if self.trainWithParkingData == False:
                    x = x[:,:,1:x.shape[2]]

                teIndex = np.load(os.path.join(self.root, "testIndex.npy"),allow_pickle=True)
                x = x[teIndex]
                y = y[teIndex]
            else:
                x = np.load(os.path.join(self.root, "tx.npy"),allow_pickle=True)
                y = np.load(os.path.join(self.root, "ty.npy"),allow_pickle=True)
            
            x = x.astype(np.float32)            
            y = y.astype(np.float32)
            self.dataset_size = x.shape[0]
        return x, y


def loadDatasets(path,src,tar,tIndex,trainWithParkingData = True):
    return {"srcTrain":CityDataset(path+src,train = True,tIndex=tIndex,trainWithParkingData = trainWithParkingData),
            "srcTest":CityDataset(path+src,train = False,tIndex=tIndex,trainWithParkingData = trainWithParkingData),
            "tarTrain":CityDataset(path+tar,train = True,tIndex=tIndex,trainWithParkingData = trainWithParkingData),
            "tarTest":CityDataset(path+tar,train = False,tIndex=tIndex,trainWithParkingData = trainWithParkingData)}
            
def getDataSrcTarLoaders(datasets,batch_size):
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
    if np.array(np.abs(y_true) - np.abs(y_pred)).all() == 0:
        return 0
    return np.mean(np.abs((y_true - y_pred) / y_true+(1e-05))) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred): 
    print(y_true)
    print(y_pred)

    if np.array(np.abs(y_true) - np.abs(y_pred)).all() == 0:
        return 0
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def root_mean_square_error(y_true, y_pred): 
    return sqrt(metrics.mean_squared_error(y_true, y_pred))

def mean_absolut_scaled_error(y_train, y_true, y_pred):
    
    n = y_train.shape[0]
    d = np.abs(  np.diff( y_train) ).sum()/(n-1)
    #print("y_train:"+str(y_train.shape))
    errors = np.abs(y_true - y_pred)
    return errors.mean()/d

def getMetrics(y_true, y_pred, y_train):
    #acc = metrics.accuracy_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    #mape = mean_absolute_percentage_error(y_true, y_pred)
    #smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    smape = 0
    rmse = root_mean_square_error(y_true, y_pred)

    r2 = 0
    if y_true.shape[0] > 2:
        r2 = metrics.r2_score(y_true, y_pred)

    if y_train is not None:
        mase = mean_absolut_scaled_error(y_train,y_true, y_pred)
        return {"mae":mae,"rmse":rmse,"r2":r2,"mase":mase,"smape":smape}
    else:
        return {"mae":mae,"rmse":rmse,"r2":r2,"smape":smape}