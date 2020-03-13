from sklearn import metrics
import numpy as np
from math import sqrt



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