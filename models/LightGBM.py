import os.path

import pandas as pd
import numpy as np
import math
import time

import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
import joblib

from . import ModelUtils


# lightGBM model training method
def train(data, reTrain = False):
    print("lightGBM Training [" + data["col"] + "] start!")

    modelFile = data["filepath"] + 'lightgbm_' + data["col"] + '.pkl'

    start = time.time()

    eval_size = int(data["x"].shape[0] * 0.1)
    # print("eval_size:"+str(eval_size))
    # print(data["x"].shape)
    data["x"] = data["x"].reshape(-1,data["x"].shape[1]*data["x"].shape[2])
    
    eval_x = data["x"][:eval_size, :]
    eval_y = data["y"][:eval_size, :]
    # print(eval_x.shape)
    # print(data["x"].shape)

    # print(eval_y.values.shape)
    # print(data["y"].values.shape)

    # yy = np.array(eval_y).squeeze()
    # print(yy)

    gbm = lgb.LGBMRegressor(num_leaves=20 * data["parkingSlotsNum"],
                             learning_rate=0.05,
                             n_estimators=10 * data["parkingSlotsNum"])

    if os.path.isfile(modelFile) and data["reTrain"] == False:
        print (data["col"] + " Training Model File exist, skip training, load it")
        gbm = joblib.load(modelFile)
    else:
        gbm.fit(data["x"][eval_size:,:], np.array(data["y"][eval_size:, :]).squeeze(),
                eval_set=[(eval_x, np.array(eval_y).squeeze())],
                early_stopping_rounds=5, verbose=False)
        joblib.dump(gbm, modelFile)

    end = time.time()
    print("lightGBM Training Done [" + data["col"] + "], spent: %.2fs" % (end - start))
    return gbm


# lightGBM model training method
def test(data):
    print(data["loghead"] + data["col"] + " start!")
    data["x"] = data["x"].reshape(-1,data["x"].shape[1]*data["x"].shape[2])
    
    pred_y = data["clf"].predict(data["x"])

    metrics = ModelUtils.getMetrics(data["y"].squeeze(), pred_y)

    print(data["loghead"] + data["col"] + (' lightGBM Test MAE: %.2f' % metrics["mae"]))
    
    return {data["col"]:metrics}
    # return 0.0