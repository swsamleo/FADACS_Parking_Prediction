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
    
    params = {
          'device': 'cpu'
          }

    gbm = lgb.LGBMRegressor(num_leaves=20 * data["parkingSlotsNum"],
                             learning_rate=0.01,
                             n_estimators=10 * data["parkingSlotsNum"],device_type = "cpu")

    if os.path.isfile(modelFile) and data["reTrain"] == False:
        print (data["col"] + " Training Model File exist, skip training!")
        return 
        #gbm = joblib.load(modelFile)
    else:
        open(modelFile,"a").close()
        gbm.fit(data["x"][eval_size:,:], np.array(data["y"][eval_size:, :]).squeeze(),
                eval_set=[(eval_x, np.array(eval_y).squeeze())],
                early_stopping_rounds=5, verbose=False)
#         gbm = lgb.train(params, train_set=dtrain, num_boost_round=10,
#                 valid_sets=None, valid_names=None,
#                 fobj=None, feval=None, init_model=None,
#                 feature_name='auto', categorical_feature='auto',
#                 early_stopping_rounds=None, evals_result=None,
#                 verbose_eval=True,
#                 keep_training_booster=False, callbacks=None)
        joblib.dump(gbm, modelFile)

    end = time.time()
    print("lightGBM Training Done [" + data["col"] + "], spent: %.2fs" % (end - start))
    #return gbm


# lightGBM model training method
def test(data):
    print(data["loghead"] + data["col"] + " start!")
    modelFile = data["filepath"] + 'lightgbm_' + data["col"] + '.pkl'
    start = time.time()
    
    data["x"] = data["x"].reshape(-1,data["x"].shape[1]*data["x"].shape[2])
    
    if os.path.isfile(modelFile):
        print(data["col"] + "loading model file:"+modelFile)
        
        gbm = joblib.load(modelFile)
        
        pred_y = gbm.predict(data["x"])

        metrics = ModelUtils.getMetrics(data["y"].squeeze(), pred_y)

        end = time.time()
        print(data["loghead"] + data["col"] + (' lightGBM Test MAE: %.2f' % metrics["mae"])+", spent: %.2fs" % (end - start))
        return {data["col"]:metrics}
    else:
        print(data["col"] + "No model file:"+modelFile)
        print("PLZ confirm model file first!!!")
        
    # return 0.0
