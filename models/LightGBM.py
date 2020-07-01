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
def train(data, reTrain=False):
    print("lightGBM Training [" + data["col"] + "] start!")

    modelFile = data["filepath"] + 'lightgbm_' + data["col"] + '.pkl'

    start = time.time()

    eval_size = int(data["x"].shape[0] * 0.1)
    # print("eval_size:"+str(eval_size))
    # print(data["x"].shape)
    data["x"] = data["x"].reshape(-1, data["x"].shape[1]*data["x"].shape[2])

    eval_x = data["x"][:eval_size, :]
    eval_y = data["y"][:eval_size, :]

    boosting_type           = 'gbdt'
    num_leaves              = 31
    max_depth               = -1
    learning_rate           = 0.1
    n_estimators            = 100
    early_stopping_rounds   = 150
    subsample_for_bin       = 200000
    objective               = None
    class_weight            = None
    min_split_gain          = 0.0
    min_child_weight        = 0.001
    min_child_samples       = 20
    subsample               = 1.0
    subsample_freq          = 0
    colsample_bytree        = 1.0
    reg_alpha               = 0.0
    reg_lambda              = 0.0
    random_state            = None
    n_jobs                  = -1
    silent                  = True
    importance_type         = 'split'
    verbose                 = 1

    if data["parameters"] is not None:
        if "learning_rate" in data["parameters"]:
            learning_rate = data["parameters"]["learning_rate"]
        if "num_leaves" in data["parameters"]:
            num_leaves = data["parameters"]["num_leaves"]
        if "verbose" in data["parameters"]:
            verbose = data["parameters"]["verbose"]
        if "n_estimators" in data["parameters"]:
            n_estimators = data["parameters"]["n_estimators"]
        if "early_stopping_rounds" in data["parameters"]:
            early_stopping_rounds = data["parameters"]["early_stopping_rounds"]

        if "boosting_type" in data["parameters"]: boosting_type= data["parameters"]["boosting_type"]
        if "max_depth" in data["parameters"]:max_depth= data["parameters"]["max_depth"]
        if "subsample_for_bin" in data["parameters"]:subsample_for_bin= data["parameters"]["subsample_for_bin"]
        if "objective" in data["parameters"]:objective= data["parameters"]["objective"]
        if "class_weight" in data["parameters"]:class_weight= data["parameters"]["class_weight"]
        if "min_split_gain" in data["parameters"]:min_split_gain= data["parameters"]["min_split_gain"]
        if "min_child_weight" in data["parameters"]:min_child_weight= data["parameters"]["min_child_weight"]
        if "min_child_samples" in data["parameters"]:min_child_samples= data["parameters"]["min_child_samples"]
        if "subsample" in data["parameters"]:subsample= data["parameters"]["subsample"]
        if "subsample_freq" in data["parameters"]:subsample_freq= data["parameters"]["subsample_freq"]
        if "reg_alpha" in data["parameters"]:reg_alpha= data["parameters"]["reg_alpha"]
        if "reg_lambda" in data["parameters"]:reg_lambda= data["parameters"]["reg_lambda"]
        if "random_state" in data["parameters"]:random_state= data["parameters"]["random_state"]
        if "n_jobs" in data["parameters"]:n_jobs= data["parameters"]["n_jobs"]
        if "silent" in data["parameters"]:silent= data["parameters"]["silent"]
        if "importance_type" in data["parameters"]:importance_type= data["parameters"]["importance_type"]

    gbm = lgb.LGBMRegressor(boosting_type           = boosting_type,
                            num_leaves              = num_leaves,
                            max_depth               = max_depth,
                            learning_rate           = learning_rate,
                            n_estimators            = n_estimators,
                            early_stopping_rounds   = early_stopping_rounds,
                            subsample_for_bin       = subsample_for_bin,
                            objective               = objective,
                            class_weight            = class_weight,
                            min_split_gain          = min_split_gain,
                            min_child_weight        = min_child_weight,
                            min_child_samples       = min_child_samples,
                            subsample               = subsample,
                            subsample_freq          = subsample_freq,
                            colsample_bytree        = colsample_bytree,
                            reg_alpha               = reg_alpha,
                            reg_lambda              = reg_lambda,
                            random_state            = random_state,
                            n_jobs                  = n_jobs,
                            silent                  = silent,
                            importance_type         = importance_type,
                            verbose                 = verbose)

    if os.path.isfile(modelFile) and data["reTrain"] == False:
        print(data["col"] + " Training Model File exist, skip training!")
        return
        #gbm = joblib.load(modelFile)
    else:
        # open(modelFile,"a").close()
        gbm.fit(data["x"][eval_size:, :], np.array(data["y"][eval_size:, :]).squeeze(),
                eval_set=[(eval_x, np.array(eval_y).squeeze())],
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose)
#         gbm = lgb.train(params, train_set=dtrain, num_boost_round=10,
#                 valid_sets=None, valid_names=None,
#                 fobj=None, feval=None, init_model=None,
#                 feature_name='auto', categorical_feature='auto',
#                 early_stopping_rounds=None, evals_result=None,
#                 verbose_eval=True,
#                 keep_training_booster=False, callbacks=None)
        joblib.dump(gbm, modelFile)

    end = time.time()
    print(
        "lightGBM Training Done [" + data["col"] + "], spent: %.2fs" % (end - start))
    # return gbm


# lightGBM model training method
def test(data):
    print(data["loghead"] + data["col"] + " start!")
    modelFile = data["filepath"] + 'lightgbm_' + data["col"] + '.pkl'
    start = time.time()

    data["x"] = data["x"].reshape(-1, data["x"].shape[1]*data["x"].shape[2])

    if os.path.isfile(modelFile):
        print(data["col"] + "loading model file:"+modelFile)

        gbm = joblib.load(modelFile)

        pred_y = gbm.predict(data["x"])

        metrics = ModelUtils.getMetrics(data["y"].squeeze(), pred_y,data["y_train"])

        end = time.time()
        print(data["loghead"] + data["col"] + (' lightGBM Test MAE: %.2f' %
                                               metrics["mae"])+", spent: %.2fs" % (end - start))
        return {data["col"]: metrics}
    else:
        print(data["col"] + "No model file:"+modelFile)
        print("PLZ confirm model file first!!!")

    # return 0.0
