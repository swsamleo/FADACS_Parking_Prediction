import pandas as pd
import numpy as np
import math
import time
import os

import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
import joblib

from . import ModelUtils

def enableGPU():
    config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 7} ) 
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)
    print("Enabled GPU!")

"""
 Define Keras LSTM model.
 input_size shoule have 2 dimensions: (number of timestamps in history, number of features)
 ouput_size shoule be one iteger number: number of timestamps in perdiction
"""

def lstm(input_size,output_size,learningRate = 0.01,loss = 'mae'):
    inputs = Input(input_size)
    lstm1 = LSTM(100,activation ='relu')(inputs)
    #lstm2 = LSTM(100,activation ='relu')(lstm1)
    dense = Dense(output_size)(lstm1)
    
    model = Model(inputs=inputs, outputs=dense)
    model.summary()
    
    model.compile(optimizer = Adam(lr = learningRate), loss=loss,metrics = ['acc'])
    
    return model


# +
def train(data):
    print("LSTM Training [" + data["col"] + "] start!")

    modelFile = data["filepath"] + 'lstm_' + data["col"] + '.pkl'

    start = time.time()

    batch_size = 16
    epochs = 20
    monitor = 'val_loss'
    mode = "min"
    learningRate = 0.01
    loss = 'mae'
    verbose = 1
    gpu = False
    
    print("Input parameters:")
    print(data["parameters"])
    
    if data["parameters"] is not None:
        if "batch_size" in data["parameters"]: batch_size = data["parameters"]["batch_size"]
        if "epochs" in data["parameters"]: epochs = data["parameters"]["epochs"]
        if "monitor" in data["parameters"]: monitor = data["parameters"]["monitor"]
        if "mode" in data["parameters"]: mode = data["parameters"]["mode"]
        if "verbose" in data["parameters"]: verbose = data["parameters"]["verbose"]
        if "learningRate" in data["parameters"]: learningRate = data["parameters"]["learningRate"]
        if "loss" in data["parameters"]: loss = data["parameters"]["loss"]
        if "gpu" in data["parameters"]: gpu = data["parameters"]["gpu"]
    
    eval_size = int(data["x"].shape[0] * 0.1)
    # print("eval_size:"+str(eval_size))
    # print(data["x"].shape)
    #data["x"] = data["x"].reshape(-1,1, rowSize)
    #data["y"] = data["y"].reshape(-1,1)

    train_X = data["x"]
    test_X = data["x"][:eval_size, :,:]

    train_y = data["y"]
    test_y = data["y"][:eval_size, :]

    if os.path.isfile(modelFile) and data["reTrain"] == False:
        print (data["col"] + " Training Model File exist, skip training!")
        return
#         model = joblib.load(modelFile)
    else:
        open(modelFile,"a").close()

#         if gpu == True:
#             enableGPU()
            
#         with tf.device("gpu"):
        model = lstm(input_size = (train_X.shape[1],train_X.shape[2]),
                     output_size = train_y.shape[1],
                     learningRate = learningRate,
                     loss = loss
                    )
        es = EarlyStopping(monitor=monitor, mode=mode, verbose=verbose)

#         with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        model.fit(train_X, train_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=(test_X, test_y),
                    callbacks=[es])

        score = model.evaluate(test_X, test_y, verbose=0)
        print('Test loss:', score[0])
        print('Test Acc:', score[1])
        
        joblib.dump(model, modelFile)

    end = time.time()
    print("LSTM Training Done [" + data["col"] + "], spent: %.2fs" % (end - start))
    return
#     return model


# -

def test(data):
    print(data["loghead"] + data["col"] + " start!")
    start = time.time()
    modelFile = data["filepath"] + 'lstm_' + data["col"] + '.pkl'

    X_test = data["x"]
    y_test = data["y"]
    
    batch_size = 16
    epochs = 20
    steps = None
    max_queue_size = 10
    workers = 1
    use_multiprocessing = False
    verbose = 1
    gpu = False

    if data["parameters"] is not None:
        if "batch_size" in data["parameters"]: batch_size = data["parameters"]["batch_size"]
        if "epochs" in data["parameters"]: epochs = data["parameters"]["epochs"]
        if "verbose" in data["parameters"]: verbose = data["parameters"]["verbose"]
        if "steps" in data["parameters"]: steps = data["parameters"]["steps"]
        if "max_queue_size" in data["parameters"]: max_queue_size = data["parameters"]["max_queue_size"]
        if "workers" in data["parameters"]: workers = data["parameters"]["workers"]
        if "use_multiprocessing" in data["parameters"]: use_multiprocessing = data["parameters"]["use_multiprocessing"]
        if "gpu" in data["parameters"]: gpu = data["parameters"]["gpu"]

    if os.path.isfile(modelFile):
        print (data["col"] + "loading model file:"+modelFile)
        model = joblib.load(modelFile)
    
        pred_y = model.predict(X_test, 
                                     batch_size=batch_size, 
                                     verbose=0, 
                                     steps=steps, 
                                     callbacks=None, 
                                     max_queue_size=max_queue_size, 
                                     workers=workers, 
                                     use_multiprocessing=use_multiprocessing)

        metrics = ModelUtils.getMetrics(data["y"].squeeze(), pred_y)
        #rmse = root_mean_square_error(data["y"].squeeze(), pred_y)
        end = time.time()
        print(data["loghead"] + data["col"] + (' LSTM Test MAE: %.2f' % metrics["mae"])+",spent: %.2fs" % (end - start))
        return {data["col"]:metrics}
    # return 0.0
