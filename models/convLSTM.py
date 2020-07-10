import pandas as pd
import numpy as np
import math
import time
import os
import datetime

import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
import joblib
import pickle

from . import ModelUtils

"""
 Define Keras LSTM model.
 input_size shoule have 2 dimensions: (number of timestamps in history, number of features)
 ouput_size shoule be one iteger number: number of timestamps in perdiction
"""

def lstm(input_size,hidden_size,output_size,learningRate = 0.01,loss = 'mae'):

    model = Sequential()
    model.add(ConvLSTM2D(filters=128, kernel_size=(3,3), activation='relu',input_shape=input_size))
    model.add(Flatten())
    model.add(Dense(output_size))
    
    #model = Model(inputs=inputs, outputs=dense)    
    model.compile(optimizer = Adam(lr = learningRate), loss=loss,metrics = ['accuracy', 'mse','mae'])
    #model.summary()

    return model


# +
def train(data):
    print("LSTM Training [" + data["col"] + "] start!")
    
    log_dir = data["filepath"]+"/fit-" + data["col"]+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    modelFile = data["filepath"] + 'lstm_' + data["col"] + '.pkl'
    historyFile = data["filepath"] + 'hist_' + data["col"] + '.pkl'

    start = time.time()

    batch_size = 16
    epochs = 20
    monitor = 'val_loss'
    mode = "min"
    learningRate = 0.01
    loss = 'mae'
    verbose = 1
    gpu = False
    hidden_size = 100
    
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
        if "hidden_size" in data["parameters"]: hidden_size = data["parameters"]["hidden_size"]
    
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
        #open(modelFile,"a").close()

#         if gpu == True:
#             enableGPU()
            
#         with tf.device("gpu"):
        train_X = train_X.reshape(train_X.shape[0],1,train_X.shape[1],train_X.shape[2],1)
        test_X = test_X.reshape(test_X.shape[0],1,test_X.shape[1],test_X.shape[2],1)
        print(train_X.shape)
        model = lstm(input_size = (train_X.shape[1],train_X.shape[2],train_X.shape[3],train_X.shape[4]),
                     hidden_size = hidden_size,
                     output_size = train_y.shape[1],
                     learningRate = learningRate,
                     loss = loss
                    )
        es = EarlyStopping(monitor=monitor, mode=mode, verbose=verbose,patience = 3)

        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():

#         with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
            history = model.fit(train_X, train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=(test_X, test_y),
                        callbacks=[es])

            score = model.evaluate(test_X, test_y, verbose=0)
            print('Test loss:', score[0])
            print('Test Acc:', score[1])
            
            joblib.dump(model, modelFile)
            with open(historyFile, 'wb') as file_pi:
                pickle.dump(history.history, file_pi)

    end = time.time()
    print("LSTM Training Done [" + data["col"] + "], spent: %.2fs" % (end - start))
    return



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
        
        X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1],X_test.shape[2],1)

        pred_y = model.predict(X_test, 
                                     batch_size=batch_size, 
                                     verbose=0, 
                                     steps=steps, 
                                     callbacks=None, 
                                     max_queue_size=max_queue_size, 
                                     workers=workers, 
                                     use_multiprocessing=use_multiprocessing)

        metrics = ModelUtils.getMetrics(data["y"].squeeze(), pred_y,data["y_train"])
        #rmse = root_mean_square_error(data["y"].squeeze(), pred_y)
        end = time.time()
        print(data["loghead"] + data["col"] + (' LSTM Test MAE: %.2f' % metrics["mae"])+",spent: %.2fs" % (end - start))
        return {data["col"]:metrics}
    # return 0.0
