import os
import numpy as np
import math
import time

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import joblib

from . import ModelUtils



def fnn_keras(input_size = (1,30),
              learningRate = 0.01,
              momentum = 0.9,
              decay = 0.01,
              nesterov = False,
              loss = 'mean_squared_error',
              activation = "relu"):
    inputs = Input(input_size)
    conv1 = Conv1D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv1D(input_size[1], 5, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling1D(pool_size=1)(conv1)
    flat = Flatten()(pool1)
    results = Dense(1, activation='relu')(flat)

    model = Model(inputs=inputs, outputs=results)
    model.summary()

    sgd = SGD(lr=learningRate, momentum=momentum, decay=decay,nesterov=nesterov)
    model.compile(optimizer=Adam(lr = learningRate), loss=loss, metrics = ['accuracy'])

    return model


def train(data):
    print("FNN Training [" + data["col"] + "] start!")

    modelFile = data["filepath"] + 'fnn_' + data["col"] + '.pkl'

    start = time.time()

    batch_size = 16
    epochs = 20
    learningRate = 0.01
    momentum = 0.9
    decay = 0.01
    nesterov = False
    loss = 'mean_squared_error'
    activation = "relu"
    verbose = 1
    gpu = False

    if data["parameters"] is not None:
        if "batch_size" in data["parameters"]: batch_size = data["parameters"]["batch_size"]
        if "epochs" in data["parameters"]: epochs = data["parameters"]["epochs"]
        if "learningRate" in data["parameters"]: learningRate = data["parameters"]["learningRate"]
        if "momentum" in data["parameters"]: momentum = data["parameters"]["momentum"]
        if "decay" in data["parameters"]: decay = data["parameters"]["decay"]
        if "nesterov" in data["parameters"]: nesterov = data["parameters"]["nesterov"]
        if "loss" in data["parameters"]: loss = data["parameters"]["loss"]
        if "verbose" in data["parameters"]: verbose = data["parameters"]["verbose"]
        if "gpu" in data["parameters"]: gpu = data["parameters"]["gpu"]
        if "activation" in data["parameters"]: activation = data["parameters"]["activation"]

    if gpu:
        K.tensorflow_backend._get_available_gpus()
    
    rowSize = data["x"].shape[1]*data["x"].shape[2]

    eval_size = int(data["x"].shape[0] * 0.1)
    # print("eval_size:"+str(eval_size))
    # print(data["x"].shape)
    data["x"] = data["x"].reshape(-1,1, rowSize)
    #data["y"] = data["y"].reshape(-1,1)

    X_train = data["x"]
    X_test = data["x"][:eval_size, :]

    y_train = data["y"]
    y_test = data["y"][:eval_size, :]

    if os.path.isfile(modelFile) and data["reTrain"] == False:
        print (data["col"] + " Training Model File exist, skip training!")
        return
        #model = joblib.load(modelFile)
    else:
        open(modelFile,"a").close()
        model = fnn_keras(input_size=(1,rowSize),
                          learningRate = learningRate,
                          momentum = momentum,
                          decay = decay,
                          nesterov = nesterov,
                          loss = loss,
                          activation = activation
                         )
        model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                validation_data=(X_test, y_test))
        joblib.dump(model, modelFile)

    end = time.time()
    print("FNN Training Done [" + data["col"] + "], spent: %.2fs" % (end - start))
    return model


def test(data):
    print(data["loghead"] + data["col"] + " start!")
    start = time.time()

    modelFile = data["filepath"] + 'fnn_' + data["col"] + '.pkl'

    rowSize = data["x"].shape[1]*data["x"].shape[2]
    data["x"] = data["x"].reshape(-1,1,rowSize)
    #data["y"] = data["y"].reshape(-1,1)
    
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
        print (data["col"] + " loading model file:"+modelFile)
        model = joblib.load(modelFile)
            
        pred_y = model.predict(X_test, 
                                     batch_size=batch_size, 
                                     verbose=0, 
                                     steps=steps, 
                                     callbacks=None, 
                                     max_queue_size=max_queue_size, 
                                     workers=workers, 
                                     use_multiprocessing=use_multiprocessing)

        metrics = ModelUtils.getMetrics(y_test.squeeze(), pred_y)
        
        end = time.time()
        print(data["loghead"] + data["col"] + (' FNN Test MAE: %.2f' % metrics["mae"])+", spent: %.2fs" % (end - start))
        return {data["col"]:metrics}
    else:
        print(data["col"] + "No model file:"+modelFile)
        print("PLZ confirm model file first!!!")
    # return 0.0
