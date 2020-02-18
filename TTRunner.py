import DatasetConvertor

from multiprocessing import cpu_count
from multiprocessing import Pool
import os.path

import pandas as pd
import numpy as np
import math
import time
from sklearn import metrics
from sklearn.svm import SVR, SVC, LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import torch.utils.data as utils
import torch
from torch.autograd import Variable
from torch import nn
import matplotlib.pyplot as plt
from sklearn.externals import joblib

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.optimizers import SGD

import _pickle as cPickle

import cProfile

filepath = "/run/user/1000/gvfs/smb-share:server=ds-tokyo.local,share=downloads/pp/"+"2D50S/"
PROCESS_NUM = cpu_count()

# SVC model training method
def svcTrain(data):
    print("SVR Training [" + data["col"] + "] start!")

    modelFile = data["filepath"]+'svr_'+data["col"]+'.pkl'

    start = time.time()
    weight = [0.5 if i == 0 else 1 for i in data["y"].values]
    clf = SVC(gamma='auto')

    if os.path.isfile(modelFile):
        print (data["col"] + " Training Model File exist, skip training, load it")
        with open(modelFile, 'rb') as fid:
            clf = cPickle.load(fid)
    else:
        clf.fit(data["x"].values, data["y"].values.ravel(), sample_weight=weight)
        with open(modelFile, 'wb') as fid:
            cPickle.dump(clf, fid)

    end = time.time()
    print("SVR Training Done [" + data["col"] + "], spent: %.2fs" % (end - start))
    return clf
    # return 0.0


# SVC model training method
def svcTest(data):
    print(data["loghead"] + data["col"] + " start!")

    modelFile = data["filepath"]+'svr_'+data["col"]+'.pkl'

    pred_y = data["clf"].predict(data["x"].values)
    #print(pred_y)
    #print(pred_y.shape)
    acc = metrics.accuracy_score(data["y"].values, pred_y)
    print(data["loghead"] + data["col"] + (' SVR Test Acc: %.2f' % acc))
    return acc
    # return 0.0


# LSTM Model
class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x


use_cuda = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LSTM model training method
def lstmTrain(data):
    '''
    data.col : time serial point, e.g. t1, t2 ,....,tN
    data.x : train_x dataset
    data.y : train_y dataset
    '''
    modelFile = data["filepath"] + 'lstm_params_' + data["col"] + '.pkl'

    start = time.time()

    net = lstm(30, 15)

    if os.path.isfile(modelFile):
        print (data["col"] + " Training Model File exist, skip training, load it")
        net.load_state_dict(torch.load(modelFile))
        return net

    print("LSTM Training [" + data["col"] + "] start!")

    if use_cuda and torch.cuda.is_available():
        print("using GPU")
        net = lstm(30, 15).to(DEVICE)

    x = data["x"].values.reshape(-1, 1, 30)
    # print(x.shape)
    y = data["y"].values.reshape(-1, 1, 1)
    # print(y.shape)

    tx = torch.from_numpy(x).float()
    ty = torch.from_numpy(y).float()

    var_x = Variable(tx)
    var_y = Variable(ty)

    if use_cuda and torch.cuda.is_available():
        var_x = Variable(tx).to(DEVICE)
        var_y = Variable(ty).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    for e in range(500):
        # print("e:"+str(e) + " @"+ data["col"])
        # print(var_x.shape)
        # print(var_x.shape)
        # print(var_y.shape)
        # print(type(var_x))
        # print(var_x)
        # print(var_x.float())
        out = net(var_x)

        loss = criterion(out, var_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (e + 1) % 100 == 0:
            print(data["col"] + 'Epoch: {}, Loss:{:.5f}'.format(e + 1, loss.data))

    end = time.time()
    print("LSTM Training Done [" + data["col"] + "], spent: %.2fs" % (end - start))
    torch.save(net.state_dict(), modelFile)
    return net
    # return 0.0


# LSTM model Test method
def lstmTest(data):
    '''
    data.col : time serial point, e.g. t1, t2 ,....,tN
    data.clf : model
    data.x : test_x dataset
    data.y : test_y dataset
    data.loghead : e.g. "40.00%(10003S)[2/5]"
    '''
    modelFile = data["filepath"] + 'lstm_params_' + data["col"] + '.pkl'

    net = lstm(30, 15)

    if os.path.isfile(modelFile):
        # print (data["col"]+" Training Model File exist,load it")
        net.load_state_dict(torch.load(modelFile))

    print(data["loghead"] + data["col"] + " LSTM Test start!")

    tx = data["x"].values.reshape(-1, 1, 30)
    # print(x.shape)
    y = data["y"].values
    # print(y.shape)
    # print(y)

    var_x = Variable(torch.from_numpy(tx).float())

    # if use_cuda and torch.cuda.is_available():
    #    var_x = Variable(torch.from_numpy(tx).float()).to(DEVICE)

    # pred_test = data["clf"](var_x)
    pred_test = net(var_x)
    pred_test = np.array([1 if x >= 0.5 else 0 for x in pred_test])
    # print(pred_test)
    # print(pred_test.shape)

    acc = metrics.accuracy_score(y, pred_test)
    print(data["loghead"] + data["col"] + (' LSTM Test Acc: %.2f' % acc))

    return acc
    # return 0.0


# lightGBM model training method
def lightGBMTrain(data, reTrain = False):
    print("lightGBM Training [" + data["col"] + "] start!")

    modelFile = data["filepath"] + 'lightgbm_' + data["col"] + '.pkl'

    start = time.time()

    eval_size = int(data["x"].shape[0] * 0.1)
    # print("eval_size:"+str(eval_size))
    # print(data["x"].shape)

    eval_x = data["x"].iloc[:eval_size, :]
    eval_y = data["y"].iloc[:eval_size, :]
    # print(eval_x.shape)
    # print(data["x"].shape)

    # print(eval_y.values.shape)
    # print(data["y"].values.shape)

    # yy = np.array(eval_y).squeeze()
    # print(yy)

    gbm = lgb.LGBMClassifier(num_leaves=20 * data["parkingSlotsNum"],
                             learning_rate=0.05,
                             n_estimators=10 * data["parkingSlotsNum"])

    if os.path.isfile(modelFile) and reTrain == False:
        print (data["col"] + " Training Model File exist, skip training, load it")
        gbm = joblib.load(modelFile)
    else:
        gbm.fit(data["x"].values, np.array(data["y"]).squeeze(),
                eval_set=[(eval_x.values, np.array(eval_y).squeeze())],
                eval_metric='l1',
                early_stopping_rounds=25*data["parkingSlotsNum"])
        joblib.dump(gbm, modelFile)

    end = time.time()
    print("lightGBM Training Done [" + data["col"] + "], spent: %.2fs" % (end - start))
    return gbm


# lightGBM model training method
def lightGBMTest(data):
    print(data["loghead"] + data["col"] + " start!")
    pred_y = data["clf"].predict(data["x"].values)

    #print(pred_y)
    #print(pred_y.shape)

    #print(np.array(data["y"]).squeeze())
    #print(np.array(data["y"]).squeeze().shape)

    acc = metrics.accuracy_score(np.array(data["y"]).squeeze(), pred_y)
    print(data["loghead"] + data["col"] + (' lightGBM Test Acc: %.2f' % acc))
    return acc
    # return 0.0


def fnn_keras(input_size = (1,30)):
    inputs = Input(input_size)
    conv1 = Conv1D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv1D(30, 5, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling1D(pool_size=1)(conv1)
    flat = Flatten()(pool1)
    results = Dense(1, activation='sigmoid')(flat)

    model = Model(inputs=inputs, outputs=results)
    model.summary()

    sgd = SGD(lr=0.01, momentum=0.9, decay=0.01,nesterov=False)
    model.compile(optimizer=Adam(lr = 0.01), loss='binary_crossentropy', metrics = ['accuracy'])

    return model


def fnnTrain(data,reTrain = False):
    print("FNN Training [" + data["col"] + "] start!")

    modelFile = data["filepath"] + 'fnn_' + data["col"] + '.pkl'

    start = time.time()

    batch_size = 16
    epochs = 20

    model = fnn_keras()

    eval_size = int(data["x"].shape[0] * 0.1)
    # print("eval_size:"+str(eval_size))
    # print(data["x"].shape)

    eval_x = data["x"].iloc[:eval_size, :]
    eval_y = data["y"].iloc[:eval_size, :]

    X_train = data["x"].values
    X_train = X_train.reshape(X_train.shape[0],1,30)
    X_test = eval_x.values
    X_test = X_test.reshape(X_test.shape[0],1,30)

    y_train = data["y"].values
    y_train = y_train.reshape(-1,1)
    y_test = eval_y.values
    y_test = y_test.reshape(-1,1)

    if os.path.isfile(modelFile) and reTrain == False:
        print (data["col"] + " Training Model File exist, skip training, load it")
        model = joblib.load(modelFile)
    else:
        model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(X_test, y_test))
        joblib.dump(model, modelFile)

    end = time.time()
    print("FNN Training Done [" + data["col"] + "], spent: %.2fs" % (end - start))
    return model


def fnnTest(data):
    print(data["loghead"] + data["col"] + " start!")
    
    X_test = data["x"].values
    X_test = X_test.reshape(X_test.shape[0],1,30)
    y_test = data["y"].values
    y_test = y_test.reshape(-1,1)
    
    score = data["clf"].evaluate(X_test, y_test, verbose=0)

    print(data["loghead"] + data["col"] + (' FNN Test Acc: %.2f' % score[1]))
    return score[1]
    # return 0.0


def haEvaluate(test_x,test_y,resultFile):
    #test_x = pd.read_csv(dataset_path + "/" + test_x_file_name)
    #test_y = pd.read_csv(dataset_path + "/" + test_y_file_name)

    pred_result = test_x.mean(axis=1).apply(lambda x: 1 if x>=0.5 else 0) # Use mean for prediction
    number_of_data = test_y.shape[0]/test_y['id'].nunique() # The number of data of each slot

    col_list = train_y.iloc[:,1:].columns.values # Timing list
    pred_test = test_y['id'].to_frame()
    pred_test = pred_test.assign(**{col:pred_result for col in col_list}) # Use same mean result for all timing to be predicted

    accuracy_matrix = test_y.iloc[:,1:].values == pred_test.iloc[:,1:].values
    pd_accuracy_matrix = pd.DataFrame.from_records(accuracy_matrix) # Numpy array to dataframe
    pd_accuracy_matrix['id'] = test_y['id']
    pd_accuracy_matrix = pd_accuracy_matrix.set_index('id')

    output = pd_accuracy_matrix.groupby('id').apply(lambda x: round(x[x == True].count()/number_of_data,2)) # Evaluation result
    output.to_csv(resultFile)


def runTTXY(trainMethod, testMethod,filepath, train_X, train_Y, test_X, test_Y, output, filename, multiProcess):
    parkingSlotsNum = len(train_X.id.unique())

    start = time.time()

    trainx = train_X
    _trainy = train_Y.copy()
    del _trainy["id"]
    del trainx["id"]
    trainDatasets = []
    clfs = []
    for col in _trainy.columns:
        tdataset = {"col": col, "x": trainx, "y": _trainy[[col]],"filepath" : filepath,"parkingSlotsNum":parkingSlotsNum}
        trainDatasets.append(tdataset)
        if multiProcess == 0:
            clfs.append(trainMethod(tdataset))

    if multiProcess > 0:
        p = Pool(processes=multiProcess)
        clfs = p.map(trainMethod, trainDatasets)
        p.close()
        p.join()

    end = time.time()
    print("All Training Done, spent: %.2fs" % (end - start))

    start = time.time()
    it = 1
    for i in train_Y.id.unique():
        log_head = ("{:.2f}%(" + str(i) + ")[" + str(it) + "/" + str(parkingSlotsNum) + "]").format(
            100 * it / parkingSlotsNum)
        res = [i]
        # print("debug i:"+i+test_X.head())
        testx = test_X[test_X["id"] == i]
        _testy = test_Y[test_Y["id"] == i]
        del testx["id"]
        del _testy["id"]
        colIndex = 0
        testDatasets = []
        for col in _trainy.columns:
            trainData = {"loghead": log_head, "col": col, "x": testx, "y": _testy[[col]], "clf": clfs[colIndex],"filepath":filepath,"parkingSlotsNum":parkingSlotsNum}
            testDatasets.append(
                trainData)
            if multiProcess == 0:
                # print(log_head + col + " start!")
                # testy = _testy[[col]]
                # pred_y = clfs[colIndex].predict(testx.values)
                # acc = metrics.accuracy_score(testy.values, pred_y)
                # res.append(acc)
                # print(log_head + col + (' Test Acc: %.3f' % acc))
                res.append(testMethod(trainData))
            colIndex += 1

        if multiProcess > 0:
            p2 = Pool(processes=multiProcess)
            ares = p2.map(testMethod, testDatasets)
            p2.close()
            p2.join()
            output.loc[+1] = res + ares

        if multiProcess == 0:
            output.loc[+1] = res

        output.index = output.index - 1
        it += 1
        print(log_head + " All Finished!")

    end = time.time()
    print("All Prediction(Test) Done, spent: %.2fs" % (end - start))

    output = output.sort_index()
    output.to_csv(filepath + filename, index=False)
    print("All Finished and Saved!")


def loadDataset(filepath):
    print("start loading datasets......")
    train_X = pd.read_csv(filepath + "train_x.csv")
    train_Y = pd.read_csv(filepath + "train_y.csv")
    test_X = pd.read_csv(filepath + "test_x.csv")
    test_Y = pd.read_csv(filepath + "test_y.csv")
    print("loaded datasets")
    return train_X, train_Y, test_X, test_Y

def generateTTDataSetAndRun(
        parkingMinsFile='/run/user/1000/gvfs/smb-share:server=ds-tokyo.local,share=downloads/pp/rmit-parking-prediction/car_parking_2017_1mins_point.csv',
        outputDir='/run/user/1000/gvfs/smb-share:server=ds-tokyo.local,share=downloads/pp/',
        parkingSlotsNum=5,
        interval = 1,
        trainStart="01/01/2017",
        trainEnd="01/03/2017",
        testStart="01/08/2017",
        testEnd="01/11/2017",
        tts=["SVR","LSTM","LightGBM"]):
    datasetName = str(interval)+"M"+str(DatasetConvertor.diffDays(trainStart, trainEnd)) + 'D' + str(parkingSlotsNum) + 'S'
    filepath = outputDir + datasetName + '/'

    if os.path.isfile(filepath + "readme.txt"):
        print("The TT datasets of "+datasetName+" are exist,skip generation.")
    else:
        DatasetConvertor.genTTDataset(source=parkingMinsFile,
                                     interval = interval,
                                     outputPath=outputDir,
                                     dirName=datasetName,
                                     slotsNum=parkingSlotsNum,
                                     trainStart=trainStart,
                                     trainEnd=trainEnd,
                                     testStart=testStart,
                                     testEnd=testEnd
                                     )

    # HA
    if "HA" in tts:
        if os.path.isfile(filepath + "ha.csv"):
            print("HA acc data is exist, skip!")
        else:
            print("Start HA models evaluation!")
            train_X, train_Y, test_X, test_Y = loadDataset(filepath)
            haEvaluate(test_X,test_Y,filepath+"svr.csv")
            

    # SVR
    if "SVR" in tts:
        if os.path.isfile(filepath + "svr.csv"):
            print("SVR acc data is exist, skip!")
        else:
            print("Start SVR models Training and Test!")
            train_X, train_Y, test_X, test_Y = loadDataset(filepath)
            rowsNum = len(open(filepath + "train_x.csv").readlines())
            if rowsNum >= 10000:
                print("Skip SVR Train and Test because the sample is more than 10000")
            else:
                runTTXY(svcTrain, svcTest, filepath, train_X, train_Y, test_X, test_Y, pd.DataFrame(columns=train_Y.columns), "svr.csv", PROCESS_NUM)


    #LSTM
    if "LSTM" in tts:
        if os.path.isfile(filepath+"lstm.csv"):
            print("LSTM acc data is exist, skip!")
        else:
            print("Start LSTM models Training and Test!")
            train_X, train_Y, test_X, test_Y = loadDataset(filepath)
            runTTXY(lstmTrain, lstmTest, filepath, train_X, train_Y, test_X, test_Y, pd.DataFrame(columns=train_Y.columns), "lstm.csv", 0)


    # LightGBM
    if "LightGBM" in tts:
        reTrain = False

        if os.path.isfile(filepath + "lightGBM.csv") and reTrain == False:
            print("lightGBM acc data is exist, skip!")
        else:
            print("Start LightGBM models Training and Test!")
            train_X, train_Y, test_X, test_Y = loadDataset(filepath)
            runTTXY(lightGBMTrain, lightGBMTest, filepath, train_X, train_Y, test_X, test_Y, pd.DataFrame(columns=train_Y.columns), "lightGBM.csv", PROCESS_NUM//2)

    # FNN
        if "FNN" in tts:
            reTrain = False

            if os.path.isfile(filepath + "fnn.csv") and reTrain == False:
                print("FNN acc data is exist, skip!")
            else:
                print("Start FNN models Training and Test!")
                train_X, train_Y, test_X, test_Y = loadDataset(filepath)
                runTTXY(FNNTrain, FNNTest, filepath, train_X, train_Y, test_X, test_Y, pd.DataFrame(columns=train_Y.columns), "fnn.csv", PROCESS_NUM//2)


