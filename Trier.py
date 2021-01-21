# @Time     : Jul. 10, 2020 19:45
# @Author   : Zhen Zhang
# @Email    : david.zhen.zhang@gmail.com
# @FileName : Trier.py
# @Version  : 1.0
# @IDE      : VSCode

from models import LightGBM as lgbm
from models import FNN as fnn
from models import LSTM as lstm
from models import convLSTM as convlstm
from models import DANN as dann
from models import ADDA as adda
from models import HA as ha
from models import ARIMA as armia
from models import ModelUtils

import itertools as it
import uuid
import arrow
import json
import os
import time
import numpy as np
from multiprocessing import cpu_count
from multiprocessing import Pool
from tqdm import trange, tqdm
import pandas as pd
from sklearn.feature_selection import f_regression
from IPython.display import HTML, display

import FeatureConvertor as fc
import ParkingDataConvertor as pdc
from TLogger import *

logger = Logger("Triyer")

melbFC = None
mpsFC = None

PROCESS_NUM = cpu_count() // 2

def _checkAndCreatDir(dirPath):
    try:
        os.makedirs(dirPath)
        logger.info("Directory "+dirPath+ " Created ")
    except FileExistsError:
        logger.info("Directory "+dirPath+ " already exists")


def _getParkingIdDataset(id,interval,start, end,features, paType = "lot",location = "MelbCity"):
    startIndex = 0
    endIndex = 0
    if location == "MelbCity":
        startIndex = pdc.getDateOffsetIndex(start,interval)
        endIndex = pdc.getDateOffsetIndex(end,interval)
    elif location == "Mornington":
        startIndex = pdc.getDateOffsetIndex(start,interval,start = pdc.getMorningtonParkingStartDate(interval,paType = paType))
        endIndex = pdc.getDateOffsetIndex(end,interval,start = pdc.getMorningtonParkingStartDate(interval,paType = paType))
    
    _type = paType
    logger.debug("_getParkingIdDataset startIndex:{} endIndex:{} start:{} end:{}".format(startIndex,endIndex,start,end))
    p1 = pdc.getParkingEventsArray(id,
                                   interval,
                                   paType = _type,
                                   start = startIndex, 
                                   end = endIndex,
                                   output="numpy",
                                   location = location)
    
    if len(features) > 0:
        f1 = None
        if location == "MelbCity":
            f1 = melbFC.getFeatures(id,features,start,end,interval,paType = _type)
        elif location == "Mornington":
            f1 = mpsFC.getFeatures(id,features,start,end,interval,paType = _type)
        logger.debug("_getParkingIdDataset p1.shape:{}".format(p1.shape))
        logger.debug("_getParkingIdDataset f1.shape:{}".format(f1.shape))
        return np.concatenate((p1.reshape((-1, 1)),f1),axis=1)
    else:
        return p1.reshape((-1, 1))


def _genXYDatasetByParkingIdDataset(parkingData,unit,yIndexes):
    parkingDataLength = parkingData.shape[0]
    parkingDataWidth = parkingData.shape[1]
    
    parkingDataY = parkingData[:,:1]
    
    YWidth = len(yIndexes)
    count = parkingDataLength - unit - yIndexes[len(yIndexes) - 1] + 1
    
    yindexs = np.array(yIndexes) + unit
    X = None
    y = None
    
    logger.debug("_genXYDatasetByParkingIdDataset parkingDataLength:{} parkingDataWidth:{} parkingDataY.shape:{}".format(parkingDataLength,parkingDataWidth,parkingDataY.shape))
    logger.debug("_genXYDatasetByParkingIdDataset YWidth:{} unit:{} count:{} len(yindexs):{}".format(YWidth,unit,count,len(yindexs)))
    
    #for i in range(count):
    for i in trange(count):
        logger.debug("["+str(i)+"/"+str(count)+"] generate X/Y Datasets")
        if X is None:
            X = parkingData[i:unit+i].reshape(-1, unit,parkingDataWidth)
            logger.debug("X.shape:{}".format(X.shape))
            y = parkingDataY[yindexs+i-1].reshape(-1, 1,YWidth)
        else:
            _x = parkingData[i:unit+i].reshape(-1, unit,parkingDataWidth)
            X = np.concatenate((X,_x),axis=0)
            _y = parkingDataY[yindexs+i-1].reshape(-1, 1,YWidth)
            logger.debug("_y.shape:{}".format(_y.shape))
            y = np.concatenate((y,_y),axis=0)
    
    logger.debug("y:"+str(y.shape))
    logger.debug("X:"+str(X.shape))
    
    return X,y


def genSTDataset(lots,start,end,interval,features,outputfile,location = "MelbCity",paType = "lot"):
    dd = np.array([])
    for lot in lots:
        pdd = _getParkingIdDataset(lot,interval,start, end,features, paType = "lot")
        if dd.shape[0] == 0:
            dd = pdd.reshape(-1,1,4)
        else:
            dd = np.concatenate((dd,pdd.reshape(-1,1,len(features)+1)),axis=1)
    print(dd.shape)


class Experiment:
    def __init__(self,*args, **kwargs):
        self.config = {
            "path" : ".",
            "locations":{
                "MelbCity" : {
                    "start" : "01/01/2017",
                    "end" : "01/05/2017",
                    #"testStart" : "02/02/2017",
                    #"testEnd" : "01/06/2017",
                    "parkingIDs":[],
                    "number":None,
                    "randomParkingIDs":False,
                    "ParkingIDMetric":"median"
                },
                "Mornington" : {
                    "start" : "01/31/2020",
                    "end" : "02/07/2020",
                    #"testStart" : "02/010/2020",
                    #"testEnd" : "02/20/2020",
                    "parkingIDs":[],
                    "number":None,
                    "randomParkingIDs":False,
                    "ParkingIDMetric":"median"
                }
            },
            "medianSource":None,
            "interval" : 1,
            "parkingType" : "lot",
            "features" : [],
            "predictionOffests" : [],
            "id" : str(uuid.uuid1()),
            "createDate" : arrow.now().format("YYYY-MM-DD HH:mm:ss"),
            "baseUnit": 30,
            "experiments":{},
            "results":{}
        }

    def getParkingIds(self,location):
        return self.config["locations"][location]["parkingIDs"];

    def convertHADataset(self,location):

        path = self.config["path"]+"/"+location
        
        xFile = path+'/x.npy'
        yFile = path+'/y.npy'

        ids = self.getParkingIds(location)
        
        x = np.load(xFile,allow_pickle=True)
        y = np.load(yFile,allow_pickle=True)
        
        tx = x[:,:,0]
        y = y.reshape(y.shape[0],y.shape[2])

        
        l = tx.shape[0]/len(ids)
        
        dx=pd.DataFrame(data=tx[0:,0:],index=[i for i in range(tx.shape[0])],columns=['t'+str(tx.shape[1] - i-1) for i in range(tx.shape[1])])
        dy=pd.DataFrame(data=y[0:,0:],index=[i for i in range(tx.shape[0])],columns=['t'+str(i) for i in self.config["predictionOffests"]])
        
        dx["id"] = ""
        dy["id"] = ""
        
        for index, id in enumerate(ids):
            #print(str(index*l)+" -> " +str((index+1)*l ))
            dx.iloc[int(index*l):int((index+1)*l )]["id"] = id
            dy.iloc[int(index*l):int((index+1)*l )]["id"] = id
        
        return dx,dy
    
    def showFeaturesCorrelationCoefficient(self,location,start,end,features = None):
        #location = "MelbCity"
        if features is None:
            features = self.config["features"]

        ids = self.getParkingIds(location)

        npd = None
        for id in ids:
            p1 = _getParkingIdDataset(id,self.config["interval"],start,end,features,location=location)
            #print(p1.shape)
            if npd is None:
                npd = p1
            else:
                npd = np.concatenate((npd,p1),axis=0)

        df=pd.DataFrame(data=npd[0:,0:],index=[i for i in range(npd.shape[0])],columns=["occupancy"]+features)

        c = df.corr(method ='pearson')["occupancy"].values[1:]

        F, pval = f_regression(npd[0:,1:len(features)+1],npd[0:,0])

        rst = np.column_stack( (features,c,F,pval))
        rdf=pd.DataFrame(data=rst[0:,0:],index=[i for i in range(rst.shape[0])],columns=["Features","Pearson Correlation Coefficient","F Value","p-Value"])
        rdf =rdf.sort_values(by=['Pearson Correlation Coefficient'],ascending=False)

        html = '<table><tr><th>Features</th><th>Pearson<p>Correlation<p>Coefficient</th><th>F Value</th><th>p-Value</th></tr>'

        for i in rdf.index:
            line =  ('<tr><td>{}</td><td>{:10.2f}</td><td>{:10.2f}</td><td>{}</td></tr>').format(features[i],c[i],F[i],("0" if pval[i] == 0 else "{:.2e}".format(pval[i]) ) )
            html += line

        html += '</table>'
        display(HTML(html))
    
    def setup(self,*args, **kwargs):
        
        if "path" in kwargs: self.config["path"] = kwargs["path"]
        if "medianSource" in kwargs: self.config["medianSource"] = kwargs["medianSource"]
        if "locations" in kwargs: self.config["locations"] = kwargs["locations"]
        if "baseUnit" in kwargs: self.config["baseUnit"] = kwargs["baseUnit"]
        if "interval" in kwargs: self.config["interval"] = kwargs["interval"]
        if "parkingType" in kwargs: self.config["parkingType"] = kwargs["parkingType"]
        if "features" in kwargs: self.config["features"] = kwargs["features"]
        if "predictionOffests" in kwargs: self.config["predictionOffests"] = kwargs["predictionOffests"]
        
        if "reLoadExistDir" in kwargs and kwargs["reLoadExistDir"] == True and os.path.exists(kwargs["path"]+'/exp.json'):
            self.loadConfig(kwargs["path"]+'/exp.json')
        else:
            interval = self.config["interval"]
            self.config["predictionOffests"] = [5//interval,15//interval,30//interval]
            
            # self.config["predictionOffests"] = np.arange(1, 31, 1).tolist() + \
            #         np.arange(31, 60, 2).tolist() + \
            #         np.arange(61, 120, 4).tolist() + \
            #         np.arange(121, 240, 8).tolist() + \
            #         np.arange(241, 480, 16).tolist() + \
            #         np.arange(481, 24 * 60 + 1, 32).tolist()

            # if self.config["interval"] > 1:
            #     interval = self.config["interval"]
            #     self.config["predictionOffests"] = np.arange(1, 30 // interval + 1, 1).tolist() + \
            #           np.arange(30 // interval + 1, 60 // interval, 1).tolist() + \
            #           np.arange(60 // interval + 1, 120 // interval, 2).tolist() + \
            #           np.arange(120 // interval + 1, 240 // interval, 4).tolist() + \
            #           np.arange(240 // interval + 1, 480 // interval, 8).tolist() + \
            #           np.arange(480 // interval + 1, 24 * 60 // interval + 1, 16).tolist()
            #     units = 2*30//interval

            #if len(self.config["parkingIDs"]) == 0:

            median = None
            if self.config["medianSource"] is not None:
                location = self.config["medianSource"]
                median = pdc.getMedian(self.config["locations"][location]["start"],
                                        self.config["locations"][location]["end"],
                                        interval = self.config["interval"],
                                        location = location,
                                        metric = self.config["locations"][location]["ParkingIDMetric"])

            for location in self.config["locations"]:
                if "parkingIDs" not in self.config["locations"][location]:
                    if self.config["locations"][location]["randomParkingIDs"] == False:
                        logger.info("Choose "+location+" ParkingIDs via ["+self.config["locations"][location]["ParkingIDMetric"]+"]")
                        self.config["locations"][location]["parkingIDs"] = pdc.getSimilarLots(self.config["locations"][location]["start"],
                                                                                            self.config["locations"][location]["end"],
                                                                                            interval = self.config["interval"],
                                                                                            location = location,
                                                                                            number = self.config["locations"][location]["number"],
                                                                                            metric = self.config["locations"][location]["ParkingIDMetric"],
                                                                                            median = median)
                    else:
                        self.config["locations"][location]["parkingIDs"] = pdc.getALLParkingAeraArray(self.config["interval"],
                                                                                                    paType = self.config["parkingType"],
                                                                                                    number = self.config["locations"][location]["number"],
                                                                                                    location = location)
                    logger.info(self.config["locations"][location]["parkingIDs"])
                    
   
            _checkAndCreatDir(self.config["path"])

            with open(self.config["path"]+'/exp.json', 'w+') as outfile:
                json.dump(self.config, outfile)
            
    def loadConfig(self,filePath):
        with open(filePath) as json_file:
            self.config = json.load(json_file)
            
    def saveConfig(self):
        with open(self.config["path"]+'/exp.json', 'w+') as outfile:
                json.dump(self.config, outfile)
                
        
    def showConfig(self):
        print(json.dumps(self.config, indent=2, sort_keys=True))
        
    def showFeatureList(self):
        print(melbFC.getFeatureList())

    def getAllFeaturesList(self):
        return melbFC.getFeatureList()

    def _generateTTDatasets(self,location,parkingIDs,start,end):
        logger.debug(parkingIDs)
        X = None
        y = None
        
        global _getXYDate
        def _getXYDate(parkingId):
            logger.debug("generate X/y Data for:{} start:{} end:{}".format(parkingId,start,end))
            
            pdata = _getParkingIdDataset(parkingId,
                                            self.config["interval"],
                                            start,
                                            end,
                                            self.config["features"],
                                            self.config["parkingType"],
                                            location = location)
            logger.debug("pdata shape:{}".format(pdata.shape))
            _x,_y = _genXYDatasetByParkingIdDataset(pdata,
                                                    self.config["baseUnit"],
                                                    self.config["predictionOffests"])
            logger.debug("_y:"+str(_y.shape))
            logger.debug("_x:"+str(_x.shape))
            return {"x":_x,"y":_y}
        
        
        PROCESS_NUM = cpu_count()
        
        p = Pool(processes = PROCESS_NUM - 1)
        XYs = p.map(_getXYDate, parkingIDs)
       
        p.close()
        p.join()
        
        for i in range(len(parkingIDs)):
            if X is None:
                X = XYs[i]["x"]
                y = XYs[i]["y"]
            else:
                X = np.concatenate((X,XYs[i]["x"]),axis=0)
                y = np.concatenate((y,XYs[i]["y"]),axis=0)
        
        where_are_NaNs = np.isnan(X)
        X[where_are_NaNs] = 0
        
        where_are_NaNs = np.isnan(y)
        y[where_are_NaNs] = 0
                
        return X,y
    
    def _prepareTTDatasets(self,location,force = False):
        logger.info("Prepare Train Datasets for "+location)
        
        path = self.config["path"]+"/"+location
        
        _checkAndCreatDir(path)
        
        xFile = path+'/x.npy'
        yFile = path+'/y.npy'
        
        trainRrcNum = 0
        
        if os.path.isfile(xFile) and force == False:
            logger.info("X/Y Files are exist, skip generation!")
        else:
            logger.info("Generate X/Y Train Files!")
            x,y = self._generateTTDatasets(location,
                                           self.config["locations"][location]["parkingIDs"],
                                           self.config["locations"][location]["start"],
                                           self.config["locations"][location]["end"])
            trainRrcNum = x.shape[0]
            
            np.save(xFile, x)
            np.save(yFile, y)
        
        if "testStart" in self.config["locations"][location]:
            logger.info("Prepare Test Datasets for "+location)
            xFile = path+'/tx.npy'
            yFile = path+'/ty.npy'
            
            if os.path.isfile(xFile) and force == False:
                logger.info("X/Y Files are exist, skip generation!")
            else:
                logger.info("Generate X/Y Test Files!")
                x,y = self._generateTTDatasets(location,
                                            self.config["locations"][location]["parkingIDs"],
                                            self.config["locations"][location]["testStart"],
                                            self.config["locations"][location]["testEnd"])
                np.save(xFile, x)
                np.save(yFile, y)
        else:
            logger.info("NOT Generate Test Datasets for "+location+" in a time range!")
            ttrate = 0.25
            if "TTRate" in self.config["locations"][location]:
                ttrate = self.config["locations"][location]["TTRate"]
            
            logger.info("The Train:Test datasets split rate is {}:{}".format(1-ttrate,ttrate))
            
            splitIndex = int((1-ttrate) * trainRrcNum*10//10)
            
            arr = np.arange(trainRrcNum)
            
            if "random" in self.config["locations"][location]:
                logger.info("Generate random index for datasets")
                np.random.shuffle(arr)

            trIndex = path+'/trainIndex.npy'
            teIndex = path+'/testIndex.npy'
            
            np.save(trIndex, arr[:splitIndex])
            np.save(teIndex, arr[splitIndex:])


    def prepareTTDatasets(self,force = False):
        for location in self.config["locations"]:
            self._prepareTTDatasets(location,force)


    def _trainAndTest(self,modelsName,uuid,trainMethod, testMethod, multiProcess,trainParameters = None,reTrain = False,Test = True):
        filepath = self.config["path"]+"/"+uuid+"/"
        modelParas = self.config["experiments"][uuid]

        trainWithParkingData = True
        if "trainWithParkingData" in modelParas:
            trainWithParkingData = modelParas["trainWithParkingData"]

        _checkAndCreatDir(filepath)
        
        tSerial = np.core.defchararray.add("t", np.char.mod('%d', self.config["predictionOffests"]))
        
        parkingSlotsNum = self.config["locations"][modelParas["location"]]["number"]

        start = time.time()
        
        train_X = None
        train_Y = None
        #_checkAndCreatDir(self.config["path"]+"/"+modelParas["location"])
        dX = np.load(self.config["path"]+"/"+modelParas["location"]+'/x.npy',allow_pickle=True)
        dY = np.load(self.config["path"]+"/"+modelParas["location"]+'/y.npy',allow_pickle=True)
        
        trIndex = None
        teIndex = None
        if os.path.exists(self.config["path"]+"/"+modelParas["location"]+'/trainIndex.npy') == True:
            trIndex = np.load(self.config["path"]+"/"+modelParas["location"]+'/trainIndex.npy',allow_pickle=True)
            
            train_X = dX[trIndex]
            train_Y = dY[trIndex]
        else:
            train_X = dX
            train_Y = dY

        if trainWithParkingData == False:
            train_X = train_X[:,:,1:train_X.shape[2]]
        
        trainDatasets = []
        clfs = []
        
        for i in range(len(tSerial)):
            col = tSerial[i]
            tdataset = {
                        "col": col, 
                        "x": train_X, 
                        "y": train_Y[:,:,i],
                        "filepath" : filepath,
                        "parkingSlotsNum":parkingSlotsNum,
                        "reTrain":reTrain,
                        "parameters":trainParameters
                       }
            if multiProcess <= 1:
#                 clfs.append(trainMethod(tdataset))
                trainMethod(tdataset)
            else:
                trainDatasets.append(tdataset)
                
        if multiProcess > 1:
            p = Pool(processes=multiProcess)
            p.map(trainMethod, trainDatasets)
            p.close()
            p.join()

        end = time.time()
        logger.info("All Training Done, spent: %.2fs" % (end - start))
        
        if Test:
            logger.info("Start Test Models with Test Dataset!")
            start = time.time()
            
            test_X = None
            test_Y = None

            tlocation = modelParas["location"]
            if "testLocation" in modelParas:
                tlocation = modelParas["testLocation"]
            
            if os.path.exists(self.config["path"]+"/"+tlocation+'/testIndex.npy') == True:
                teIndex = np.load(self.config["path"]+"/"+tlocation+'/testIndex.npy',allow_pickle=True)
                print(teIndex)
                test_X = dX[teIndex]
                test_Y = dY[teIndex]
            else:
                test_X = np.load(self.config["path"]+"/"+tlocation+'/tx.npy',allow_pickle=True)
                test_Y = np.load(self.config["path"]+"/"+tlocation+'/ty.npy',allow_pickle=True)

            if trainWithParkingData == False:
                test_X = test_X[:,:,1:test_X.shape[2]]

            logger.info("test_X.shape {}".format(test_X.shape))
            logger.info("test_Y.shape {}".format(test_Y.shape))

            testDatasets = []
            res = []
            for i in range(len(tSerial)):
                log_head = "[" + str(i) + "/" + str(len(tSerial)) + "]"
                col = tSerial[i]
                testData = {"loghead": log_head, 
                            "col": col, 
                            "x": test_X, 
                            "y": test_Y[:,:,i],
                            "y_train":train_Y[:,:,i].ravel(),
                            "filepath":filepath,
                            "parkingSlotsNum":parkingSlotsNum,
                            "parameters":trainParameters}
                if multiProcess <= 1:
                    res.append(testMethod(testData))
                else:
                    testDatasets.append(testData)

            if multiProcess > 1:
                p2 = Pool(processes=multiProcess)
                ares = p2.map(testMethod, testDatasets)
                p2.close()
                p2.join()
                res = res + ares

        logger.info("TrainAndTest All Finished!")

        end = time.time()
        logger.info("All Prediction(Test) Done, spent: %.2fs" % (end - start))
        
        self.loadConfig(self.config["path"]+"/exp.json")
        self.config["results"][uuid] = res
        self.config["experiments"][uuid]["status"] = "done"
        self.saveConfig()
        logger.info("All Finished and Results Saved!")
    
    
    def runModel(self,modelName,model,processNum,uuid,reTrain = False,Test = True):
        if uuid in self.config["results"] and reTrain == False:
            logger.info(modelName+" results data is exist, skip!")
        else:
            logger.info("Start "+modelName+" ["+uuid+"] models Training and Test!")
            p = None
            if "parameters" in self.config["experiments"][uuid]:
                p = self.config["experiments"][uuid]["parameters"]
            self._trainAndTest(modelName,uuid,model.train, model.test, processNum,trainParameters = p,reTrain = reTrain,Test = Test)
    
    def runARMIA(self,uuid,reTrain = False):
        if "status" in self.config["experiments"][uuid] and self.config["experiments"][uuid]["status"] == "running":
            logger.info("ARMIA " + uuid + " is running, skip!")
            return 

        start = time.time()
        if uuid in self.config["results"] and reTrain == False:
            logger.info("ARIMA evaluate results data is exist, skip!")
        else:
            res = []
            modelParas = self.config["experiments"][uuid]            
            
            if "testStart" not in modelParas or "testEnd" not in modelParas:
                logger.info("ARIMA evaluate STOP! missing test date range (testStart,testEnd)")
                return

            params = None
            if "parameters" in self.config["experiments"][uuid]:
                params = modelParas["parameters"]

            tSerial = np.core.defchararray.add("t", np.char.mod('%d', self.config["predictionOffests"]))

            #for i in range(len(tSerial)):
            i = len(tSerial) -1
            predictOffest = self.config["predictionOffests"][i]
            
            m = {
            "mae" :0,
            "rmse"  : 0,
            "mase" : 0,
            "r2" : 0,
            }

            for pid in self.config["locations"][modelParas["location"]]["parkingIDs"]:
                trainY = _getParkingIdDataset(str(int(pid)),
                                            self.config["interval"],
                                            self.config["locations"][modelParas["location"]]["start"],
                                            self.config["locations"][modelParas["location"]]["end"],
                                            [],
                                            location = modelParas["location"])
                testY = _getParkingIdDataset(pid,
                                            self.config["interval"],
                                            modelParas["testStart"],
                                            modelParas["testEnd"],
                                            [],
                                            location = modelParas["location"])

                print("testStart:{}  testEnd:{} ".format(modelParas["testStart"],modelParas["testEnd"]))
                print("trainY:{},testY:{}".format(trainY.shape,testY.shape))

                trainY = trainY.reshape(trainY.shape[0])
                testY = testY.reshape(testY.shape[0])

                _m = armia.evaluate(predictOffest,trainY, testY, params)

                m["mae"] += _m["mae"]
                m["rmse"] += _m["rmse"]
                m["rmse"] += _m["mase"]
                m["r2"] += _m["r2"]

                xlen = len(self.config["locations"][modelParas["location"]]["parkingIDs"])
                    
                m["mae"] = m["mae"]/xlen
                m["rmse"] = m["rmse"]/xlen
                m["rmse"] = m["mase"]/xlen
                m["r2"] = m["r2"]/xlen

                res.append({tSerial[i]:m})

            end = time.time()
            logger.info("ARIMA evaluation, spent: %.2fs" % (end - start))

            self.loadConfig(self.config["path"]+"/exp.json")
            self.config["results"][uuid] = res
            self.config["experiments"][uuid]["status"] = "done"
            self.saveConfig()
            logger.info("All ARIMA evaluation Finished and Results Saved!")


    def runHA(self,uuid,multiProcess,reTrain = False):
        if "status" in self.config["experiments"][uuid] and self.config["experiments"][uuid]["status"] == "running":
            logger.info("HA " + uuid + " is running, skip!")
            return

        start = time.time()
        if uuid in self.config["results"] and reTrain == False:
            logger.info("HA evaluate results data is exist, skip!")
        else:
            modelParas = self.config["experiments"][uuid]

            logger.info("Start evaluate HA ["+uuid+"] models Training and Test!")

            tSerial = np.core.defchararray.add("t", np.char.mod('%d', self.config["predictionOffests"]))

            x,y = self.convertHADataset(modelParas["location"])

            res = []
            datasets = []

            for i in range(len(tSerial)):
                log_head = "[" + str(i) + "/" + str(len(tSerial)) + "]"
                col = tSerial[i]
                data = {"loghead": log_head, 
                            "col": col, 
                            "x": x, 
                            "y": y,
                            "y_train":None,
                            "tIndex":i
                            }
                if multiProcess <= 1:
                    res.append(ha.evaluate(data))
                else:
                    datasets.append(data)

            if multiProcess > 1:
                p2 = Pool(processes=multiProcess)
                ares = p2.map(ha.evaluate, datasets)
                p2.close()
                p2.join()
                res = res + ares

            end = time.time()
            logger.info("HA evaluation, spent: %.2fs" % (end - start))
            
            self.loadConfig(self.config["path"]+"/exp.json")
            self.config["results"][uuid] = res
            self.config["experiments"][uuid]["status"] = "done"
            self.saveConfig()
            logger.info("All HA evaluation Finished and Results Saved!")


    def add(self,ep):
        ep["uuid"] = str(uuid.uuid1())
        ep["createDate"] = arrow.now().format("YYYY-MM-DD HH:mm:ss"),
        self.config["experiments"][ep["uuid"]] = ep
        self.saveConfig()
        
        print("Added experiment "+ep["uuid"])
        print(json.dumps(ep, indent=2, sort_keys=True))
        return ep["uuid"]
    

    def addADDAExps(self,options):
        keys = options.keys()
        values = (options[key] for key in keys)
        combinations = [dict(zip(keys, combination)) for combination in it.product(*values)]
        for para in combinations:
            self.add(
            {
                "model": "FADACS",
                "source": "MelbCity",
                "target": "Mornington",
                "trainWithParkingData":False,
                "parameters": para
            })
    
    def update(self,uuid,newExp):
        if uuid in self.config["experiments"]:
            newExp["uuid"] = uuid
            self.config["experiments"][uuid] = newExp
            print("Updated experiment "+uuid)
        else:
            print("experiment "+uuid +" is not Exist!")
    
    
    def rmAll(self):
        for ep in list(self.config["experiments"]):
            #print(ep)
            self.rm(ep)


    def rm(self,ep):
        if ep in self.config["experiments"]:
            xexp = self.config["experiments"][ep]
            del self.config["experiments"][ep]
            
            logger.info("Removed experiment "+ep)
            print(json.dumps(xexp, indent=2, sort_keys=True))
            
            if ep in self.config["results"]:
                del self.config["results"][ep]
                logger.info("Removed experiment Result of "+ep)
                
            self.saveConfig()
        else:
            logger.warning("experiment "+ep +" is not Exist!")
            
    
    def show(self,modelType = "all"):
        for ep in self.config["experiments"]:
            if modelType == "all" or modelType == self.config["experiments"][ep]["model"]:
                print(json.dumps(self.config["experiments"][ep], indent=2, sort_keys=True))
                
    
    def runTFModel(self,modelName,model,processNum,uuid,reTrain = False):
        if uuid in self.config["results"] and reTrain == False:
            logger.info(modelName+" results data is exist, skip!")
            return

        if "status" in self.config["experiments"][uuid] and self.config["experiments"][uuid]["status"] == "running":
            logger.info(modelName + " " + uuid + " is running, skip!")
            return

        self.config["experiments"][uuid]["status"] = "running"
        self.saveConfig()

        logger.info("Start transfer learning model [{}] training/testing".format(modelName))
        
        start = time.time()
        
        modelParas = self.config["experiments"][uuid]
        filepath = self.config["path"]+"/"+uuid+"/"
        _checkAndCreatDir(filepath)
        
        logger.info("loading datasets for dataloaders...")
        #datasets = ModelUtils.loadDatasets(self.config["path"]+"/",modelParas["source"],modelParas["target"])
        
        tSerial = np.core.defchararray.add("t", np.char.mod('%d', self.config["predictionOffests"]))

        trainWithParkingData = True
        if "trainWithParkingData" in modelParas:
            trainWithParkingData = modelParas["trainWithParkingData"]
        
        # For DANN
        if "featureSize" in modelParas["parameters"] and int(modelParas["parameters"]["featureSize"]) != int(self.config["baseUnit"]) * (1 + len(self.config["features"])):
            logger.info("featureSize:{} baseUnit:{}".format(modelParas["parameters"]["featureSize"],self.config["baseUnit"]))
            modelParas["parameters"]["featureSize"] = int(self.config["baseUnit"]) * ((1 if trainWithParkingData else 0) + len(self.config["features"]))
            logger.warning("runTFModel Correct featureSize:{}".format(modelParas["parameters"]["featureSize"]))
     
        # For FADACS / ADDA
        if "e_input_dims" in modelParas["parameters"] and int(modelParas["parameters"]["e_input_dims"]) != int(self.config["baseUnit"]) * (1 + len(self.config["features"])):
            logger.info("e_input_dims:{} baseUnit:{}".format(modelParas["parameters"]["e_input_dims"],self.config["baseUnit"]))
            modelParas["parameters"]["e_input_dims"] = int(self.config["baseUnit"]) * ((1 if trainWithParkingData else 0) + len(self.config["features"]))
            logger.warning("runTFModel Correct e_input_dims:{}".format(modelParas["parameters"]["e_input_dims"]))
     
        tdata = []
        res = []
        for i in range(len(tSerial)):
            col = tSerial[i]
            datasets = ModelUtils.loadDatasets(self.config["path"]+"/",modelParas["source"],modelParas["target"],tIndex=i,trainWithParkingData = trainWithParkingData)
            y_train = datasets["srcTrain"].getY()
            _data = {
                "dataloaders":ModelUtils.getDataSrcTarLoaders(datasets,modelParas["parameters"]["batchSize"]),
                "col": col, 
                "filepath" : filepath,
                "reTrain":reTrain,
                "baseUnit":self.config["baseUnit"],
                "featureNum":(1 if trainWithParkingData else 0) + len(self.config["features"]),
                "y_train":y_train[:,:,i].ravel(),
                "parameters":modelParas["parameters"]
            }
            if processNum > 1:
                tdata.append(_data)
            else:
                res.append(model.train(_data))
                
        if processNum > 1:
            p2 = Pool(processes=processNum)
            ares = p2.map(model.train, tdata)
            p2.close()
            p2.join()
            res = res + ares
            
        end = time.time()
        logger.info("TF Model ["+modelName+"] Train/Test Done, spent: %.2fs" % (end - start))
        
        self.loadConfig(self.config["path"]+"/exp.json")
        self.config["results"][uuid] = res
        self.config["experiments"][uuid]["status"] = "done"
        self.saveConfig()
        logger.info("Results Saved!")
        
    
    #def runModels(self,models = ["LightGBM"],reTrain = False,trainParameters = None):
    def run(self,ep = None,reTrain = False,Test = True):
        
        if ep == None:
            logger.info("Run all experiments! because no special experiment assigned")
            for uuid in self.config["experiments"]:
                logger.info("Start Run "+uuid)
                self.run(uuid)
        else:
            experiment = ep
            uuid = None
            #if ep is uuid
            if type(ep) == str:
                experiment = self.config["experiments"][ep]
                uuid = ep
            elif "uuid" not in experiment:
                uuid = self.add(experiment)
                experiment = self.config["experiments"][uuid]

            if "LightGBM" == experiment["model"]: self.runModel("lightGBM",lgbm,1,uuid,reTrain,Test)
            if "FNN"  == experiment["model"]: self.runModel("FNN",fnn,1,uuid,reTrain,Test)
            if "convLSTM" == experiment["model"]: self.runModel("convLSTM",convlstm,1,uuid,reTrain,Test)
            if "LSTM" == experiment["model"]: self.runModel("LSTM",lstm,1,uuid,reTrain,Test)
            if "DANN" == experiment["model"]: self.runTFModel("DANN",dann,1,uuid,reTrain)
            if "FADACS" == experiment["model"]: self.runTFModel("FADACS",adda,1,uuid,reTrain)
            if "HA" == experiment["model"]: self.runHA(uuid,3,reTrain)
            if "ARIMA" == experiment["model"]: self.runARMIA(uuid,reTrain)
            
    def rmResult(self, uuid):
        if uuid in self.config["results"]:
            del self.config["results"][uuid]
            self.saveConfig()
            logger.info("Removed experiment Result of "+uuid)
        else:
            logger.warning("experiment "+uuid +" is not Exist!")