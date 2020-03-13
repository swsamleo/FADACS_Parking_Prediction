from models import LightGBM as lgbm
from models import FNN as fnn
from models import LSTM as lstm

import uuid
import arrow
import json
import os
import time
import numpy as np
from multiprocessing import cpu_count
from multiprocessing import Pool
from tqdm import trange, tqdm

import FeatureConvertor as fc
import ParkingDataConvertor as pdc


fc1 = fc.FeatureConvertor()

def _checkAndCreatDir(dirPath):
    try:
        os.makedirs(dirPath)
        print("Directory ", dirPath, " Created ")
    except FileExistsError:
        print("Directory ", dirPath, " already exists")


def _getDateOffsetIndex(day,interval):
    return (arrow.get(day+" 00:00","MM/DD/YYYY HH:mm") - arrow.get("01/01/2017"+" 00:00","MM/DD/YYYY HH:mm")).days*24*60//interval


def _getParkingIdDataset(id,interval,start, end,features, type = "lot"):
    startIndex = _getDateOffsetIndex(start,interval)
    endIndex = _getDateOffsetIndex(end,interval)
    _type = type
    
    p1 = pdc.getParkingEventsArray(id,interval,type = _type,start = startIndex, end = endIndex,output="numpy")
    
    if len(features) > 0:
        f1 = fc1.getFeatures(id,features,start,end,interval,type = _type)
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
    
    #for i in range(count):
    for i in trange(count):
        #print("["+str(i)+"/"+str(count)+"] X/Y Datasets")
        if X is None:
            X = parkingData[i:unit+i].reshape(-1, unit,parkingDataWidth)
            #print(X.shape)
            y = parkingDataY[yindexs+i-1].reshape(-1, 1,YWidth)
        else:
            _x = parkingData[i:unit+i].reshape(-1, unit,parkingDataWidth)
            X = np.concatenate((X,_x),axis=0)
            _y = parkingDataY[yindexs+i-1].reshape(-1, 1,YWidth)
            y = np.concatenate((y,_y),axis=0)
    
    #print("y:"+str(y.shape))
    #print("X:"+str(X.shape))
    
    return X,y
        
    
class Experience:
    def __init__(self,*args, **kwargs):
        self.config = {
            "path" : ".",
            "start" : "01/01/2017",
            "end" : "01/02/2017",
            "testStart" : "01/02/2017",
            "testEnd" : "01/03/2017",
            "location" : "MelbCity",
            "interval" : 1,
            "parkingType" : "lot",
            "features" : [],
            "predictionOffests" : [],
            "id" : str(uuid.uuid1()),
            "createDate" : arrow.now().format("YYYY-MM-DD HH:mm:ss"),
            "baseUnit": 30,
            "number":None,
            "parkingIDs":[],
            "results":{}
        }
    
    def setup(self,*args, **kwargs):
        if "reLoadExistDir" in kwargs and kwargs["reLoadExistDir"] == True:
            self.loadConfig(kwargs["path"]+'/exp.json')
        else:
            self.config["predictionOffests"] = np.arange(1, 31, 1).tolist() + \
                    np.arange(31, 60, 2).tolist() + \
                    np.arange(61, 120, 4).tolist() + \
                    np.arange(121, 240, 8).tolist() + \
                    np.arange(241, 480, 16).tolist() + \
                    np.arange(481, 24 * 60 + 1, 32).tolist()

            if self.config["interval"] > 1:
                self.config["predictionOffests"] = np.arange(1, 30 // interval + 1, 1).tolist() + \
                      np.arange(30 // interval + 1, 60 // interval, 1).tolist() + \
                      np.arange(60 // interval + 1, 120 // interval, 2).tolist() + \
                      np.arange(120 // interval + 1, 240 // interval, 4).tolist() + \
                      np.arange(240 // interval + 1, 480 // interval, 8).tolist() + \
                      np.arange(480 // interval + 1, 24 * 60 // interval + 1, 16).tolist()
                units = 2*30//self.config["interval"]

            if "path" in kwargs: self.config["path"] = kwargs["path"]
            if "parkingIDs" in kwargs: self.config["parkingIDs"] = kwargs["parkingIDs"]
            if "number" in kwargs: self.config["number"] = kwargs["number"]
            if "start" in kwargs: self.config["start"] = kwargs["start"]
            if "end" in kwargs: self.config["end"] = kwargs["end"]
            if "testStart" in kwargs: self.config["testStart"] = kwargs["testStart"]
            if "testEnd" in kwargs: self.config["testEnd"] = kwargs["testEnd"]
            if "location" in kwargs: self.config["location"] = kwargs["location"]
            if "baseUnit" in kwargs: self.config["baseUnit"] = kwargs["baseUnit"]
            if "interval" in kwargs: self.config["interval"] = kwargs["interval"]
            if "parkingType" in kwargs: self.config["parkingType"] = kwargs["parkingType"]
            if "features" in kwargs: self.config["features"] = kwargs["features"]
            if "predictionOffests" in kwargs: self.config["predictionOffests"] = kwargs["predictionOffests"]

            if len(self.config["parkingIDs"]) == 0:
                self.config["parkingIDs"] = pdc.getALLParkingAeraArray(self.config["interval"],type = self.config["parkingType"],number = self.config["number"])
   
            _checkAndCreatDir(self.config["path"])

            with open(self.config["path"]+'/exp.json', 'w') as outfile:
                json.dump(self.config, outfile)
            
    def loadConfig(self,filePath):
        with open(filePath) as json_file:
            self.config = json.load(json_file)
            
    def saveConfig(self):
        with open(self.config["path"]+'/exp.json', 'w') as outfile:
                json.dump(self.config, outfile)
                
    
    def showConfig(self):
        print(self.config)
        
    def showFeatureList(self):
        print(fc.getFeatureList())

    def _generateTTDatasets(self,type = "train"):
        
        X = None
        y = None
        
        PROCESS_NUM = cpu_count()
        
        global _getXYDate
        def _getXYDate(parkingId):
            print("generate X/y Data for " + parkingId)
            
            pdatata = None
            
            if type == "train":
                pdata = _getParkingIdDataset(parkingId,self.config["interval"],self.config["start"],self.config["end"],self.config["features"],self.config["parkingType"])
            else:
                pdata = _getParkingIdDataset(parkingId,self.config["interval"],self.config["testStart"],self.config["testEnd"],self.config["features"],self.config["parkingType"])
            
            _x,_y = _genXYDatasetByParkingIdDataset(pdata,self.config["baseUnit"],self.config["predictionOffests"])
            print("_y:"+str(_y.shape))
            print("_x:"+str(_x.shape))
            return {"x":_x,"y":_y}
        
        p = Pool(processes = PROCESS_NUM)
        XYs= p.map(_getXYDate, self.config["parkingIDs"])
       
        p.close()
        p.join()
        
        for i in range(len(self.config["parkingIDs"])):
            if X is None:
                X = XYs[i]["x"]
                y = XYs[i]["y"]
            else:
                X = np.concatenate((X,XYs[i]["x"]),axis=0)
                y = np.concatenate((y,XYs[i]["y"]),axis=0)
                
        return X,y
    
    def prepareTTDatasets(self,force = False):
        print("Prepare Train Datasets")
        xFile = self.config["path"]+'/x.npy'
        yFile = self.config["path"]+'/y.npy'
        
        if os.path.isfile(xFile) and force == False:
            print("X/Y Files are exist, skip generation!")
        else:
            print("Generate X/Y Train Files!")
            x,y = self._generateTTDatasets()
            np.save(xFile, x)
            np.save(yFile, y)
        
        xFile = self.config["path"]+'/tx.npy'
        yFile = self.config["path"]+'/ty.npy'
        
        if os.path.isfile(xFile) and force == False:
            print("X/Y Files are exist, skip generation!")
        else:
            print("Generate X/Y Test Files!")
            x,y = self._generateTTDatasets(type="test")
            np.save(xFile, x)
            np.save(yFile, y)
    
    def _trainAndTest(self,modelsName,trainMethod, testMethod, multiProcess,trainParameters = None,reTrain = False):
        filepath = self.config["path"]+"/"
        tSerial = np.core.defchararray.add("t", np.char.mod('%d', self.config["predictionOffests"]))
        
        parkingSlotsNum = self.config["number"]

        start = time.time()
        
        train_X = np.load(self.config["path"]+'/x.npy')
        train_Y = np.load(self.config["path"]+'/y.npy')
        
        trainDatasets = []
        clfs = []
        
        for i in range(len(tSerial)):
            col = tSerial[i]
            tdataset = {"col": col, 
                        "x": train_X, 
                        "y": train_Y[:,:,i],
                        "filepath" : filepath,
                        "parkingSlotsNum":self.config["number"],
                        "reTrain":reTrain,
                        "parameters":trainParameters
                       }
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
        
        test_X = np.load(self.config["path"]+'/tx.npy')
        test_Y = np.load(self.config["path"]+'/ty.npy')
        
        testDatasets = []
        res = []
        for i in range(len(tSerial)):
            log_head = "[" + str(i) + "/" + str(len(tSerial)) + "]"
            col = tSerial[i]
            testData = {"loghead": log_head, 
                        "col": col, 
                        "x": test_X, 
                        "y": test_Y[:,:,i], 
                        "clf": clfs[i],
                        "filepath":filepath,
                        "parkingSlotsNum":parkingSlotsNum,
                        "parameters":trainParameters}
            if multiProcess == 0:
                res.append(testMethod(testData))
            else:
                testDatasets.append(testData)

        if multiProcess > 0:
            p2 = Pool(processes=multiProcess)
            ares = p2.map(testMethod, testDatasets)
            p2.close()
            p2.join()
            res = res + ares

        print("TrainAndTest All Finished!")

        end = time.time()
        print("All Prediction(Test) Done, spent: %.2fs" % (end - start))

        self.config["results"][modelsName] = res
        self.saveConfig()
        print("All Finished and Results Saved!")
    
    
    def runModel(self,modelName,model,processNum,reTrain = False,trainParameters = None):
        if modelName in self.config["results"] and reTrain == False:
            print(modelName+" results data is exist, skip!")
        else:
            print("Start "+modelName+" models Training and Test!")
            p = None
            if trainParameters is not None and modelName in trainParameters:
                p = trainParameters[modelName]
            self._trainAndTest(modelName,model.train, model.test, processNum,p,reTrain)
                
            
    def runModels(self,models = ["LightGBM"],reTrain = False,trainParameters = None):
        PROCESS_NUM = cpu_count()
        filepath = self.config["path"]

#         if "HA" in models:
#             if os.path.isfile(filepath + "ha.csv"):
#                 print("HA acc data is exist, skip!")
#             else:
#                 print("Start HA models evaluation!")
#                 train_X, train_Y, test_X, test_Y = loadDataset(filepath)
#                 haEvaluate(test_X,test_Y,filepath+"svr.csv")

        if "LightGBM" in models: self.runModel("lightGBM",lgbm,PROCESS_NUM//2,reTrain,trainParameters)
        if "FNN" in models: self.runModel("FNN",fnn,0,reTrain,trainParameters)
        if "LSTM" in models: self.runModel("LSTM",lstm,0,reTrain,trainParameters)
            
            
    def removeResult(self, modelName):
        if modelName in self.config["results"] :
            del self.config["results"][modelName]
            print("Remove "+modelName+" Model Test Results")
        else:
            print(""+modelName+" Model Test Results NOT EXIST!")
        self.saveConfig()