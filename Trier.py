from models import LightGBM as lgbm
from models import FNN as fnn
from models import LSTM as lstm
from models import DANN as dann
from models import ADDA as adda
from models import ModelUtils

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
from TLogger import *

logger = Logger("Triyer")

melbFC = fc.FeatureConvertor()
mpsFC = fc.FeatureConvertor(location = "Mornington")

PROCESS_NUM = cpu_count() // 2

def _checkAndCreatDir(dirPath):
    try:
        os.makedirs(dirPath)
        logger.info("Directory "+dirPath+ " Created ")
    except FileExistsError:
        logger.info("Directory "+dirPath+ " already exists")


def _getDateOffsetIndex(day,interval,start = arrow.get("01/01/2017"+" 00:00","MM/DD/YYYY HH:mm")):
    dis = (arrow.get(day+" 00:00","MM/DD/YYYY HH:mm") - start).days*24*60//interval
    logger.debug("_getDateOffsetIndex start:{} day:{} interval:{} -> {}".format(start,day,interval,dis))
    return dis


def _getParkingIdDataset(id,interval,start, end,features, paType = "lot",location = "MelbCity"):
    startIndex = 0
    endIndex = 0
    if location == "MelbCity":
        startIndex = _getDateOffsetIndex(start,interval)
        endIndex = _getDateOffsetIndex(end,interval)
    elif location == "Mornington":
        startIndex = _getDateOffsetIndex(start,interval,start = pdc.getMorningtonParkingStartDate(interval,paType = paType))
        endIndex = _getDateOffsetIndex(end,interval,start = pdc.getMorningtonParkingStartDate(interval,paType = paType))
    
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


class Experiment:
    def __init__(self,*args, **kwargs):
        self.config = {
            "path" : ".",
            "locations":{
                "MelbCity" : {
                    "start" : "01/01/2017",
                    "end" : "01/05/2017",
                    "testStart" : "02/02/2017",
                    "testEnd" : "01/06/2017",
                    "parkingIDs":[],
                    "number":None,
                },
                "Mornington" : {
                    "start" : "01/31/2020",
                    "end" : "02/07/2020",
                    "testStart" : "02/010/2020",
                    "testEnd" : "02/20/2020",
                    "parkingIDs":[],
                    "number":None,
                }
            },
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
    
    def setup(self,*args, **kwargs):
        
        if "path" in kwargs: self.config["path"] = kwargs["path"]
        if "locations" in kwargs: self.config["locations"] = kwargs["locations"]
        if "baseUnit" in kwargs: self.config["baseUnit"] = kwargs["baseUnit"]
        if "interval" in kwargs: self.config["interval"] = kwargs["interval"]
        if "parkingType" in kwargs: self.config["parkingType"] = kwargs["parkingType"]
        if "features" in kwargs: self.config["features"] = kwargs["features"]
        if "predictionOffests" in kwargs: self.config["predictionOffests"] = kwargs["predictionOffests"]
        
        if "reLoadExistDir" in kwargs and kwargs["reLoadExistDir"] == True and os.path.exists(kwargs["path"]+'/exp.json'):
            self.loadConfig(kwargs["path"]+'/exp.json')
        else:
            self.config["predictionOffests"] = np.arange(1, 31, 1).tolist() + \
                    np.arange(31, 60, 2).tolist() + \
                    np.arange(61, 120, 4).tolist() + \
                    np.arange(121, 240, 8).tolist() + \
                    np.arange(241, 480, 16).tolist() + \
                    np.arange(481, 24 * 60 + 1, 32).tolist()

            if self.config["interval"] > 1:
                interval = self.config["interval"]
                self.config["predictionOffests"] = np.arange(1, 30 // interval + 1, 1).tolist() + \
                      np.arange(30 // interval + 1, 60 // interval, 1).tolist() + \
                      np.arange(60 // interval + 1, 120 // interval, 2).tolist() + \
                      np.arange(120 // interval + 1, 240 // interval, 4).tolist() + \
                      np.arange(240 // interval + 1, 480 // interval, 8).tolist() + \
                      np.arange(480 // interval + 1, 24 * 60 // interval + 1, 16).tolist()
                units = 2*30//interval

            #if len(self.config["parkingIDs"]) == 0:
            for location in self.config["locations"]:
                if "parkingIDs" not in self.config["locations"][location]:
                    self.config["locations"][location]["parkingIDs"] = pdc.getALLParkingAeraArray(self.config["interval"],
                                                                                                  paType = self.config["parkingType"],
                                                                                                  number = self.config["locations"][location]["number"],
                                                                                                  location = location)
   
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
        
        if os.path.isfile(xFile) and force == False:
            logger.info("X/Y Files are exist, skip generation!")
        else:
            logger.info("Generate X/Y Train Files!")
            x,y = self._generateTTDatasets(location,
                                           self.config["locations"][location]["parkingIDs"],
                                           self.config["locations"][location]["start"],
                                           self.config["locations"][location]["end"])
            np.save(xFile, x)
            np.save(yFile, y)
        
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
            
    def prepareTTDatasets(self,force = False):
        for location in self.config["locations"]:
            self._prepareTTDatasets(location,force)
    
    
    def _trainAndTest(self,modelsName,uuid,trainMethod, testMethod, multiProcess,trainParameters = None,reTrain = False,Test = True):
        filepath = self.config["path"]+"/"+uuid+"/"
        modelParas = self.config["experiments"][uuid]

        _checkAndCreatDir(filepath)
        
        tSerial = np.core.defchararray.add("t", np.char.mod('%d', self.config["predictionOffests"]))
        
        parkingSlotsNum = self.config["locations"][modelParas["location"]]["number"]

        start = time.time()
        #_checkAndCreatDir(self.config["path"]+"/"+modelParas["location"])
        train_X = np.load(self.config["path"]+"/"+modelParas["location"]+'/x.npy',allow_pickle=True)
        train_Y = np.load(self.config["path"]+"/"+modelParas["location"]+'/y.npy',allow_pickle=True)
        
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
            
            #_checkAndCreatDir(self.config["path"]+"/"+modelParas["location"])
            test_X = np.load(self.config["path"]+"/"+modelParas["location"]+'/tx.npy',allow_pickle=True)
            test_Y = np.load(self.config["path"]+"/"+modelParas["location"]+'/ty.npy',allow_pickle=True)

            testDatasets = []
            res = []
            for i in range(len(tSerial)):
                log_head = "[" + str(i) + "/" + str(len(tSerial)) + "]"
                col = tSerial[i]
                testData = {"loghead": log_head, 
                            "col": col, 
                            "x": test_X, 
                            "y": test_Y[:,:,i], 
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
    
    
    def add(self,ep):
        ep["uuid"] = str(uuid.uuid1())
        ep["createDate"] = arrow.now().format("YYYY-MM-DD HH:mm:ss"),
        self.config["experiments"][ep["uuid"]] = ep
        self.saveConfig()
        
        print("Added experiment "+ep["uuid"])
        print(json.dumps(ep, indent=2, sort_keys=True))
        return ep["uuid"]
        
        
    
    def update(self,uuid,newExp):
        if uuid in self.config["experiments"]:
            newExp["uuid"] = uuid
            self.config["experiments"][uuid] = newExp
            print("Updated experiment "+uuid)
        else:
            print("experiment "+uuid +" is not Exist!")
    
    
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
        logger.info("Start transfer learning model [{}] training/testing".format(modelName))
        
        start = time.time()
        
        modelParas = self.config["experiments"][uuid]
        filepath = self.config["path"]+"/"+uuid+"/"
        _checkAndCreatDir(filepath)
        
        logger.info("loading datasets for dataloaders...")
        datasets = ModelUtils.loadDatasets(self.config["path"]+"/",modelParas["source"],modelParas["target"])
        
        tSerial = np.core.defchararray.add("t", np.char.mod('%d', self.config["predictionOffests"]))
        
        # For DANN
        if "featureSize" in modelParas["parameters"] and int(modelParas["parameters"]["featureSize"]) != int(self.config["baseUnit"]) * (1 + len(self.config["features"])):
            logger.info("featureSize:{} baseUnit:{}".format(modelParas["parameters"]["featureSize"],self.config["baseUnit"]))
            modelParas["parameters"]["featureSize"] = int(self.config["baseUnit"]) * (1 + len(self.config["features"]))
            logger.warning("runTFModel Correct featureSize:{}".format(modelParas["parameters"]["featureSize"]))
     
        # For ADDA
        if "e_input_dims" in modelParas["parameters"] and int(modelParas["parameters"]["e_input_dims"]) != int(self.config["baseUnit"]) * (1 + len(self.config["features"])):
            logger.info("e_input_dims:{} baseUnit:{}".format(modelParas["parameters"]["e_input_dims"],self.config["baseUnit"]))
            modelParas["parameters"]["e_input_dims"] = int(self.config["baseUnit"]) * (1 + len(self.config["features"]))
            logger.warning("runTFModel Correct e_input_dims:{}".format(modelParas["parameters"]["e_input_dims"]))
     
        tdata = []
        res = []
        for i in range(len(tSerial)):
            col = tSerial[i]
            _data = {
                "dataloaders":ModelUtils.getDataSrcTarLoaders(datasets,modelParas["parameters"]["batchSize"],i),
                "col": col, 
                "filepath" : filepath,
                "reTrain":reTrain,
                "parameters":modelParas["parameters"]
            }
            if processNum > 1:
                tdata.append(_data)
            else:
                model.train(_data)
                
        if processNum > 1:
            p2 = Pool(processes=multiProcess)
            ares = p2.map(testMethod, testDatasets)
            p2.close()
            p2.join()
            res = res + ares
            
        end = time.time()
        logger.info("TF Model ["+modelName+"] Train/Test Done, spent: %.2fs" % (end - start))
        
        self.loadConfig(self.config["path"]+"/exp.json")
        self.config["results"][uuid] = res
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
            if "LSTM" == experiment["model"]: self.runModel("LSTM",lstm,1,uuid,reTrain,Test)
            if "DANN" == experiment["model"]: self.runTFModel("DANN",dann,1,uuid,reTrain)
            if "ADDA" == experiment["model"]: self.runTFModel("ADDA",adda,1,uuid,reTrain)

            
    def rmResult(self, uuid):
        if uuid in self.config["results"]:
            del self.config["results"][uuid]
            self.saveConfig()
            logger.info("Removed experiment Result of "+uuid)
        else:
            logger.warning("experiment "+uuid +" is not Exist!")