from sklearn.metrics import mean_squared_error
from scipy import stats
import pandas as pd
import arrow
import os
import random
import numpy as np
from tqdm import tqdm
import math

from TLogger import *

logger = Logger("ParkingDataConvertor")
logger.setLevel(logging.INFO)


MAXPARKING_NUM = 38

MelbLotsDF = None
mpslotsDF = None
mpsSectorsDF = None

mps1mDF = None
mps5mDF = None
mps15mDF = None

def getCSVFileNames(dir):
    files = []
    for filename in os.listdir(dir):
        if filename.endswith(".csv"):
            files.append(filename[:-4])
    return files

def getDateOffsetIndex(day,interval,start = arrow.get("01/01/2017"+" 00:00","MM/DD/YYYY HH:mm")):
    dis = (arrow.get(day+" 00:00","MM/DD/YYYY HH:mm") - start).days*24*60//interval
    logger.debug("_getDateOffsetIndex start:{} day:{} interval:{} -> {}".format(start,day,interval,dis))
    return dis

def getALLParkingAeraArray(interval,paType = "lot" ,location = "MelbCity",number = None):
    if location == "MelbCity":
        list =  getCSVFileNames("./datasets/"+location+"/parking/"+paType+"s/"+str(interval)+"m/")
        if number is None:
            return list
        else:
            return random.sample(list, number)
    if location == "Mornington":
        if paType == "lot" or paType == "slot":
            df =  pd.read_csv("./datasets/"+location+"/DeviceId_Lot.csv")
            if paType == "lot":
                if number is None:
                    return df.LotId.unique()
                else:
                    return random.sample(df.LotId.unique().tolist(), number)
            elif paType == "slot":
                if number is None:
                    return df.DeviceId.unique()
                else:
                    return random.sample(df.DeviceId.unique().tolist(), number)
        elif paType == "sector":
            df =  pd.read_csv("./datasets/"+location+"/parking/sectors/sectorCounts.csv")
            if number is None:
                return df.sector.unique()
            else:
                return random.sample(df.sector.unique().tolist(), number)


def loadParkingDatasets(interval,location,paType = "lot",id = None):
    df = None
    logger.debug("loadParkingDatasets for {}, interval is {}m".format(id,interval))
    if location == "MelbCity":
        logger.debug("loadParkingDatasets interval {}".format(interval))
        if id is None:
            logger.error("loadParkingDatasets id can not be None in MelbCity dataset")
        df = pd.read_csv("./datasets/"+location+"/parking/{}".format(paType)+"s/{}".format(interval)+"m/{}".format(id)+".csv",index_col=0,parse_dates=True)
    elif location == "Mornington":
        global mps1mDF 
        global mps5mDF 
        global mps15mDF 

        if interval == 1:
            if mps1mDF is None:
                mps1mDF =  pd.read_csv("./datasets/"+location+"/parking/"+paType+"s/{}".format(interval)+"m.csv")
            df = mps1mDF
        elif interval == 5:
            if mps5mDF is None:
                mps5mDF =  pd.read_csv("./datasets/"+location+"/parking/"+paType+"s/{}".format(interval)+"m.csv")
            df = mps5mDF
        elif interval == 15:
            if mps15mDF is None:
                mps15mDF =  pd.read_csv("./datasets/"+location+"/parking/"+paType+"s/{}".format(interval)+"m.csv")
            df = mps15mDF
    
    return df


def getMorningtonParkingStartDate(interval,paType = "lot"):
    df = loadParkingDatasets(interval,"Mornington",paType = paType)
    return arrow.get(df.columns.values[1])


def getParkingEventsArray(id,interval,paType = "lot" ,location = "MelbCity",start = None,end = None, output = "list",normalize = False):
    logger.debug("getParkingEventsArray [{}] for {}, interval is {}m".format(location,id,interval))
    df = loadParkingDatasets(interval,location,paType = paType,id = id)

    if location == "MelbCity":
        if normalize:
            df[id] = df[id]/MAXPARKING_NUM
        else:
            lot = MelbLotsDF[MelbLotsDF["LotId"] == int(id)]
            df[id] = df[id]/lot.shape[0]

        if start is not None and end is not None:
            if output == "list":
                return df[id].tolist()[start:end]
            elif output == "numpy":
                return df[id].to_numpy()[start:end]
        else:
            if output == "list":
                return df[id].tolist()
            elif output == "numpy":
                return df[id].to_numpy()
            
    elif location == "Mornington":
        
        df["id"] = df["id"].astype(str)
        logger.debug("getParkingEventsArray id:{} start:{} end:{}".format(str(int(id)),start,end))
        #print(df.head())
        dfx = df[df["id"] == str(int(id))]
        del dfx["id"]

        logger.debug("getParkingEventsArray dfx:{}".format(dfx.shape))
        logger.debug("getParkingEventsArray dfx:{}".format(dfx))
        
        dd = None
        if paType == "lot":
            num = mpslotsDF[mpslotsDF["LotId"] == int(id)].shape[0]
            #print(mpslotsDF["LotId"].unique())
            if dfx.shape[0] == 1:
                dd = dfx.values[0]/num
            else:
                dd = dfx.values/num
        elif paType == "scetor":
            num = mpsSectorsDF[mpsSectorsDF["sector"] == id]["count"].values[0]
            if dfx.shape[0] == 1:
                dd = dfx.values[0]/num
            else:
                dd = dfx.values/num
        else:
            if dfx.shape[0] == 1:
                dd = dfx.values[0]
            else:
                dd = dfx.values
        
        if start is not None and end is not None:
            logger.debug("getParkingEventsArray for Mornington start:{} end:{}".format(start,end))
            return dd[start:end]
        else:
            return dd
        

# approximate radius of earth in km
def getDistance(_lat1,_lon1,_lat2,_lon2):
    R = 6373.0

    lat1 = math.radians(_lat1)
    lon1 = math.radians(_lon1)
    lat2 = math.radians(_lat2)
    lon2 = math.radians(_lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def getDistanceMatrix(lots,outputFile,location="MelbCity"):
    lotsLocation = pd.read_csv("./datasets/"+location+"/parking/LotsLocation.csv")

    ll = []
    for l in lots:
        L1d = lotsLocation[lotsLocation["LotId"] == int(l)].values[0]
        lr = []
        for n in lots:
            L2d = lotsLocation[lotsLocation["LotId"] == int(n)].values[0]
            D12 = getDistance(L1d[1],L1d[2],L2d[1],L2d[2])
            lr.append(D12)
        ll.append(lr)
    ll = np.array(ll)
    logger.debug(ll.shape)
    
    mms = MinMaxScaler()
    llx = mms.fit_transform(ll)
    llx
    np.save(outputFile, llx)
    
    
def genParkingDataMedian(interval = 5,location = "MelbCity",paType = "lot",number = None,metric = "median",minSlots = 3):
    melbLots = getALLParkingAeraArray(interval,location = location,number = number,paType = paType)

    pls = []
    lotsNum = 0
    for i in range(len(melbLots)):
        if (location == "MelbCity" and MelbLotsDF[MelbLotsDF["LotId"] == int(melbLots[i])].shape[0]  >=  minSlots ) or (location == "Mornington" and mpslotsDF[mpslotsDF["LotId"] == int(melbLots[i])].shape[0]  >=  minSlots ):
            p = getParkingEventsArray(melbLots[i],interval,output = "numpy",location = location)
            print("genParkingDataMedian for {}/{}s/{}m {}/{} [{}] ->{}".format(location,paType,interval,i,lotsNum,melbLots[i],p.shape))
            pls = np.concatenate((pls,p),axis=0)
            lotsNum = lotsNum+1

    pls = pls.reshape(lotsNum,pls.shape[0]//lotsNum)
    
    median = []
    for i in range(pls.shape[1]):
        if metric == "median":
            median.append(np.median(pls[:,i]))
        elif metric == "mode":
            median.append(stats.mode(pls[:,i]))

    np.save("./datasets/"+location+"/parking/"+paType+"s/"+str(interval)+"m-"+metric+"-"+str(minSlots)+".npy",median)

    
def getMedian(start,end,interval = 5,location = "MelbCity",metric = "median",minSlots = 3,paType = "lot"):
    startIndex = 0
    endIndex = 0
    if location == "MelbCity":
        startIndex = getDateOffsetIndex(start,interval)
        endIndex = getDateOffsetIndex(end,interval)
    elif location == "Mornington":
        startIndex = getDateOffsetIndex(start,interval,start = getMorningtonParkingStartDate(interval,paType = paType))
        endIndex = getDateOffsetIndex(end,interval,start = getMorningtonParkingStartDate(interval,paType = paType))

    median = np.load("./datasets/"+location+"/parking/lots/"+str(interval)+"m-"+metric+"-"+str(minSlots)+".npy",allow_pickle=True)

    if len(median.shape) == 3:
        median = np.delete(median,0,1)
        median = median.reshape(median.shape[0])

    return median[startIndex:endIndex]


def getSimilarLots(start,end,interval = 5,location = "MelbCity",number = None,metric = "median",minSlots = 3, median = None,paType = "lot"):
    startIndex = 0
    endIndex = 0
    if location == "MelbCity":
        startIndex = getDateOffsetIndex(start,interval)
        endIndex = getDateOffsetIndex(end,interval)
    elif location == "Mornington":
        startIndex = getDateOffsetIndex(start,interval,start = getMorningtonParkingStartDate(interval,paType = paType))
        endIndex = getDateOffsetIndex(end,interval,start = getMorningtonParkingStartDate(interval,paType = paType))

    #days = (endIndex - startIndex)//(24*60//interval)

    if median is None:
        median = getMedian(startIndex,endIndex,interval = interval,location = location,metric = metric,minSlots = minSlots)
    
    print(median.shape)

    melbLots = getALLParkingAeraArray(interval,location = location)
    
    pls = []
    lotsNum = len(melbLots)
    
    for i in range(len(melbLots)):
        if (location == "MelbCity" and MelbLotsDF[MelbLotsDF["LotId"] == int(melbLots[i])].shape[0]  >=  minSlots ) or (location == "Mornington" and mpslotsDF[mpslotsDF["LotId"] == int(melbLots[i])].shape[0]  >=  minSlots ):
            print("getSimilarLots {}/{} {}".format(i,lotsNum,melbLots[i]))
            d = mean_squared_error(median, getParkingEventsArray(melbLots[i],interval,start = startIndex,end = endIndex,output = "numpy",location = location))
            pls.append([melbLots[i],d])
    
    pls = np.array(pls)
    pls.sort(axis=0)

    if number is not None:
        pls = pls[:number]
        
    return pls[:,0].tolist()