import pandas as pd
import arrow
import os
import random
import numpy as np
from TLogger import *

logger = Logger("ParkingDataConvertor")
logger.setLevel(logging.INFO)


MAXPARKING_NUM = 38

lotsDF = pd.read_csv("./datasets/MelbCity/parking/StreetMarker_Lot.csv")
mpslotsDF = pd.read_csv("./datasets/Mornington/DeviceId_Lot.csv")
mpsSectorsDF = pd.read_csv("./datasets/Mornington/parking/sectors/sectorCounts.csv")

mps1mDF = None
mps5mDF = None
mps15mDF = None

def getCSVFileNames(dir):
    files = []
    for filename in os.listdir(dir):
        if filename.endswith(".csv"):
            files.append(filename[:-4])
    return files

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
    df = loadParkingDatasets(interval,location,paType = paType,id = id)
    
    logger.debug("getParkingEventsArray [{}] for {}, interval is {}m".format(location,id,interval))
    
    if location == "MelbCity":
        if normalize:
            df[id] = df[id]/MAXPARKING_NUM
        else:
            lot = lotsDF[lotsDF["LotId"] == int(id)]
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
        
        dfx = df[df["id"] == str(id)]
        del dfx["id"]
        
        dd = None
        if paType == "lot":
            num = mpslotsDF[mpslotsDF["LotId"] == id].shape[0]
            dd = dfx.values[0]/num
        elif paType == "scetor":
            num = mpsSectorsDF[mpsSectorsDF["sector"] == id]["count"].values[0]
            dd = dfx.values[0]/num
        else:
            dd = dfx
        
        if start is not None and end is not None:
            logger.debug("getParkingEventsArray for Mornington start:{} end:{}".format(start,end))
            return dd[start:end]
        else:
            return dd