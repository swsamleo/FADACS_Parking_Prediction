import pandas as pd
import arrow
import os
import random
import numpy as np

MAXPARKING_NUM = 38

def getCSVFileNames(dir):
    files = []
    for filename in os.listdir(dir):
        if filename.endswith(".csv"):
            files.append(filename[:-4])
    return files

def getALLParkingAeraArray(interval,type = "lot" ,location = "MelbCity",number = None):
    list =  getCSVFileNames("./datasets/"+location+"/parking/"+type+"s/"+str(interval)+"m/")
    if number is None:
        return list
    else:
        return random.sample(list, number)

def getParkingEventsArray(id,interval,type = "lot" ,location = "MelbCity",start = None,end = None, output = "list",normalize = False):
    df = pd.read_csv("./datasets/"+location+"/parking/"+type+"s/"+str(interval)+"m/"+id+".csv",index_col=0,parse_dates=True)
    if normalize:
        df[id] = df[id]/MAXPARKING_NUM
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