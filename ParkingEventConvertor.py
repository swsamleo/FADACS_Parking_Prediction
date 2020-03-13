import DatasetConvertor
import pandas as pd
import numpy as np
import csv
import time
import arrow
from multiprocessing import cpu_count
from multiprocessing import Pool
from itertools import product
from itertools import repeat
import os.path

# Function to insert row in the dataframe 
def Insert_row_(row_number, df, row_value): 
    # Slice the upper half of the dataframe 
    df1 = df[0:row_number] 
   
    # Store the result of lower half of the dataframe 
    df2 = df[row_number:] 
   
    # Inser the row in the upper half dataframe 
    df1.loc[row_number]=row_value 
   
    # Concat the two dataframes 
    df_result = pd.concat([df1, df2]) 
   
    # Reassign the index labels 
    df_result.index = [*range(df_result.shape[0])] 
   
    # Return the updated dataframe 
    return df_result


def resampleParkingEvents(df,StreetMarker,interval):
    d1 = df[df["StreetMarker"] == StreetMarker]
    d1.reset_index(inplace = True)
    d1 = d1.rename(columns={"availibility":"status"})
    #del d1["index"]
    del d1["StreetMarker"]
    
    dh = d1[d1["datetime"] == "2017-01-01 00:00"]
    d2016 = d1[d1["datetime"].str.contains("2016")]

    if dh.shape[0] == 1: # and d2016.shape[0] > 0 :
        d1 = d1.drop(np.arange(dh.index.values[0]).tolist())
    #else if dh.shape[0] == 1  and d2016.shape[0] == 0 :
    elif dh.shape[0] == 0 and d2016.shape[0] == 0 :
        first = d1.iloc[0].values
        first[2] = "2017-01-01 00:00"
        d1 = Insert_row_(0,d1,first)
        #d1.loc[len(d1)] = first
        #d1 = d1.sort_index() 
        #d1.insert(loc=0,value=first)
    
    end = d1.iloc[-1].values
    #print("end 0 :"+str(end))
    end[2] = "2017-12-31 23:59"
    d1 = Insert_row_(len(d1),d1,end)
    #d1.loc[len(d1)] = end
    
    #print(d1.iloc[-1].values)
    
    del d1['Unnamed: 0']
    del d1['index']

    d1.index = pd.to_datetime(d1.apply(lambda row: arrow.get(row["datetime"],"YYYY-MM-DD HH:mm").format('X'), axis=1), unit='s')
    d1 = d1.loc[~d1.index.duplicated(keep='first')]
    
    
    d1 = d1.resample(str(interval)+'T').bfill()
    #print(d1.iloc[-1].values)
    del d1["datetime"]

    return d1


def convertParkingEventData(input = "./datasets/MelbCity/car_parking_2017.csv",
                            output = "./datasets/MelbCity/carParking_2017_event2.csv"):
    parkingDataFile = input
    #print("skip checking line number of the input file ...")
    # count = len(open(parkingDataFile).readlines())
    count = 17932633
    print("parking Mins file has lines number:"+str(count))

    outputCSV = open(output, 'w')
    outputCSV.write("datetime,StreetMarker,status\n")

    parkingSlotNum = 0

    with open(parkingDataFile, "r") as f:
        reader = csv.reader(f, delimiter=",")
        streetMarker = ""
        for i, line in enumerate(reader):
            log_head = ("{:.2f}%(" + str(parkingSlotNum) + ")[" + str(i) + "/" + str(count) + "][" + line[
                    0] + "]").format(100 * i / count)

            if line[0] == "DeviceId":
                    continue
            else:
                if streetMarker != line[4]:
                    parkingSlotNum += 1
                    streetMarker = line[4]
                    print(log_head + " -> " + streetMarker)
                arv = arrow.get(line[1],"MM/DD/YYYY HH:mm:ss")
                dpt = arrow.get(line[2],"MM/DD/YYYY HH:mm:ss")
                outputCSV.write(arv.shift(seconds=-60).format("YYYY-MM-DD HH:mm")+","+streetMarker+",0\n")
                outputCSV.write(arv.format("YYYY-MM-DD HH:mm")+","+streetMarker+",1\n")
                outputCSV.write(dpt.format("YYYY-MM-DD HH:mm")+","+streetMarker+",1\n")
                outputCSV.write(dpt.shift(seconds=+60).format("YYYY-MM-DD HH:mm")+","+streetMarker+",0\n")
                #print(log_head + "" + streetMarker + " " + line[1] + " 1")
                #print(log_head + "" + streetMarker + " " + line[2] + " 0")

        outputCSV.close()

def resampleAllParkingSlotsStatus(df,interval):
    global _getResampledParkingSlotStatus
    def _getResampledParkingSlotStatus(streetMarker):
        start = time.time()
        print("Resample "+streetMarker)
        dx1 = resampleParkingEvents(df,streetMarker,interval)
        dx1 = dx1.rename(columns={"status":streetMarker})
        #slots.append(dx1)
        end = time.time()
        print(streetMarker+" Done!, spent: %.2fs" % (end - start))
        dx1.to_csv("./datasets/MelbCity/slots/"+streetMarker+".csv")
        return []
#     slots = []
#     for streetMarker in df.StreetMarker.unique():
#         start = time.time()
#         print("Resample "+streetMarker)
#         dx1 = resampleParkingEvents(dcp,streetMarker,interval)
#         dx1 = dx1.rename(columns={"status":streetMarker})
#         slots.append(dx1)
#         end = time.time()
#         print(streetMarker+" Done!, spent: %.2fs" % (end - start))
    
    with Pool(processes=6) as pool:
        slots = pool.starmap(_getResampledParkingSlotStatus, zip(df.StreetMarker.unique()))

    #print("Combining all StreetMarkers")
    #return pd.concat(slots, axis=1)
    return []


def resampleAllParkingLotsStatus(df,lotsInfoArray,interval,save = True):
    global _getResampledParkingSlotsStatus
    #dxx = pd.DataFrame()
    
    def _getResampledParkingSlotsStatus(lotInfo):
        streetMarkers = lotInfo["streetmarker"]
        lotId = lotInfo["lotid"]
        #dxx = pd.DataFrame()
        print("Resample and SUM for LotId:"+lotId+" StreetMarkers:"+ str(streetMarkers))
        pss = []
        for streetMarker in streetMarkers:
            start1 = time.time()
            dx1 = resampleParkingEvents(df,streetMarker,interval)
            dx1 = dx1.rename(columns={"status":streetMarker})
            #slots.append(dx1)
            end1 = time.time()
            pss.append(dx1)
            if save:
                dx1.to_csv("./datasets/MelbCity/slots/"+str(interval)+"m/"+streetMarker+".csv")
            print(streetMarker+" Done!, spent: %.2fs" % (end1 - start1))

        lotdf = pd.concat(pss, axis=1)
        lotdf[lotId] = lotdf[streetMarkers].sum(axis=1)
        lotdf.drop(streetMarkers, axis=1, inplace=True)
        if save:
            lotdf.to_csv("./datasets/MelbCity/lots/"+str(interval)+"m/"+lotId+".csv")
        else:
            return lotdf

        #return dxx[lotId]

#     slots = []
#     for streetMarker in df.StreetMarker.unique():
#         start = time.time()
#         print("Resample "+streetMarker)
#         dx1 = resampleParkingEvents(dcp,streetMarker,interval)
#         dx1 = dx1.rename(columns={"status":streetMarker})
#         slots.append(dx1)
#         end = time.time()
#         print(streetMarker+" Done!, spent: %.2fs" % (end - start))
    
    with Pool(processes=6) as pool:
        lots = pool.starmap(_getResampledParkingSlotsStatus, zip(lotsInfoArray))

    if save == False:
        return pd.concat(lots, axis=1)