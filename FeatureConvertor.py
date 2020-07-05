import pandas as pd
import numpy as np
import arrow
from sklearn.preprocessing import MinMaxScaler
from TLogger import *

logger = Logger("FeatureConvertor")

DATE_FORMAT = 'MM/DD/YYYY'

class FeatureConvertor():

    def __init__(self,
                 location = "MelbCity"
                ):
        
        weather_csv = "./datasets/MelbCity/features/weather.csv"
        slot_poi_csv = "./datasets/MelbCity/features/poi.csv"
        lot_poi_csv = "./datasets/MelbCity/features/poi.lots.csv"
        rules_nph_cvs = "./datasets/MelbCity/features/rule.nph.csv"
        rules_ph_cvs = "./datasets/MelbCity/features/rule.ph.csv"
        rules_lot_nph_cvs = "./datasets/MelbCity/features/rule.lots.nph.csv"
        rules_lot_ph_cvs = "./datasets/MelbCity/features/rule.lots.ph.csv"
        self.START_DATE = arrow.get("01/01/2017",DATE_FORMAT)
        
        if location == "Mornington":
            weather_csv = "./datasets/Mornington/features/weather.csv"
            slot_poi_csv = "./datasets/Mornington/features/poi.csv"
            lot_poi_csv = "./datasets/Mornington/features/poi.lots.csv"
            rules_nph_cvs = "./datasets/Mornington/features/rule.nph.csv"
            rules_ph_cvs = "./datasets/Mornington/features/rule.ph.csv"
            rules_lot_nph_cvs = "./datasets/Mornington/features/rule.lots.nph.csv"
            rules_lot_ph_cvs = "./datasets/Mornington/features/rule.lots.ph.csv"
                
        self.weather_csv = weather_csv
        self.slot_poi_csv = slot_poi_csv
        self.lot_poi_csv = lot_poi_csv
        self.rules_nph_cvs = rules_nph_cvs
        self.rules_ph_cvs = rules_ph_cvs
        self.rules_lot_nph_cvs = rules_lot_nph_cvs
        self.rules_lot_ph_cvs = rules_lot_ph_cvs

        self.weatherFs = ["Temp","Wind","Humidity","Barometer","Extreme_weather"]
        self.slot_poiFs = ['min_dis0.05', 'num_of_poi0.05',
               'num_of_open_poi0.05', 'mean_dis0.05', 'min_dis0.1', 'num_of_poi0.1',
               'num_of_open_poi0.1', 'mean_dis0.1', 'min_dis0.2', 'num_of_poi0.2',
               'num_of_open_poi0.2', 'mean_dis0.2', 'min_dis0.3', 'num_of_poi0.3',
               'num_of_open_poi0.3', 'mean_dis0.3', 'min_dis0.4', 'num_of_poi0.4',
               'num_of_open_poi0.4', 'mean_dis0.4', 'min_dis0.5', 'num_of_poi0.5',
               'num_of_open_poi0.5', 'mean_dis0.5', 'min_dis0.8', 'num_of_poi0.8',
               'num_of_open_poi0.8', 'mean_dis0.8', 'min_dis1.0', 'num_of_poi1.0',
               'num_of_open_poi1.0', 'mean_dis1.0']
        #self.slot_ruleFs = ["availability","duration"]
        self.slot_ruleFs = ["availability"]
        
        self.df_weather = None
        self.df_slot_poi = None
        self.df_lot_poi = None
        self.df_rules_nhp = None
        self.df_rules_hp = None
        self.df_rules_lot_nhp = None
        self.df_rules_lot_hp = None
        
        self.START_DATE = self.getWeatherDatasetStartDate()

#         self.df_weather = pd.read_csv(weather_csv,index_col=0,parse_dates=True)
#         self.df_slot_poi = pd.read_csv(slot_poi_csv)
#         self.df_rules_nhp = pd.read_csv(rules_nph_cvs)
#         self.df_rules_hp = pd.read_csv(rules_ph_cvs)
    
    def getFeatureList(self):
        return self.weatherFs + self.slot_poiFs + self.slot_ruleFs + ["Day","Hour","Minute","DayOfWeek","DayOfMonth","Month","DayOfYear"]
    

    def loadWeatherDataset(self):
        if self.df_weather is None:
            mms = MinMaxScaler()
            logger.debug("loadWeatherDataset "+self.weather_csv)
            self.df_weather = pd.read_csv(self.weather_csv,index_col=0,parse_dates=True)
            logger.debug(self.df_weather.isna().any())
            self.df_weather[self.weatherFs] = mms.fit_transform(self.df_weather[self.weatherFs])
            logger.debug(self.df_weather.isna().any())
    

    def getWeatherDatasetStartDate(self):
        self.loadWeatherDataset()
        return arrow.get(self.df_weather.index[0])


    def getWeatherYearSerial(self, interval):
        self.loadWeatherDataset()
        return self.df_weather.resample(str(interval)+"T").bfill()
    
    def getPOISerial(self,id,dateSerial,interval,paType = "lot"):
        if paType == "slot" and self.df_slot_poi is None:
            mms = MinMaxScaler()
            self.df_slot_poi = pd.read_csv(self.slot_poi_csv)
            self.df_slot_poi[self.slot_poiFs] = mms.fit_transform(self.df_slot_poi[self.slot_poiFs])
        elif paType == "lot" and self.df_lot_poi is None:
            mms = MinMaxScaler()
            self.df_lot_poi = pd.read_csv(self.lot_poi_csv)
            self.df_lot_poi[self.slot_poiFs] = mms.fit_transform(self.df_lot_poi[self.slot_poiFs])
        
        df1 = None
        if paType == "slot":
            if "StreetMarker" in self.df_slot_poi.columns.values:
                df1 = self.df_slot_poi[self.df_slot_poi["StreetMarker"] == id]
            else:
                df1 = self.df_slot_poi[self.df_slot_poi["id"] == id]
        elif paType == "lot":
            if "LotId" in self.df_lot_poi.columns.values:
                df1 = self.df_lot_poi[self.df_lot_poi["LotId"] == int(id)]
            else:
                df1 = self.df_lot_poi[self.df_lot_poi["id"] == int(id)]

        df1['day'] = df1["datetime"].str[:10]
        df1['time'] = df1["datetime"].str[11:]
        logger.debug(df1.head())
        df1["wd"] = df1.apply(lambda row: arrow.get(row["datetime"]).format("d"), axis=1)

        #del df1["datetime"]
        if paType == "slot":
            del df1["StreetMarker"]
        elif paType == "lot":
            del df1["LotId"]

        arr = []

        for t in dateSerial:
            dx = df1[df1["wd"] == arrow.get(t).format("d") ]
            dx["datetime"] = t+" "+dx["time"]
            del dx["day"]
            del dx["time"]
            del dx["wd"]
            arr.append(dx)

        #end = "2018-01-01 00:00"
        end = arrow.get(dateSerial[len(dateSerial) - 1] + " 23:59").shift(seconds =+ 60)
        #end = arrow.get(dateSerial[len(dateSerial) - 1] + " 23:59:00")
        logger.debug("end "+end.format("YYYY-MM-DD HH:mm"))
        dx = df1[(df1["wd"] == end.format("d")) & (df1["time"] == end.format("HH:mm"))]
        dx["datetime"] = end.format("YYYY-MM-DD HH:mm")
        del dx["day"]
        del dx["time"]
        del dx["wd"]
        arr.append(dx)

        outdf = pd.concat(arr)
        outdf.reset_index(drop=True, inplace=True)
        #return outdf

        outdf.index = pd.to_datetime(outdf.apply(lambda row: arrow.get(row["datetime"]).format('X'), axis=1), unit='s')
        del outdf["datetime"]
        return outdf.resample(str(interval)+'T').bfill()
    
    
    def getRuleSerial(self, id,dateSerial,interval,onHoliday=False,paType = "lot"):
        if paType == "slot":
            if self.df_rules_nhp is None:
                mms = MinMaxScaler()
                self.df_rules_nhp = pd.read_csv(self.rules_nph_cvs)
                self.df_rules_nhp[self.slot_ruleFs] = mms.fit_transform(self.df_rules_nhp[self.slot_ruleFs])
            if self.df_rules_hp is None:
                mms = MinMaxScaler()
                self.df_rules_hp = pd.read_csv(self.rules_ph_cvs)
                self.df_rules_hp[self.slot_ruleFs] = mms.fit_transform(self.df_rules_hp[self.slot_ruleFs])
        elif paType == "lot":
            if self.df_rules_lot_nhp is None:
                mms = MinMaxScaler()
                self.df_rules_lot_nhp = pd.read_csv(self.rules_lot_nph_cvs)
                self.df_rules_lot_nhp[self.slot_ruleFs[:1]] = mms.fit_transform(self.df_rules_lot_nhp[self.slot_ruleFs[:1]])
            if self.df_rules_lot_hp is None:
                mms = MinMaxScaler()
                self.df_rules_lot_hp = pd.read_csv(self.rules_lot_ph_cvs)
                self.df_rules_lot_hp[self.slot_ruleFs[:1]] = mms.fit_transform(self.df_rules_lot_hp[self.slot_ruleFs[:1]])
        

        df1 = None
        logger.debug(self.df_rules_lot_nhp.head())
        
        if paType == "slot":
            if onHoliday:
                df1 = self.df_rules_hp[self.df_rules_hp["StreetMarker"] == id]
            else:
                df1 = self.df_rules_nhp[self.df_rules_nhp["StreetMarker"] == id]
        elif paType == "lot":
            if onHoliday:
                df1 = self.df_rules_lot_hp[self.df_rules_lot_hp["LotId"] == int(id)]
            else:
                df1 = self.df_rules_lot_nhp[self.df_rules_lot_nhp["LotId"] == int(id)]


        df1['day'] = df1["datetime"].str[:10]
        df1['time'] = df1["datetime"].str[11:]
        logger.debug(df1.head())
        df1["wd"] = df1.apply(lambda row: arrow.get(row["datetime"]).format("d"), axis=1)

        #del df1["datetime"]
        if paType == "slot":
            del df1["StreetMarker"]
        elif paType == "lot":
            del df1["LotId"]

        hdf = None
        if onHoliday == True:
            hda = ['2017-01-01', '2017-01-02', '2017-01-26', '2017-03-13', '2017-04-14',
                    '2017-04-15', '2017-04-16', '2017-04-17', '2017-04-25',
                    '2017-06-12', '2017-09-29', '2017-11-07', '2017-12-25',
                    '2017-12-26']

            hdf = df1[df1["day"].isin(hda)]
            df1 = df1[~df1["day"].isin(hda)]

        arr = []

        for t in dateSerial:
            dx = None
            if onHoliday == True and t in hda:
                dx = hdf[hdf["day"] == t ]
            else:
                dx = df1[df1["wd"] == arrow.get(t).format("d") ]
                #dx["day"] = t
                dx["datetime"] = t+" "+dx["time"]
            del dx["day"]
            del dx["time"]
            del dx["wd"]
            arr.append(dx)

        #end = "2018-01-01 00:00"
        end = arrow.get(dateSerial[len(dateSerial) - 1] + " 23:59").shift(seconds =+ 60)
        #end = arrow.get(dateSerial[len(dateSerial) - 1] + " 23:59:00")
        logger.debug("end "+end.format("YYYY-MM-DD HH:mm"))
        dx = df1[(df1["wd"] == end.format("d")) & (df1["time"] == end.format("HH:mm"))]
        dx["datetime"] = end.format("YYYY-MM-DD HH:mm")
        del dx["day"]
        del dx["time"]
        del dx["wd"]
        arr.append(dx)

        outdf = pd.concat(arr)
        outdf.reset_index(drop=True, inplace=True)
        #return outdf

        outdf.index = pd.to_datetime(outdf.apply(lambda row: arrow.get(row["datetime"]).format('X'), axis=1), unit='s')
        outdf['availability'].astype(float).astype(bool)
        del outdf["datetime"]
        #return outdf.resample('T')

        outdf = outdf.resample(str(interval)+'T').interpolate(method='linear')
        if paType == "slot":
            outdf.loc[outdf["availability"] < 1.0, 'duration'] = 0.0
        outdf.loc[outdf["availability"] < 1.0, 'availability'] = 0.0
        return outdf
    
    
    def getFeatures(self,id,featureNames,startDay,endDay,interval,onHoliday=False,paType = "lot"):
        logger.debug(self.START_DATE)
        logger.debug(arrow.get(startDay, DATE_FORMAT))
        logger.debug(arrow.get(endDay, DATE_FORMAT))
        logger.debug("startDay "+startDay + " endDay "+endDay)
        logger.debug((arrow.get(startDay, DATE_FORMAT) - self.START_DATE).days * 60 * 24)
        logger.debug(interval)
        startIndex = (arrow.get(startDay, DATE_FORMAT) - self.START_DATE).days * 60 * 24 // interval
        endIndex = (arrow.get(endDay, DATE_FORMAT) - self.START_DATE).days * 60 * 24 // interval
        
        ds = [ arrow.get(startDay, DATE_FORMAT).shift(days=+x).format("YYYY-MM-DD") for x in np.arange((arrow.get(endDay, DATE_FORMAT) - arrow.get(startDay, DATE_FORMAT)).days).tolist()]
        
        wDf = None
        spoiDf = None
        sruleDf = None
        
        logger.debug("ds:"+str(ds))
        _type = paType

        if any(item in self.weatherFs for item in featureNames):
            logger.debug("startIndex:{} endIndex:{}".format(startIndex,endIndex))
            wDf = self.getWeatherYearSerial(interval)[startIndex:endIndex]
        
        if any(item in self.slot_poiFs for item in featureNames):
            spoiDf = self.getPOISerial(id,ds,interval,paType = _type)
            
        if any(item in self.slot_ruleFs for item in featureNames):
            sruleDf = self.getRuleSerial(id,ds,interval,onHoliday,paType = _type)
        
        logger.debug(str(wDf.shape)+" "+str(spoiDf.shape)+" "+str(sruleDf.shape))
        
        outputDf = pd.DataFrame(columns = featureNames)
        
        for fname in featureNames:
            if fname in self.weatherFs:
                outputDf[fname] = wDf[fname]
            if fname in self.slot_poiFs:
                outputDf[fname] = spoiDf[fname]
            if fname in self.slot_ruleFs:
                outputDf[fname] = sruleDf[fname]

        #outputDf.drop(outputDf[:-1])
        
        if "Day" in featureNames:
            outputDf["Day"] = outputDf.apply(lambda row: int(arrow.get(row.name).format("D"))/31 ,axis=1)
        if "Hour" in featureNames:
            outputDf["Hour"] = outputDf.apply(lambda row: int(arrow.get(row.name).format("H"))/24 ,axis=1)
        if "Minute" in featureNames:
            outputDf["Minute"] = outputDf.apply(lambda row: int(arrow.get(row.name).format("m"))/59 ,axis=1)
        if "DayOfWeek" in featureNames:
            outputDf["DayOfWeek"] = outputDf.apply(lambda row: int(arrow.get(row.name).format("d"))/7 ,axis=1)
        if "DayOfMonth" in featureNames:
            outputDf["DayOfMonth"] = outputDf.apply(lambda row: int(arrow.get(row.name).format("D"))/31 ,axis=1)
        if "Month" in featureNames:
            outputDf["Month"] = outputDf.apply(lambda row: int(arrow.get(row.name).format("M"))/12 ,axis=1)
        if "DayOfYear" in featureNames:
            outputDf["DayOfYear"] = outputDf.apply(lambda row: int(arrow.get(row.name).format("DDD"))/365 ,axis=1)
            
        logger.debug(outputDf.isna().any())
        #print(outputDf)
        return outputDf