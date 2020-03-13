import pandas as pd
import numpy as np
import arrow
from sklearn.preprocessing import MinMaxScaler

DATE_FORMAT = 'MM/DD/YYYY'

class FeatureConvertor():

    def __init__(self, weather_csv = "./datasets/MelbCity/features/weather.csv", 
                 slot_poi_csv = "./datasets/MelbCity/features/poi.csv",
                 lot_poi_csv = "./datasets/MelbCity/features/poi.lots.csv",
                 rules_nph_cvs = "./datasets/MelbCity/features/rule.nph.csv",
                 rules_ph_cvs = "./datasets/MelbCity/features/rule.ph.csv",
                 rules_lot_nph_cvs = "./datasets/MelbCity/features/rule.lots.nph.csv",
                 rules_lot_ph_cvs = "./datasets/MelbCity/features/rule.lots.ph.csv",
                ):
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
        self.slot_ruleFs = ["availability","duration"]
        
        self.df_weather = None
        self.df_slot_poi = None
        self.df_lot_poi = None
        self.df_rules_nhp = None
        self.df_rules_hp = None
        self.df_rules_lot_nhp = None
        self.df_rules_lot_hp = None
        
#         self.df_weather = pd.read_csv(weather_csv,index_col=0,parse_dates=True)
#         self.df_slot_poi = pd.read_csv(slot_poi_csv)
#         self.df_rules_nhp = pd.read_csv(rules_nph_cvs)
#         self.df_rules_hp = pd.read_csv(rules_ph_cvs)
    
    def getFeatuerList(self):
        return self.weatherFs + self.slot_poiFs + self.slot_ruleFs + ["Day","Hour","Minute","DayOfWeek","DayOfMonth","DayOfYear"]
    
    def getWeatherYearSerial(self, interval):
        if self.df_weather is None:
            mms = MinMaxScaler()
            self.df_weather = pd.read_csv(self.weather_csv,index_col=0,parse_dates=True)
            self.df_weather[self.weatherFs] = mms.fit_transform(self.df_weather[self.weatherFs])

        return self.df_weather.resample(str(interval)+"T").bfill()
    
    def getPOISerial(self,id,dateSerial,interval,type = "lot"):
        if type == "slot" and self.df_slot_poi is None:
            mms = MinMaxScaler()
            self.df_slot_poi = pd.read_csv(self.slot_poi_csv)
            self.df_slot_poi[self.slot_poiFs] = mms.fit_transform(self.df_slot_poi[self.slot_poiFs])
        elif type == "lot" and self.df_lot_poi is None:
            mms = MinMaxScaler()
            self.df_lot_poi = pd.read_csv(self.lot_poi_csv)
            self.df_lot_poi[self.slot_poiFs] = mms.fit_transform(self.df_lot_poi[self.slot_poiFs])
        
        df1 = None
        if type == "slot":
            df1 = self.df_slot_poi[self.df_slot_poi["StreetMarker"] == id]
        elif type == "lot":
            df1 = self.df_lot_poi[self.df_lot_poi["LotId"] == int(id)]

        df1['day'] = df1["datetime"].str[:10]
        df1['time'] = df1["datetime"].str[11:]
        #print(df1.head())
        df1["wd"] = df1.apply(lambda row: arrow.get(row["datetime"]).format("d"), axis=1)

        #del df1["datetime"]
        if type == "slot":
            del df1["StreetMarker"]
        elif type == "lot":
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
        #print("end "+end.format("YYYY-MM-DD HH:mm"))
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
    
    
    def getRuleSerial(self, id,dateSerial,interval,onHoliday=False,type = "lot"):
        if type == "slot":
            if self.df_rules_nhp is None:
                mms = MinMaxScaler()
                self.df_rules_nhp = pd.read_csv(self.rules_nph_cvs)
                self.df_rules_nhp[self.slot_ruleFs] = mms.fit_transform(self.df_rules_nhp[self.slot_ruleFs])
            if self.df_rules_hp is None:
                mms = MinMaxScaler()
                self.df_rules_hp = pd.read_csv(self.rules_ph_cvs)
                self.df_rules_hp[self.slot_ruleFs] = mms.fit_transform(self.df_rules_hp[self.slot_ruleFs])
        elif type == "lot":
            if self.df_rules_lot_nhp is None:
                mms = MinMaxScaler()
                self.df_rules_lot_nhp = pd.read_csv(self.rules_lot_nph_cvs)
                self.df_rules_lot_nhp[self.slot_ruleFs[:1]] = mms.fit_transform(self.df_rules_lot_nhp[self.slot_ruleFs[:1]])
            if self.df_rules_lot_hp is None:
                mms = MinMaxScaler()
                self.df_rules_lot_hp = pd.read_csv(self.rules_lot_ph_cvs)
                self.df_rules_lot_hp[self.slot_ruleFs[:1]] = mms.fit_transform(self.df_rules_lot_hp[self.slot_ruleFs[:1]])
        

        df1 = None
        #print(self.df_rules_lot_nhp.head())
        
        if type == "slot":
            if onHoliday:
                df1 = self.df_rules_hp[self.df_rules_hp["StreetMarker"] == id]
            else:
                df1 = self.df_rules_nhp[self.df_rules_nhp["StreetMarker"] == id]
        elif type == "lot":
            if onHoliday:
                df1 = self.df_rules_lot_hp[self.df_rules_lot_hp["LotId"] == int(id)]
            else:
                df1 = self.df_rules_lot_nhp[self.df_rules_lot_nhp["LotId"] == int(id)]


        df1['day'] = df1["datetime"].str[:10]
        df1['time'] = df1["datetime"].str[11:]
        #print(df1.head())
        df1["wd"] = df1.apply(lambda row: arrow.get(row["datetime"]).format("d"), axis=1)

        #del df1["datetime"]
        if type == "slot":
            del df1["StreetMarker"]
        elif type == "lot":
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
        #print("end "+end.format("YYYY-MM-DD HH:mm"))
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
        if type == "slot":
            outdf.loc[outdf["availability"] < 1.0, 'duration'] = 0.0
        outdf.loc[outdf["availability"] < 1.0, 'availability'] = 0.0
        return outdf
    
    
    def getFeatures(self,id,featureNames,startDay,endDay,interval,onHoliday=False,type = "lot"):
        START_DATE_OF_YEAR = "01/01/2017"
        END_DATE_OF_YEAR = "12/31/2017"
        
        startIndex = (arrow.get(startDay, DATE_FORMAT) - arrow.get(START_DATE_OF_YEAR, DATE_FORMAT)).days * 60 * 24 // interval
        endIndex = (arrow.get(endDay, DATE_FORMAT) - arrow.get(START_DATE_OF_YEAR, DATE_FORMAT)).days * 60 * 24 // interval +1
        
        ds = [ arrow.get(startDay, DATE_FORMAT).shift(days=+x).format("YYYY-MM-DD") for x in np.arange((arrow.get(endDay, DATE_FORMAT) - arrow.get(startDay, DATE_FORMAT)).days).tolist()]
        
        wDf = None
        spoiDf = None
        sruleDf = None
        
        #print("ds:"+str(ds))
        _type = type

        if any(item in self.weatherFs for item in featureNames):
            wDf = self.getWeatherYearSerial(interval)[startIndex:endIndex]
        
        if any(item in self.slot_poiFs for item in featureNames):
            spoiDf = self.getPOISerial(id,ds,interval,type = _type)
            
        if any(item in self.slot_ruleFs for item in featureNames):
            sruleDf = self.getRuleSerial(id,ds,interval,onHoliday,type = _type)
        
        #print(str(wDf.shape)+" "+str(spoiDf.shape)+" "+str(sruleDf.shape))
        
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
        if "WeekDay" in featureNames:
            outputDf["WeekDay"] = outputDf.apply(lambda row: int(arrow.get(row.name).format("d"))/7 ,axis=1)
        if "Month" in featureNames:
            outputDf["Month"] = outputDf.apply(lambda row: int(arrow.get(row.name).format("M"))/12 ,axis=1)
        if "DayOfYear" in featureNames:
            outputDf["DayOfYear"] = outputDf.apply(lambda row: int(arrow.get(row.name).format("DDD"))/365 ,axis=1)
            
        
        return outputDf[:-1]