from datetime import datetime
import json

import arrow
import pandas as pd


class FeatureApi(object):

    def __init__(self, weather_path, poi_path, rules_path,lots_rules_path, location_path):
        """Initial the Object.

        Load the DataFrame from the given three files path.

        Args:
            weather_path: String, "Weather_total_new.csv" file path.
            poi_path: String, "POI.csv" file path.
            rules_path: String, "parking_bay_restrictions.csv" file path.
            lots_rules_path: String, "Lot_rule.csv" file path.
            location_path: String, "StreetMarker_Location.csv" file path.
            lot_location_path: String, "StreetMarker_Lot.csv" file path.

        Returns:
            None. Initial the Object, load the data from the given paths.
        """
        self.df_weather = pd.read_csv(weather_path)
        self.df_poi = pd.read_csv(poi_path)
        self.df_rules = pd.read_csv(rules_path)
        self.df_lots_rules = pd.read_csv(lots_rules_path)
        self.df_lots_rules.LotId = self.df_lots_rules.LotId.astype(int,copy=False ).astype(str,copy=False )
        self.df_location = pd.read_csv(location_path)
        #self.df_lots_location = pd.read_csv(lot_location_path)

        # Define the maximum number of description in rules
        print("self.df_rules.shape[1]:"+str(self.df_rules.shape))
        self.num_of_descriptions = (self.df_rules.shape[1]-1)//9
        
        print("self.df_lots_rules.shape[1]:"+str(self.df_lots_rules.shape[1]))
        self.num_of_lot_rule_descriptions = (self.df_lots_rules.shape[1]-1)//9

        # Record the curr locaiton, 
        # To check whether the distance will be re-calculated.
        self.curr_location = None
        self.lotId = None
        self.streetMarker = None
        self.withIn = None
        # Define extreme weather condition:
        # 'Broken clouds', 'Clear', 'Cloudy', 'Cool', 'Drizzle',
        # 'Duststorm', 'Extremely hot', 'Fog', 'Hail', 'Haze', 'Heavy rain',
        # 'Light fog', 'Light rain', 'Lots of rain', 'Low clouds', 'More clouds than sun',
        # 'Mostly cloudy', 'Overcast', 'Partly cloudy', 'Partly sunny', 'Passing clouds',
        # 'Rain', 'Rain showers', 'Scattered clouds', 'Scattered showers',
        # 'Sprinkles', 'Sunny', 'Thundershowers', 'Thunderstorms'
        self.extreme_weathers_condition = [
            'Duststorm', 'Extremely hot', 'Fog', 'Hail', 'Haze', 
            'Heavy rain', 'Lots of rain', 'Thundershowers', 'Thunderstorms'
        ]

        # Define public holiday in 2017
        self.public_holiday_list = [
            '2017-01-01', '2017-01-02', '2017-01-26', '2017-03-13', '2017-04-14', 
            '2017-04-15', '2017-04-16', '2017-04-17', '2017-04-25', '2017-06-12', 
            '2017-09-29', '2017-11-07', '2017-12-25', '2017-12-26'
        ]

    def get_weather(self, date_time):
        """Get the weather features.

        Return a JSON object of weather features of a specific time, 
        which is nearest to the given time. These features include temperature, 
        power of wind and weather extreme weather is.

        Args:
            date_time: a datetime, format like "2017-03-02 13:38".

        Returns:
            a JSON object, including:
                - temperature: Float.
                - wind_power: Float. 
                - extreme_weather: Boolen, 0 or 1, 1 if it is extreme weather
            For example:

            '{"temperature": 34.0, "wind_power": 32.0, "extreme_weather": 0}'
        """
        
        month = arrow.get(date_time).month
        day = arrow.get(date_time).day
        time_string = arrow.get(date_time).strftime('%H:%M') # 24-hour format, "XX:XX"

        temp_weather = self.df_weather[(self.df_weather['Month']==month) & (self.df_weather['Day']==day)]
        print (temp_weather.head().to_string())
        temp_weather['Diff_time'] = temp_weather['Time_24h'].apply(
            lambda curr_time: self._diff_time(curr_time, time_string)
        )
        
        temp_weather = temp_weather[temp_weather['Diff_time']==temp_weather['Diff_time'].min()]

        return json.dumps({
            'temperature':temp_weather['Temp'].values[0],
            'wind_power': temp_weather['Wind'].values[0],
            'extreme_weather': int(temp_weather['Extreme_weather'].values[0])
        })
    
    def get_rule(self, date_time, on_public_holiday, streetmarker):
        """Get the rule features.

        Return a JSON object of rule features of a specific time and street marker id.
        These features include parking availability and duration.

        Args:
            date_time: a datetime, format like "2017-03-02 13:38".
            on_public_holiday: Boolen, default True. 
                If True, consider the public holiday, else ignore public holiday.
            streetmarker: String, street marker id.

        Returns:
            a JSON object, including:
                - availability: Int,0 or 1 indicating No Parking or Parking.
                - duration: Int, how long the vehicle can park.
            For example:

            '{"availability": 1, "duration": 60}'
        """
        
        # Init result = None
        result = None

        df = self.df_rules[self.df_rules['StreetMarker']==streetmarker]

        # If not found StreetMarker, return (0, 0)
        if df.empty:
            result = (0,0)
        else:
            result = self._check_availability_and_duration(df.squeeze(), date_time, on_public_holiday)
        
        return json.dumps({
            'availability': result[0],
            'duration': result[1],
        })
        

    def get_lot_rule(self, date_time, on_public_holiday, lot_id):
        """Get the rule features.

        Return a JSON object of rule features of a specific time and street marker id.
        These features include parking availability and duration.

        Args:
            date_time: a datetime, format like "2017-03-02 13:38".
            on_public_holiday: Boolen, default True. 
                If True, consider the public holiday, else ignore public holiday.
            lot_id: String, parking lot id.

        Returns:
            a JSON object, including:
                - availability: Int,0 or 1 indicating No Parking or Parking.
            For example:

            '{"availability": 1}'
        """

        df = self.df_lots_rules[self.df_lots_rules['LotId']==lot_id]
        #print("get_lot_rule df.shape:"+str(df.shape))
        # If not found LotId, return 0
        if df.empty:
            result = 0
        else:
            result = self._check_lotaround_availability_and_duration(df.squeeze(), date_time, on_public_holiday)
        
        return json.dumps({
            'availability': result,
        })

    
    def get_poi(self, location, r, date_time):
    #def get_poi(self, streetMarker,r, date_time):
        """Get the POI features.

        Return a JSON object of statistical POI features, including 
        - minimum distance in km between the given location and all POIs, 
        - the number of POIs within r km, 
        - the number of opened POIs at the time within r km, 
        - the mean distance in km between the given location and all POI within r km.

        Args:
            location: String, the coordinate format like "(lat, lon)" 
            r: Float, the radius interested, unit in km. 
            date_time: a datetime. format like "2017-03-02 13:38"

        Returns:
            a JSON object, including:
                - min_dis: Float, the minimum distance in km between given location and all POIs
                - num_of_poi: Int, the number of POIs within r km
                - num_of_open_poi: Int, the number of opened POIs at the time within r km
                - mean_dis: Float, the mean distance in km between 
                    the given location and all POIs within r km
            For example:
            
            '{"min_dis": 0.813142783966162, "num_of_poi": 190, 
            "num_of_open_poi": 50, "mean_dis": 2.0002688751885582}'
        """
        
        day_of_week = arrow.get(date_time).weekday() # from 0 to 6, Monday to Sunday
        time_string = arrow.get(date_time).strftime('%H:%M') # 24-hour format, "XX:XX"
        
        #############################
        # 1. Calculate the distance #
        #############################
        # If self.curr_location is None or changed, 
        # Then re-calculate the distance, and update the self.curr_location
                
#         if streetMarker != self.streetMarker:
#             location = str((self.df_location[self.df_location["StreetMarker"] == streetMarker])["Location"].values[0])

        if self.curr_location is None or self.curr_location!=location:

            self.df_poi['Distance'] = self.df_poi['Co-ordinates'].apply(
                lambda coordinate: self._cal_distance(location, coordinate)
            )

            # Update self.curr_location
            self.curr_location = location
            
        ############################
        # 2. Check within r or not #
        ############################
        self.df_poi['Withinr'] = self.df_poi['Distance'] < r
        self.withIn =  self.df_poi[self.df_poi['Withinr'] == True]

        ########################
        # 3. Check open or not #
        ########################
        self.withIn['Opened'] = self.withIn['Weekday_text'].apply(
            lambda string: self._check_opened_or_not(string, day_of_week, time_string)
        )
        
        ########################
        # 4. Return the result #
        ########################
        return json.dumps({
            'min_dis':self.df_poi['Distance'].min(),
            'num_of_poi': self.withIn.shape[0],
            'num_of_open_poi': self.withIn[self.withIn['Opened']==1].shape[0],
            'mean_dis': self.withIn['Distance'].mean(),
        })
    
    
    #def get_lot_poi(self, location, r, date_time):
    def get_lot_poi(self, lotId, r, date_time):
        """Get the POI features.

        Return a JSON object of statistical POI features, including 
        - minimum distance in km between the given location and all POIs, 
        - the number of POIs within r km, 
        - the number of opened POIs at the time within r km, 
        - the mean distance in km between the given location and all POI within r km.

        Args:
            location: String, the coordinate format like "(lat, lon)" 
            r: Float, the radius interested, unit in km. 
            date_time: a datetime. format like "2017-03-02 13:38"

        Returns:
            a JSON object, including:
                - min_dis: Float, the minimum distance in km between given location and all POIs
                - num_of_poi: Int, the number of POIs within r km
                - num_of_open_poi: Int, the number of opened POIs at the time within r km
                - mean_dis: Float, the mean distance in km between 
                    the given location and all POIs within r km
            For example:
            
            '{"min_dis": 0.813142783966162, "num_of_poi": 190, 
            "num_of_open_poi": 50, "mean_dis": 2.0002688751885582}'
        """
        
        day_of_week = arrow.get(date_time).weekday() # from 0 to 6, Monday to Sunday
        time_string = arrow.get(date_time).strftime('%H:%M') # 24-hour format, "XX:XX"
        
        if lotId != self.lotId:
            location = str((self.df_lots_location[self.df_lots_location["lotId"] == lotId])["Location"].values[0])

            #############################
            # 1. Calculate the distance #
            #############################
            # If self.curr_location is None or changed, 
            # Then re-calculate the distance, and update the self.curr_location
            if self.curr_location!=location:

                self.df_poi['Distance'] = self.df_poi['Co-ordinates'].apply(
                    lambda coordinate: self._cal_distance(location, coordinate)
                )
            
            # Update self.curr_location
            self.curr_location = location
        
        ############################
        # 2. Check within r or not #
        ############################
        self.df_poi['Withinr'] = self.df_poi['Distance'] < r
        
        ########################
        # 3. Check open or not #
        ########################
        self.df_poi['Opened'] = self.df_poi['Weekday_text'].apply(
            lambda string: self._check_opened_or_not(string, day_of_week, time_string)
        )
        
        ########################
        # 4. Return the result #
        ########################
        return json.dumps({
            'min_dis':self.df_poi['Distance'].min(),
            'num_of_poi': self.df_poi[self.df_poi['Withinr']].shape[0],
            'num_of_open_poi': self.df_poi[(self.df_poi['Withinr']) & (self.df_poi['Opened']==1)].shape[0],
            'mean_dis': self.df_poi[self.df_poi['Withinr']]['Distance'].mean(),
        })
    
    def get_extreme_weathers_condition(self):
        
        return self.extreme_weathers_condition
    
    def _check_lotaround_availability_and_duration(self, df, date_time, on_public_holiday=True):
        """Check the parking lot availability

        Return a int - availability, given a DataFrame, datetime and on_public_holiday.

        For example,            
        >>> df = pd.DataFrame(
                json.loads(
                    '{"LotId":{"1":20.0},"Description1":
                    {"1":"120.0 | 0.0 | 00:00:00 | 23:59:59 | Disable | 1.0 | 0.0 | Disable"},
                    "Duration1":{"1":120.0},"EffectiveOnPH1":{"1":0.0},"StartTime1":{"1":"00:00:00"},
                    "EndTime1":{"1":"23:59:59"},"Exemption1":{"1":"Disable"},"FromDay1":{"1":1.0},
                    "ToDay1":{"1":0.0},"TypeDesc1":{"1":"Disable"},"Description2":
                    {"1":"120.0 | 0.0 | 00:00:00 | 07:00:00 | Disable | 1.0 | 0.0 | Disable"},
                    "Duration2":{"1":120.0},"EffectiveOnPH2":{"1":0.0},"StartTime2":{"1":"00:00:00"},
                    "EndTime2":{"1":"07:00:00"},"Exemption2":{"1":"Disable"},"FromDay2":{"1":1.0},
                    "ToDay2":{"1":0.0},"TypeDesc2":{"1":"Disable"},"Description3":
                    {"1":"120.0 | 0.0 | 09:30:00 | 23:59:59 | Disable | 1.0 | 0.0 | Disable"},
                    "Duration3":{"1":120.0},"EffectiveOnPH3":{"1":0.0},"StartTime3":{"1":"09:30:00"},
                    "EndTime3":{"1":"23:59:59"},"Exemption3":{"1":"Disable"},"FromDay3":{"1":1.0},
                    "ToDay3":{"1":0.0},"TypeDesc3":{"1":"Disable"},"Description4":{"1":null},
                    "Duration4":{"1":null},"EffectiveOnPH4":{"1":null},"StartTime4":{"1":null},
                    "EndTime4":{"1":null},"Exemption4":{"1":null},"FromDay4":{"1":null},"ToDay4":{"1":null},
                    "TypeDesc4":{"1":null}}'
                ),
            ).squeeze()

        >>> api._check_availability_and_duration(df, "2017-01-01 13:01", True)
        1
        >>> api._check_availability_and_duration(df, "2017-01-02 13:01", True)
        1
        >>> api._check_availability_and_duration(df, "2017-01-03 13:01", True)
        0
        >>> api._check_availability_and_duration(df, "2017-03-13 06:30", True)
        1

        Args:
            df: A dataframe, contains 6 different rules/Descriptions. Each descriptions including 
                Duration, EffectiveOnPH, StartTime, EndTime, Exemption, FromDay, ToDay, TypeDesc.
                The value of FromDay and ToDay is from 0 to 6, indicating Sunday to Saturday.
            date_time: date_time: a datetime, format like "2017-03-02 13:38".
            on_public_holiday: Boolen, default True. 
                If True, consider the public holiday, else ignore public holiday.

        Returns:
            A Int of availability. The availability is 0 or 1, meaning No parking or Parking; 
            For example:

            0
        """
        #print(str(df))

        # convert date_time to arrow object
        date_time_arrow = arrow.get(date_time)

        YYMMDD = date_time_arrow.strftime('%Y-%m-%d') # YYYY-MM-DD
        DOW = date_time_arrow.weekday() # from 0 to 6, Monday to Sunday
        time = date_time_arrow.strftime('%H:%M') # 24-hour format, "XX:XX"

        # Set curr_YYMMDD = YYMMDD
        curr_YYMMDD = YYMMDD
        # Set curr_DOW = DOW + 1
        curr_DOW = DOW + 1
        # Set curr_time = time
        curr_time = time

        # Check curr_YYMMDD is holiday or not
        is_holiday = curr_YYMMDD in self.public_holiday_list

        # Default availability as True
        availability = 1
        # IF Description1 not emtpy
        if not pd.isna(df['Description1']):

            # Iterate each Descriptions
            for desc in range(1, self.num_of_lot_rule_descriptions+1):

                # IF No Description, THEN break the iterate.
                if pd.isna(df['Description'+str(desc)]):
                    break

                # ELSE (IF has Description), THEN
                else:
                    # Set Duration, EffectiveOnPH, StartTime, EndTime, Exemption, FromDay, ToDay, TypeDesc
                    # Duration = df['Duration'+str(desc)]
                    EffectiveOnPH = df['EffectiveOnPH'+str(desc)]
                    StartTime = df['StartTime'+str(desc)][:5] # "XX:XX:XX"
                    EndTime = df['EndTime'+str(desc)][:5] # "XX:XX:XX"
                    Exemption = df['Exemption'+str(desc)]
                    FromDay = 7 if df['FromDay'+str(desc)]==0 else df['FromDay'+str(desc)]
                    ToDay = 7 if df['ToDay'+str(desc)]==0 else df['ToDay'+str(desc)]
                    TypeDesc = df['TypeDesc'+str(desc)]

                    # IF FromDay < curr_DOW < ToDay and StartTime < curr_time < EndTime
                    if FromDay <= curr_DOW and curr_DOW <= ToDay and StartTime<= curr_time and curr_time < EndTime:

                        # IF ONLY on_public_holiday, is_holiday and not EffectiveOnPH, THEN IF NoParking THEN 1
                        # ELSE, THEN
                        if not (is_holiday and on_public_holiday and not EffectiveOnPH):
                            # IF Not Parking, THEN 
                            # No Parking, including 
                            #   - TypeDesc contains 'Disable' or 'No' or 'Loading' OR
                            #   - Exemption is not Nan.
                            if "Disable" in TypeDesc or "No" in TypeDesc or "Loading" in TypeDesc \
                                or (not pd.isna(Exemption)):
                                #print("availability set 0 here!")
                                availability = 0

                                # Break the loop
                                break
        #print("availability end:"+str(availability))
        return availability
    
    
    def _check_availability_and_duration(self, df, date_time, on_public_holiday=True):
        """Calculate the parking availability and parking duration

        Return a tuple availability and duration, given a DataFrame, day of week and time.

        For example,            
        >>> df = pd.DataFrame(
                json.loads(
                    '{"StreetMarker":"C988","Description1":"1P SUN 7:30-18:30","Duration1":60.0,
                    "EffectiveOnPH1":0.0,"StartTime1":"07:30:00","EndTime1":"18:30:00","Exemption1":null,
                    "FromDay1":0.0,"ToDay1":0.0,"TypeDesc1":"1P","Description2":"1P MTR M-SAT 7:30-19:30",
                    "Duration2":60.0,"EffectiveOnPH2":0.0,"StartTime2":"07:30:00","EndTime2":"19:30:00",
                    "Exemption2":null,"FromDay2":1.0,"ToDay2":6.0,"TypeDesc2":"1P","Description3":null,
                    "Duration3":null,"EffectiveOnPH3":null,"StartTime3":null,"EndTime3":null,"Exemption3":null,
                    "FromDay3":null,"ToDay3":null,"TypeDesc3":null,"Description4":null,"Duration4":null,
                    "EffectiveOnPH4":null,"StartTime4":null,"EndTime4":null,"Exemption4":null,"FromDay4":null,
                    "ToDay4":null,"TypeDesc4":null}'
                ),
                index=[0]
            ).squeeze()

        >>> _check_availability_and_duration(df, "2017-03-13 06:30", True) # PH, Monday
        (1, 1560)
        >>> _check_availability_and_duration(df, "2017-03-13 07:00", True) # PH, Monday
        (1, 1530)
        >>> _check_availability_and_duration(df, "2017-03-13 19:30", True) # PH, Monday
        (1, 780)
        >>> _check_availability_and_duration(df, "2017-03-14 08:50", True) # One day after PH, Tuesday
        (1, 60)
        >>> _check_availability_and_duration(df, "2017-03-14 20:00", True) # One day after PH, Tuesday
        (1, 12.5*60)
        >>> _check_availability_and_duration(df, "2017-03-15 21:14", True)
        (1, 11.5*60-14)
        >>> _check_availability_and_duration(df, "2017-03-21 08:23", True)
        (1, 60)
        >>> _check_availability_and_duration(df, "2017-03-21 09:30", True)
        (1, 60)

        Args:
            df: A dataframe, contains 6 different rules/Descriptions. Each descriptions including 
                Duration, EffectiveOnPH, StartTime, EndTime, Exemption, FromDay, ToDay, TypeDesc.
                The value of FromDay and ToDay is from 0 to 6, indicating Sunday to Saturday.
            date_time: date_time: a datetime, format like "2017-03-02 13:38".
            on_public_holiday: Boolen, default True. 
                If True, consider the public holiday, else ignore public holiday.

        Returns:
            A tuple of availability and duration. The availability is 0 or 1, meaning No parking or Parking; 
            The duration is a nonnegative integer. The unit of duration is minute. 
            If the duration is 0, this means No Parking, else the value is corresponding to the total duration.
            For example:

            (1, 750)
        """

        # convert date_time to arrow object
        date_time_arrow = arrow.get(date_time)

        YYMMDD = date_time_arrow.strftime('%Y-%m-%d') # YYYY-MM-DD
        DOW = date_time_arrow.weekday() # from 0 to 6, Monday to Sunday
        time = date_time_arrow.strftime('%H:%M') # 24-hour format, "XX:XX"

        # Set curr_YYMMDD = YYMMDD
        curr_YYMMDD = YYMMDD
        # Set curr_DOW = DOW + 1
        curr_DOW = DOW + 1
        # Set curr_time = time
        curr_time = time
        # Set while_loop = 1 # recording Continue or not
        while_loop = 1 # recording Continue or not

        # Inital duration_time=None
        duration_time=None
        
        # IF There is no rule, set the maximum duration_time as 24h
        if pd.isna(df['Description1']):
            
            duration_time = 24*60
        
        # ELSE (IF there has rules), THEN
        else:

            # While while_loop
            while while_loop:

                curr_upper_bound = None

                # Initial duration_change, to record 
                duration_change = 0

                # Iterate each Descriptions
                for desc in range(1, self.num_of_descriptions+1):

                    # IF No Description, THEN break the loop.
                    if pd.isna(df['Description'+str(desc)]):
                        break

                    # ELSE (IF has Description), THEN
                    else:
                        # Set Duration, EffectiveOnPH, StartTime, EndTime, Exemption, FromDay, ToDay, TypeDesc
                        Duration = df['Duration'+str(desc)]
                        EffectiveOnPH = df['EffectiveOnPH'+str(desc)]
                        StartTime = df['StartTime'+str(desc)][:5] # "XX:XX:XX"
                        #print("desc 0 >"+ str(desc))
                        EndTime = df['EndTime'+str(desc)][:5] # "XX:XX:XX"
                        Exemption = df['Exemption'+str(desc)]
                        FromDay = 7 if df['FromDay'+str(desc)]==0 else df['FromDay'+str(desc)]
                        ToDay = 7 if df['ToDay'+str(desc)]==0 else df['ToDay'+str(desc)]
                        TypeDesc = df['TypeDesc'+str(desc)]

                        if on_public_holiday:
                            # ONLY if is PH and Not Effect, THEN Break
                            if not (curr_YYMMDD in self.public_holiday_list and not EffectiveOnPH):
                                # IF FromDay<=curr_DOW<=ToDay, THEN
                                if FromDay <= curr_DOW and curr_DOW <= ToDay:

                                    # IF StartTime<=curr_time<EndTime, THEN
                                    if StartTime<= curr_time and curr_time < EndTime:

                                        duration_change = 1

                                        # IF Not Parking, THEN 
                                        # No Parking, including 
                                        #   - TypeDesc contains 'Disable' or 'No' or 'Loading' OR
                                        #   - Exemption is not Nan.
                                        if "Disable" in TypeDesc or "No" in TypeDesc or "Loading" in TypeDesc \
                                            or (not pd.isna(Exemption)):
                                            # Update while_loop=0
                                            while_loop = 0

                                            # IF duration_time is None, THEN set duration_time=0
                                            if duration_time is None:
                                                duration_time = 0

                                            # Break the loop
                                            break

                                        # ELSE (IF Parking), THEN 
                                        else:

                                            # Cal the ramaining_time = EndTime - curr_time
                                            ramaining_time = self._diff_time(EndTime, curr_time).seconds / 60

                                            # IF ramaining_time <= Duration, THEN 
                                            if ramaining_time <= Duration:
                                                # Update curr_time=EndTime
                                                curr_time = EndTime

                                                # IF duration_time is None, 
                                                # THEN set duration_time=ramaining_time,
                                                if duration_time is None:
                                                    duration_time = ramaining_time

                                                # ELSE (IF duration_time is NOT None), 
                                                # THEN set duration_time+=ramaining_time
                                                else:
                                                    duration_time += ramaining_time

                                                # Update curr_time=EndTime
                                                curr_time = EndTime

                                            # ELSE IF ramaining_time > Duration, THEN 
                                            else:
                                                # Update while_loop=0
                                                while_loop = 0

                                                # IF duration_time is None, THEN set duration_time=Duration,
                                                if duration_time is None:
                                                    duration_time = Duration

                                                # ELSE (IF duration_time is NOT None), 
                                                # THEN set duration_time+=Duration
                                                else:
                                                    duration_time += Duration

                                                # Break the loop
                                                break

                                    # ELSE (IF NOT StartTime<=curr_time<EndTime), THEN
                                    else:

                                        # Update the curr_upper_bound
                                        if StartTime > curr_time and \
                                            (curr_upper_bound is None or curr_upper_bound > StartTime):
                                                curr_upper_bound = StartTime

                                # ELSE (IF NOT FromDay<=curr_DOW<=ToDay), THEN Continue the loop
                                else:
                                    continue
                            else:
                                continue
                        else:
                            # IF FromDay<=curr_DOW<=ToDay, THEN
                            if FromDay <= curr_DOW and curr_DOW <= ToDay:

                                # IF StartTime<=curr_time<EndTime, THEN
                                if StartTime<= curr_time and curr_time < EndTime:

                                    duration_change = 1

                                    # IF Not Parking, THEN 
                                    # No Parking, including 
                                    #   - TypeDesc contains 'Disable' or 'No' or 'Loading' OR
                                    #   - Exemption is not Nan.
                                    if "Disable" in TypeDesc or "No" in TypeDesc or "Loading" in TypeDesc \
                                        or (not pd.isna(Exemption)):
                                        # Update while_loop=0
                                        while_loop = 0

                                        # IF duration_time is None, THEN set duration_time=0
                                        if duration_time is None:
                                            duration_time = 0

                                        # Break the loop
                                        break

                                    # ELSE (IF Parking), THEN 
                                    else:

                                        # Cal the ramaining_time = EndTime - curr_time
                                        ramaining_time = self._diff_time(EndTime, curr_time).seconds / 60

                                        # IF ramaining_time <= Duration, THEN 
                                        if ramaining_time <= Duration:
                                            # Update curr_time=EndTime
                                            curr_time = EndTime

                                            # IF duration_time is None, 
                                            # THEN set duration_time=ramaining_time,
                                            if duration_time is None:
                                                duration_time = ramaining_time

                                            # ELSE (IF duration_time is NOT None), 
                                            # THEN set duration_time+=ramaining_time
                                            else:
                                                duration_time += ramaining_time

                                            # Update curr_time=EndTime
                                            curr_time = EndTime

                                        # ELSE IF ramaining_time > Duration, THEN 
                                        else:
                                            # Update while_loop=0
                                            while_loop = 0

                                            # IF duration_time is None, THEN set duration_time=Duration,
                                            if duration_time is None:
                                                duration_time = Duration

                                            # ELSE (IF duration_time is NOT None), 
                                            # THEN set duration_time+=Duration
                                            else:
                                                duration_time += Duration

                                            # Break the loop
                                            break

                                # ELSE (IF NOT StartTime<=curr_time<EndTime), THEN
                                else:

                                    # Update the curr_upper_bound
                                    if StartTime > curr_time and \
                                        (curr_upper_bound is None or curr_upper_bound > StartTime):
                                            curr_upper_bound = StartTime

                            # ELSE (IF NOT FromDay<=curr_DOW<=ToDay), THEN Continue the loop
                            else:
                                continue

                # IF duration_change is 0:
                if not duration_change:

                    # IF curr_upper_bound exist, THEN
                    if curr_upper_bound:

                        # Cal the ramaining_time = curr_upper_bound - curr_time
                        ramaining_time = self._diff_time(curr_upper_bound, curr_time).seconds / 60

                        if duration_time is None:
                            duration_time = ramaining_time
                        else:
                            duration_time += ramaining_time

                        # update curr_time=curr_upper_bound[:5]
                        curr_time = curr_upper_bound

                    # ELSE (IF overnight)
                    else:

                        # Find the next restriction time and counting passing days
                        next_upper_bound = None

                        for offset in range(1, 14):
    
                            curr_DOW = 7 if (curr_DOW+1)%7==0 else (curr_DOW+1)%7
                            #print(date_time_arrow.format("DD-MM-YYYY"))
                            date_time_arrow = date_time_arrow.shift(days=+1)
                            #print(date_time_arrow.format("DD-MM-YYYY"))

                            curr_YYMMDD = date_time_arrow.strftime('%Y-%m-%d')

                            # Iterate each Descriptions
                            for desc in range(1, self.num_of_descriptions+1):
    
                                # IF No Description, THEN break the loop.
                                if pd.isna(df['Description'+str(desc)]):
                                    break

                                # ELSE (IF has Description), THEN
                                else:

                                    # Set EffectiveOnPH, StartTime, FromDay, ToDay
                                    EffectiveOnPH = df['EffectiveOnPH'+str(desc)]
                                    StartTime = df['StartTime'+str(desc)][:5] # "XX:XX:XX"
                                    FromDay = 7 if df['FromDay'+str(desc)]==0 else df['FromDay'+str(desc)]
                                    ToDay = 7 if df['ToDay'+str(desc)]==0 else df['ToDay'+str(desc)]

                                    if on_public_holiday:
                                        if not (curr_YYMMDD in self.public_holiday_list and not EffectiveOnPH):
                                            # IF FromDay<=curr_DOW<=ToDay, THEN
                                            if FromDay <= curr_DOW and curr_DOW <= ToDay:

                                                # Update the next_upper_bound
                                                if next_upper_bound is None or next_upper_bound > StartTime:
                                                        next_upper_bound = StartTime
                                        else:
                                            continue
                                    else:
                                        # IF FromDay<=curr_DOW<=ToDay, THEN
                                        if FromDay <= curr_DOW and curr_DOW <= ToDay:

                                            # Update the next_upper_bound
                                            if next_upper_bound is None or next_upper_bound > StartTime:
                                                    next_upper_bound = StartTime

                            if next_upper_bound:
                                break

                        # Update curr_time, duration_time
                        if duration_time is None:
                            duration_time = 0
                        duration_time += self._diff_time(curr_time, "23:59").seconds / 60 + 1
                        duration_time += (offset - 1) * 24 * 60
                        duration_time += self._diff_time(next_upper_bound, "00:00").seconds / 60

                        curr_time = next_upper_bound

        return (int(duration_time!=0), duration_time)

    def _cal_distance(self, coordinate1, coordinate2):
        """Calculate the distance between two coordinates.

        Return the distance in km, given two coordinates: coordinate1 and coordinate2. 

        For example,
            >>> coordinate1 = '(-37.7881645889621, 144.939277838304)'
            >>> coordinate2 = '(-37.78, 144.93)'
            >>> _cal_distance(coordinate1, coordinate2)
            1.22062371641592

        Args:
            coordinate1: A string, format like "(lat, lon)".
            coordinate2: A string, format like "(lat, lon)".

        Returns:
            A float, the distance between two coordinates in km.
        """
        from math import sin, cos, sqrt, atan2, radians
        
        # Split the coordinates
        lat1, lon1 = map(lambda x: radians(float(x.strip())), coordinate1[1:-1].split(','))
        lat2, lon2 = map(lambda x: radians(float(x.strip())), coordinate2[1:-1].split(','))

        R = 6373.0 # km
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c

        return distance

    
    def _convert_24h(self, T):
        """Convert the 12 hours to 24 hours

        Return a 24 hours format time string of the given time.

        For example,
            >>> _convert_24h("12:05 AM")
            "00:05"
            >>> _convert_24h("01:05 AM")
            "01:05"
            >>> _convert_24h("12:05 PM")
            "12:05"
            >>> _convert_24h("11:05 PM")
            "23:05"

        Args:
            T: A string, format like "XX:XX AM/PM".
            
        Returns:
            A string, the 24 hours format of the given time.
        """

        H, M = map(int, T[:-3].split(':'))

        # Checking if last two elements of time 
        # is AM and first two elements are 12 
        if ("AM" in T or "am" in T) and H == 12: 
            return "00:{:02d}".format(M)

        # remove the AM     
        elif "AM" in T or "am" in T: 
            return "{:02d}:{:02d}".format(H,M)

        # Checking if last two elements of time 
        # is PM and first two elements are 12    
        elif ("PM" in T or "pm" in T) and H == 12: 
            return "{:02d}:{:02d}".format(H,M)

        else: 
            # add 12 to hours and remove PM 
            return "{:02d}:{:02d}".format(H+12,M)
 
    def _check_opened_or_not_OLD(self, string, day_of_week, time_string):
        """Check the POI is opened or not.

        Return 0 or 1 indicated not opened or opened, 
        given the JSON string, check whether poi is opened on the given day_of_week and time

        For example,
            >>> string='{"open_now": true, "periods": [{
                "close": {"day": 0, "time": "0900"}, "open": {"day": 0, "time": "0800"}}, 
                {"close": {"day": 0, "time": "1030"}, "open": {"day": 0, "time": "0930"}}, 
                {"close": {"day": 0, "time": "1200"}, "open": {"day": 0, "time": "1100"}}, 
                {"close": {"day": 0, "time": "1900"}, "open": {"day": 0, "time": "1800"}}, 
                {"close": {"day": 1, "time": "1630"}, "open": {"day": 1, "time": "0930"}}, 
                {"close": {"day": 2, "time": "1630"}, "open": {"day": 2, "time": "0930"}}, 
                {"close": {"day": 3, "time": "1630"}, "open": {"day": 3, "time": "0930"}}, 
                {"close": {"day": 4, "time": "1630"}, "open": {"day": 4, "time": "0930"}}, 
                {"close": {"day": 5, "time": "1630"}, "open": {"day": 5, "time": "0930"}}], 
                "weekday_text": ["Monday: 9:30 AM \u2013 4:30 PM", 
                "Tuesday: 9:30 AM \u2013 4:30 PM", "Wednesday: 9:30 AM \u2013 4:30 PM", 
                "Thursday: 9:30 AM \u2013 4:30 PM", "Friday: 9:30 AM \u2013 4:30 PM", 
                "Saturday: Closed", "Sunday: 8:00 \u2013 9:00 AM, 9:30 \u2013 10:30 AM, 
                11:00 AM \u2013 12:00 PM, 6:00 \u2013 7:00 PM"]}'
            >>> _check_opened_or_not(string, 6, "10:50")
            0
            >>> _check_opened_or_not(string, 6, "10:00")
            1
            
            >>> string='{"open_now": true, "periods": [{
                "close": {"day": 1, "time": "2000"}, "open": {"day": 1, "time": "1100"}}, 
                {"close": {"day": 2, "time": "2000"}, "open": {"day": 2, "time": "1100"}}, 
                {"close": {"day": 3, "time": "2100"}, "open": {"day": 3, "time": "1100"}}, 
                {"close": {"day": 4, "time": "2100"}, "open": {"day": 4, "time": "1100"}}, 
                {"close": {"day": 6, "time": "0000"}, "open": {"day": 5, "time": "1100"}}], 
                "weekday_text": ["Monday: 11:00 AM \\u2013 8:00 PM", 
                "Tuesday: 11:00 AM \\u2013 8:00 PM", "Wednesday: 11:00 AM \\u2013 9:00 PM", 
                "Thursday: 11:00 AM \\u2013 9:00 PM", "Friday: 11:00 AM \\u2013 12:00 AM", 
                "Saturday: Closed", "Sunday: Closed"]}'
            >>> _check_opened_or_not(string, 6, "13:50")
            0
            >>> _check_opened_or_not(string, 1, "13:50")
            1

        Args:
            string: a json string, contain the opening hour information.
            day_of_week: A Int, from 0 to 6, indicated Monday to Sunday
            time_string: a 24-hours time string, format like: "XX:XX".

        Returns:
            A Int. 0 or 1, indicated not opened or open
        """


        # If string is not nan
        if not pd.isna(string):

            # A list of 7 days opening hour info from Mon to Sun
            # each info format like: "DOW: info"
            #   DOW could be: 
            #     "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
            #   info could be :
            #     - Closed
            #     - Open 24 hours
            #     - 11:00 AM - 12:00 AM
            open_info = json.loads(string)['weekday_text']

            # Define offset, the length of DOW
            offset = {0:6, 1:7, 2:9, 3:8, 4:6, 5:8, 6:6}

            open_hours = open_info[day_of_week][offset[day_of_week]+1:].strip()

            # If 24 hours open
            if open_hours == 'Open 24 hours':
                return 1

            # If closed
            elif open_hours == 'Closed':
                return 0

            # Check the open time periods
            else:
                
                status = 0
                
                # Some cases open_hours have different periods, connect with ','
                for time_periods in open_hours.split(','):
                    
                    open_time, close_time = map(lambda T: T.strip(), time_periods.strip().split('–'))
                    
                    # Check the open_time or close_time missing "AM" or "PM"
                    if open_time[-1]!='M':
                        open_time += close_time[-3:]
                    if close_time[-1]!='M':
                        close_time += open_time[-3:]

                    open_time = self._convert_24h(open_time)
                    close_time = self._convert_24h(close_time)

                    if close_time == "00:00":
                        close_time="24:00"

                    status += ((open_time<time_string) and (time_string<close_time))
                    
                return status


        # If string is nan, return 0 as not open
        else:
            return 0
        
    def _check_opened_or_not(self, string, day_of_week, time_string):
        """Check the POI is opened or not.

        Return 0 or 1 indicated not opened or opened, 
        given the JSON string, check whether poi is opened on the given day_of_week and time

        For example,
            >>> string='["Monday: 12:00 - 15:00, 17:00 - 21:00", 
            "Tuesday: 12:00 - 15:00, 17:00 - 21:00", "Wednesday: 12:00 - 15:00, 17:00 - 21:00", 
            "Thursday: 12:00 - 15:00, 17:00 - 21:00", "Friday: 12:00 - 15:00, 17:00 - 21:00", 
            "Saturday: 17:00 - 21:00", "Sunday: Closed"]'
            >>> _check_opened_or_not(string, 6, "10:50")
            0
            >>> _check_opened_or_not(string, 6, "10:00")
            0

            >>> string='{"open_now": true, "periods": [{
                "close": {"day": 1, "time": "2000"}, "open": {"day": 1, "time": "1100"}}, 
                {"close": {"day": 2, "time": "2000"}, "open": {"day": 2, "time": "1100"}}, 
                {"close": {"day": 3, "time": "2100"}, "open": {"day": 3, "time": "1100"}}, 
                {"close": {"day": 4, "time": "2100"}, "open": {"day": 4, "time": "1100"}}, 
                {"close": {"day": 6, "time": "0000"}, "open": {"day": 5, "time": "1100"}}], 
                "weekday_text": ["Monday: 11:00 AM \\u2013 8:00 PM", 
                "Tuesday: 11:00 AM \\u2013 8:00 PM", "Wednesday: 11:00 AM \\u2013 9:00 PM", 
                "Thursday: 11:00 AM \\u2013 9:00 PM", "Friday: 11:00 AM \\u2013 12:00 AM", 
                "Saturday: Closed", "Sunday: Closed"]}'
            >>> _check_opened_or_not(string, 6, "13:50")
            0
            >>> _check_opened_or_not(string, 1, "13:50")
            1

        Args:
            string: a json string, contain the opening hour information.
            day_of_week: A Int, from 0 to 6, indicated Monday to Sunday
            time_string: a 24-hours time string, format like: "XX:XX".

        Returns:
            A Int. 0 or 1, indicated not opened or open
        """


        # If string is not nan
        if not pd.isna(string):

            # A list of 7 days opening hour info from Mon to Sun
            # each info format like: "DOW: info"
            #   DOW could be: 
            #     "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
            #   info could be :
            #     - Closed
            #     - Open 24 hours
            #     - 11:00 - 12:00 
            open_info = json.loads(string)

            # Define offset, the length of DOW
            offset = {0:6, 1:7, 2:9, 3:8, 4:6, 5:8, 6:6}

            open_hours = open_info[day_of_week][offset[day_of_week]+1:].strip()

            # If 24 hours open
            if open_hours == 'Open 24 hours':
                return 1

            # If closed
            elif open_hours == 'Closed':
                return 0

            # Check the open time periods
            else:

                status = 0

                # Some cases open_hours have different periods, connect with ','
                for time_periods in open_hours.split(','):

                    open_time, close_time = map(lambda T: T.strip(), time_periods.strip().split('-'))

                    status += ((open_time<time_string) and (time_string<close_time))

                return status


        # If string is nan, return 0 as not open
        else:
            return 0

    def update_public_holiday(self, public_holiday):
        """Update public holiday list

        Update public holiday list, given a "YYYY-MM-DD" format public holiday list.

        For example,
            >>> public_holiday_list=[
                    '2017-01-01', '2017-01-02', '2017-01-26', '2017-03-13', '2017-04-14', 
                    '2017-04-15', '2017-04-16', '2017-04-17', '2017-04-25', '2017-06-12', 
                    '2017-09-29', '2017-11-07', '2017-12-25', '2017-12-26'
                ]
            >>> update_public_holiday(public_holiday_list)

        Args:
            public_holiday: a list, contain the public holiday

        Returns:
            None.
            Update self.public_holiday
        """

        self.public_holiday_list = public_holiday

    def update_extreme_weather(self, extreme_weathers_condition):

        # Update extreme_weathers_condition
        self.extreme_weathers_condition = extreme_weathers_condition

        # Update COL Extreme_weather
        self.df_weather['Extreme_weather'] = self.df_weather['Weather'].apply(
            lambda weather: self._check_extreme_weather(weather)
        )

    def _check_extreme_weather(self, candidate_weather):
        """Check weather the given weather is extreme weather.

        Return 0 or 1 indicated not extreme weather or extreme weather.

        For example,
            >>> _check_extreme_weather('Sprinkles. Duststorm.')
            1
            >>> _check_extreme_weather('Lots of rain. Broken clouds.')
            1
            >>> _check_extreme_weather('Clear.')
            0

        Args:
            candidate_weather: a string, contains the weather status.

        Returns:
            A Int. 0 or 1, indicated not extreme weather or extreme weather
        """

        for weather in self.extreme_weathers_condition:
            if weather in candidate_weather:
                return True
        return False
    
    def _diff_time(self, time1, time2):
        #print("diff time:"+str(time1)+" "+str(time2))
        """Calculate the diff between 2 given times.

        Return 0 or 1 indicated not extreme weather or extreme weather.

        For example,
            >>> _diff_time('15:35', '15:20')
            datetime.timedelta(seconds=900)
            >>> _diff_time('15:13', '14:38').seconds
            2100

        Args:
            candidate_weather: a string, contains the weather status.

        Returns:
            A Int. 0 or 1, indicated not extreme weather or extreme weather
        """
        
        if time1 < time2:
            time1, time2 = time2, time1
            
        return datetime.strptime(time1, '%H:%M') - datetime.strptime(time2, '%H:%M') 