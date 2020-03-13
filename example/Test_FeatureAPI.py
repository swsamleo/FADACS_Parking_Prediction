import sys
sys.path.append("..")

from FeatureAPI import *

api = FeatureApi(
    weather_path="./Weather_total_new.csv",
    poi_path="./POI.csv",
    rules_path="./parking_bay_restrictions.csv"
)

# Test _cal_distance function
location1='(-37.7881645889621, 144.939277838304)'
location2='(-37.78, 144.93)'
assert api._cal_distance(location1, location2)==1.22062371641592


# Test _convert_24h function
assert api._convert_24h("12:05 AM")=="00:05"
assert api._convert_24h("01:05 AM")=="01:05"
assert api._convert_24h("1:05 am")=="01:05"
assert api._convert_24h("12:05 PM")=="12:05"
assert api._convert_24h("12:5 pm")=="12:05"
assert api._convert_24h("11:05 PM")=="23:05"


# Test _check_opened_or_not function
string='{"open_now": true, "periods": [{"open": {"day": 0, "time": "0000"}}], "weekday_text": ["Monday: Open 24 hours", "Tuesday: Open 24 hours", "Wednesday: Open 24 hours", "Thursday: Open 24 hours", "Friday: Open 24 hours", "Saturday: Open 24 hours", "Sunday: Open 24 hours"]}'
assert api._check_opened_or_not(string, 0, "13:50")==1

string='{"open_now": true, "periods": [{"close": {"day": 1, "time": "2000"}, "open": {"day": 1, "time": "1100"}}, {"close": {"day": 2, "time": "2000"}, "open": {"day": 2, "time": "1100"}}, {"close": {"day": 3, "time": "2100"}, "open": {"day": 3, "time": "1100"}}, {"close": {"day": 4, "time": "2100"}, "open": {"day": 4, "time": "1100"}}, {"close": {"day": 6, "time": "0000"}, "open": {"day": 5, "time": "1100"}}], "weekday_text": ["Monday: 11:00 AM \\u2013 8:00 PM", "Tuesday: 11:00 AM \\u2013 8:00 PM", "Wednesday: 11:00 AM \\u2013 9:00 PM", "Thursday: 11:00 AM \\u2013 9:00 PM", "Friday: 11:00 AM \\u2013 12:00 AM", "Saturday: Closed", "Sunday: Closed"]}'
assert api._check_opened_or_not(string, 6, "13:50")==0
assert api._check_opened_or_not(string, 1, "13:50")==1
assert api._check_opened_or_not(string, 2, "13:50")==1

string='{"open_now": true, "periods": [{"close": {"day": 0, "time": "0900"}, "open": {"day": 0, "time": "0800"}}, {"close": {"day": 0, "time": "1030"}, "open": {"day": 0, "time": "0930"}}, {"close": {"day": 0, "time": "1200"}, "open": {"day": 0, "time": "1100"}}, {"close": {"day": 0, "time": "1900"}, "open": {"day": 0, "time": "1800"}}, {"close": {"day": 1, "time": "1630"}, "open": {"day": 1, "time": "0930"}}, {"close": {"day": 2, "time": "1630"}, "open": {"day": 2, "time": "0930"}}, {"close": {"day": 3, "time": "1630"}, "open": {"day": 3, "time": "0930"}}, {"close": {"day": 4, "time": "1630"}, "open": {"day": 4, "time": "0930"}}, {"close": {"day": 5, "time": "1630"}, "open": {"day": 5, "time": "0930"}}], "weekday_text": ["Monday: 9:30 AM \u2013 4:30 PM", "Tuesday: 9:30 AM \u2013 4:30 PM", "Wednesday: 9:30 AM \u2013 4:30 PM", "Thursday: 9:30 AM \u2013 4:30 PM", "Friday: 9:30 AM \u2013 4:30 PM", "Saturday: Closed", "Sunday: 8:00 \u2013 9:00 AM, 9:30 \u2013 10:30 AM, 11:00 AM \u2013 12:00 PM, 6:00 \u2013 7:00 PM"]}'
assert api._check_opened_or_not(string, 6, "10:50")==0
assert api._check_opened_or_not(string, 6, "10:00")==1


# Test _check_extreme_weather function
assert api._check_extreme_weather('Sprinkles. Duststorm.')==1
assert api._check_extreme_weather('Lots of rain. Broken clouds.')==1
assert api._check_extreme_weather('Clear.')==0


# Test _diff_time function
assert api._diff_time('15:35', '15:20').seconds == 15*60
assert api._diff_time('15:13', '14:38').seconds == 35*60

# Test _check_availability_and_duration function
df = pd.read_json(
    '{"BayID":2496,"DeviceID":25193,"Description1":"1P MTR M-SAT 7:30-18:30","Description2":"2P MTR M-SAT 18.30 - 20.30","Description3":"1P SUN 7:30-18:30","Description4":null,"Description5":null,"Description6":null,"DisabilityExt1":120,"DisabilityExt2":240.0,"DisabilityExt3":120.0,"DisabilityExt4":null,"DisabilityExt5":null,"DisabilityExt6":null,"Duration1":60,"Duration2":120.0,"Duration3":60.0,"Duration4":null,"Duration5":null,"Duration6":null,"EffectiveOnPH1":0,"EffectiveOnPH2":0.0,"EffectiveOnPH3":0.0,"EffectiveOnPH4":null,"EffectiveOnPH5":null,"EffectiveOnPH6":null,"EndTime1":"18:30:00","EndTime2":"20:30:00","EndTime3":"18:30:00","EndTime4":null,"EndTime5":null,"EndTime6":null,"Exemption1":null,"Exemption2":null,"Exemption3":null,"Exemption4":null,"Exemption5":null,"Exemption6":null,"FromDay1":1,"FromDay2":1.0,"FromDay3":0.0,"FromDay4":null,"FromDay5":null,"FromDay6":null,"StartTime1":"07:30:00","StartTime2":"18:30:00","StartTime3":"07:30:00","StartTime4":null,"StartTime5":null,"StartTime6":null,"ToDay1":6,"ToDay2":6.0,"ToDay3":0.0,"ToDay4":null,"ToDay5":null,"ToDay6":null,"TypeDesc1":"1P Meter","TypeDesc2":"2P","TypeDesc3":"1P","TypeDesc4":null,"TypeDesc5":null,"TypeDesc6":null}',
    typ='series'
)

assert api._check_availability_and_duration(df, 0, "07:00") == (1, 90)
assert api._check_availability_and_duration(df, 0, "07:30") == (1, 60)

assert api._check_availability_and_duration(df, 0, "18:30") == (1, 14*60)
assert api._check_availability_and_duration(df, 1, "20:00") == (1, 12.5*60)
assert api._check_availability_and_duration(df, 2, "21:14") == (1, 11.5*60-14)

assert api._check_availability_and_duration(df, 6, "10:23") == (1, 60)
assert api._check_availability_and_duration(df, 6, "17:30") == (1, 15*60)
assert api._check_availability_and_duration(df, 6, "21:14") == (1, 11.5*60-14)

df = pd.read_json(
    '{"BayID":2171,"DeviceID":27586,"Description1":"No Stop M-F 7.00-09.30","Description2":"1P M-F 9:30-16:00","Description3":"S\\/ No Stop M-F  16:00-18:30","Description4":"2P MTR M-SAT 18.30 - 20.30","Description5":"1P MTR SAT 7.30-6.30PM","Description6":"2P SUN 7:30-18:30","DisabilityExt1":0,"DisabilityExt2":120.0,"DisabilityExt3":0.0,"DisabilityExt4":240.0,"DisabilityExt5":120.0,"DisabilityExt6":240.0,"Duration1":1,"Duration2":60.0,"Duration3":1.0,"Duration4":120.0,"Duration5":60.0,"Duration6":120.0,"EffectiveOnPH1":0,"EffectiveOnPH2":0.0,"EffectiveOnPH3":0.0,"EffectiveOnPH4":0.0,"EffectiveOnPH5":0.0,"EffectiveOnPH6":0.0,"EndTime1":"09:30:00","EndTime2":"16:00:00","EndTime3":"18:30:00","EndTime4":"20:30:00","EndTime5":"18:30:00","EndTime6":"18:30:00","Exemption1":null,"Exemption2":null,"Exemption3":null,"Exemption4":null,"Exemption5":null,"Exemption6":null,"FromDay1":1,"FromDay2":1.0,"FromDay3":1.0,"FromDay4":1.0,"FromDay5":6.0,"FromDay6":0.0,"StartTime1":"07:00:00","StartTime2":"09:30:00","StartTime3":"16:00:00","StartTime4":"18:30:00","StartTime5":"07:30:00","StartTime6":"07:30:00","ToDay1":5,"ToDay2":5.0,"ToDay3":5.0,"ToDay4":6.0,"ToDay5":6.0,"ToDay6":0.0,"TypeDesc1":"S\\/ (No Stopping)","TypeDesc2":"1P","TypeDesc3":"S\\/ (No Stopping)","TypeDesc4":"2P","TypeDesc5":"1P Meter","TypeDesc6":"2P"}',
    typ='series'
)
assert api._check_availability_and_duration(df, 0, "06:30") == (1, 30)
assert api._check_availability_and_duration(df, 0, "07:00") == (0, 0)
assert api._check_availability_and_duration(df, 1, "08:23") == (0, 0)
assert api._check_availability_and_duration(df, 1, "09:30") == (1, 60)
assert api._check_availability_and_duration(df, 2, "15:23") == (1, 37)
assert api._check_availability_and_duration(df, 2, "16:00") == (0, 0)
assert api._check_availability_and_duration(df, 3, "17:00") == (0, 0)
assert api._check_availability_and_duration(df, 3, "18:30") == (1, 12.5*60)
assert api._check_availability_and_duration(df, 3, "20:00") == (1, 11*60)
assert api._check_availability_and_duration(df, 4, "20:04") == (1, 12.5*60-4)
assert api._check_availability_and_duration(df, 4, "21:13") == (1, 11.5*60-13)

assert api._check_availability_and_duration(df, 5, "06:55") == (1, 60+35)
assert api._check_availability_and_duration(df, 5, "07:30") == (1, 60)
assert api._check_availability_and_duration(df, 5, "09:23") == (1, 60)
assert api._check_availability_and_duration(df, 5, "18:00") == (1, 15.5*60)
assert api._check_availability_and_duration(df, 5, "20:00") == (1, 13.5*60)
assert api._check_availability_and_duration(df, 5, "21:00") == (1, 12.5*60)

assert api._check_availability_and_duration(df, 6, "06:55") == (1, 2*60+35)
assert api._check_availability_and_duration(df, 6, "07:30") == (1, 2*60)
assert api._check_availability_and_duration(df, 6, "18:00") == (1, 13*60)
assert api._check_availability_and_duration(df, 6, "20:00") == (1, 11*60)

df = pd.read_json(
    '{"Description1":"No Stop M-F 7.00-09.30","Description2":"1P M-F 10:30-16:00","Description3":"S\\/ No Stop M-F  16:00-18:30","Description4":"2P MTR M-SAT 18.30 - 20.30","Description5":"1P MTR SAT 7.30-6.30PM","Description6":"2P SUN 7:30-18:30","DisabilityExt1":0,"DisabilityExt2":120.0,"DisabilityExt3":0.0,"DisabilityExt4":240.0,"DisabilityExt5":120.0,"DisabilityExt6":240.0,"Duration1":1,"Duration2":60.0,"Duration3":1.0,"Duration4":120.0,"Duration5":60.0,"Duration6":120.0,"EffectiveOnPH1":0,"EffectiveOnPH2":0.0,"EffectiveOnPH3":0.0,"EffectiveOnPH4":0.0,"EffectiveOnPH5":0.0,"EffectiveOnPH6":0.0,"EndTime1":"09:30:00","EndTime2":"16:00:00","EndTime3":"18:30:00","EndTime4":"20:30:00","EndTime5":"18:30:00","EndTime6":"18:30:00","Exemption1":null,"Exemption2":null,"Exemption3":null,"Exemption4":null,"Exemption5":null,"Exemption6":null,"FromDay1":1,"FromDay2":1.0,"FromDay3":1.0,"FromDay4":1.0,"FromDay5":6.0,"FromDay6":0.0,"StartTime1":"07:00:00","StartTime2":"10:30:00","StartTime3":"16:00:00","StartTime4":"18:30:00","StartTime5":"07:30:00","StartTime6":"07:30:00","ToDay1":5,"ToDay2":5.0,"ToDay3":5.0,"ToDay4":6.0,"ToDay5":6.0,"ToDay6":0.0,"TypeDesc1":"S\\/ (No Stopping)","TypeDesc2":"1P","TypeDesc3":"S\\/ (No Stopping)","TypeDesc4":"2P","TypeDesc5":"1P Meter","TypeDesc6":"2P"}',
    typ='series'
)

assert api._check_availability_and_duration(df, 0, "06:30") == (1, 30)
assert api._check_availability_and_duration(df, 0, "07:00") == (0, 0)
assert api._check_availability_and_duration(df, 1, "08:23") == (0, 0)
assert api._check_availability_and_duration(df, 1, "09:30") == (1, 2*60)
assert api._check_availability_and_duration(df, 1, "10:00") == (1, 1.5*60)
assert api._check_availability_and_duration(df, 2, "15:23") == (1, 37)
assert api._check_availability_and_duration(df, 2, "16:00") == (0, 0)
assert api._check_availability_and_duration(df, 3, "17:00") == (0, 0)
assert api._check_availability_and_duration(df, 3, "18:30") == (1, 12.5*60)
assert api._check_availability_and_duration(df, 3, "20:00") == (1, 11*60)
assert api._check_availability_and_duration(df, 4, "20:04") == (1, 12.5*60-4)
assert api._check_availability_and_duration(df, 4, "21:13") == (1, 11.5*60-13)

assert api._check_availability_and_duration(df, 5, "06:55") == (1, 60+35)
assert api._check_availability_and_duration(df, 5, "07:30") == (1, 60)
assert api._check_availability_and_duration(df, 5, "09:23") == (1, 60)
assert api._check_availability_and_duration(df, 5, "18:00") == (1, 15.5*60)
assert api._check_availability_and_duration(df, 5, "20:00") == (1, 13.5*60)
assert api._check_availability_and_duration(df, 5, "21:00") == (1, 12.5*60)

assert api._check_availability_and_duration(df, 6, "06:55") == (1, 2*60+35)
assert api._check_availability_and_duration(df, 6, "07:30") == (1, 2*60)
assert api._check_availability_and_duration(df, 6, "18:00") == (1, 13*60)
assert api._check_availability_and_duration(df, 6, "20:00") == (1, 11*60)