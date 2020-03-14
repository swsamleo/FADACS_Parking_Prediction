import sys
sys.path.append("..")

import DatasetConvertor
import numpy as np

# DatasetConvertor.convertParkingData(input = "car_parking_2017_1mins_point.csv",
#                     output=opath+"training-1day.csv",
#                     year="2017",
#                     convert = "extract",
#                     start = "01/01/2017",
#                     end = "01/02/2017",
#                     units = 30,
#                     interval = 1
#                     )

# DatasetConvertor.convertParkingData(input = "car_parking_2017_1mins_point.csv",
#                     output=opath+"test-1day.csv",
#                     year="2017",
#                     convert = "extract",
#                     start = "01/02/2017",
#                     end = "01/03/2017",
#                     units = 30,
#                     interval = 1
#                     )

yOneDaySerial = np.arange(1, 31, 1).tolist() + \
                np.arange(31, 60, 2).tolist() + \
                np.arange(61, 120, 4).tolist() + \
                np.arange(121, 240, 8).tolist() + \
                np.arange(241, 480, 16).tolist() + \
                np.arange(481, 24*60+1, 32).tolist()

opath = "/Volumes/Downloads/pp/"

# DatasetConvertor.convertParkingData(input = "car_parking_2017_1mins_point.csv",
#                     output=opath+"training-2day-50slots.csv",
#                     year="2017",
#                     convert = "extract",
#                     start = "01/01/2017",
#                     end = "01/03/2017",
#                     units = 30,
#                     interval = 1,
#                     maxParkingSlots = 50,
#                     y_oneDaySerial = yOneDaySerial
#                     )

# DatasetConvertor.convertParkingData(input = "car_parking_2017_1mins_point.csv",
#                     output=opath+"test-2day-50slots.csv",
#                     year="2017",
#                     convert = "extract",
#                     start = "01/03/2017",
#                     end = "01/05/2017",
#                     units = 30,
#                     interval = 1,
#                     maxParkingSlots = 50,
#                     y_oneDaySerial = yOneDaySerial
#                     )

DatasetConvertor.convertParkingData(input = "car_parking_2017_1mins_point.csv",
                    output=opath+"training-2day-Allslots-b.csv",
                    year="2017",
                    convert = "extract",
                    start = "01/01/2017",
                    end = "01/03/2017",
                    units = 30,
                    interval = 1,
                    maxParkingSlots = 0,
                    y_oneDaySerial = yOneDaySerial
                    )

# DatasetConvertor.convertParkingData(input = "car_parking_2017_1mins_point.csv",
#                     output=opath+"test-2day-Allslots.csv",
#                     year="2017",
#                     convert = "extract",
#                     start = "01/03/2017",
#                     end = "01/05/2017",
#                     units = 30,
#                     interval = 1,
#                     maxParkingSlots = 0,
#                     y_oneDaySerial = yOneDaySerial
#                     )

# DatasetConvertor.convertParkingData(input = "car_parking_2017_1mins_point.csv",
#                     output=opath+"training-30day-Allslots.csv",
#                     year="2017",
#                     convert = "extract",
#                     start = "01/01/2017",
#                     end = "01/31/2017",
#                     units = 30,
#                     interval = 1,
#                     maxParkingSlots = 0,
#                     y_oneDaySerial = yOneDaySerial
#                     )

# DatasetConvertor.convertParkingData(input = "car_parking_2017_1mins_point.csv",
#                     output=opath+"test-30day-Allslots.csv",
#                     year="2017",
#                     convert = "extract",
#                     start = "02/01/2017",
#                     end = "02/03/2017",
#                     units = 30,
#                     interval = 1,
#                     maxParkingSlots = 0,
#                     y_oneDaySerial = yOneDaySerial
#                     )



# DatasetConvertor.convertParkingData(input = "car_parking_2017_1mins_point.csv",
#                     output=opath+"output-all.csv",
#                     year="2017")
