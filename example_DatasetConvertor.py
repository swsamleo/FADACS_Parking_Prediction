import DatasetConvertor
import numpy as np

# DatasetConvertor.convertParkingData(input = "car_parking_2017_1mins_point.csv",
#                     output="training-1day.csv",
#                     year="2017",
#                     convert = "extract",
#                     start = "01/01/2017",
#                     end = "01/02/2017",
#                     units = 30,
#                     interval = 1
#                     )

# DatasetConvertor.convertParkingData(input = "car_parking_2017_1mins_point.csv",
#                     output="test-1day.csv",
#                     year="2017",
#                     convert = "extract",
#                     start = "01/02/2017",
#                     end = "01/03/2017",
#                     units = 30,
#                     interval = 1
#                     )


DatasetConvertor.convertParkingData(input = "car_parking_2017_1mins_point.csv",
                    output="training-2day-50slots.csv",
                    year="2017",
                    convert = "extract",
                    start = "01/01/2017",
                    end = "01/03/2017",
                    units = 30,
                    interval = 1,
                    maxParkingSlots = 50,
                    y_oneDaySerial = np.arange(1, 31, 1).tolist() + \
                                    np.arange(31, 60, 2).tolist() + \
                                    np.arange(61, 120, 4).tolist() + \
                                    np.arange(121, 240, 8).tolist() + \
                                    np.arange(241, 480, 16).tolist() + \
                                    np.arange(481, 24*60+1, 32).tolist()
                    )

# DatasetConvertor.convertParkingData(input = "car_parking_2017_1mins_point.csv",
#                     output="output-all.csv",
#                     year="2017")
