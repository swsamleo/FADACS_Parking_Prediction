import DatasetConvertor

DatasetConvertor.convertParkingData(input = "car_parking_2017_1mins_point.csv",
                    output="output.csv",
                    year="2017",
                    convert = "extract",
                    start = "03/01/2017",
                    end = "05/01/2017",
                    units = 30,
                    interval = 1
                    )


# DatasetConvertor.convertParkingData(input = "car_parking_2017_1mins_point.csv",
#                     output="output-1.csv",
#                     year="2017")
