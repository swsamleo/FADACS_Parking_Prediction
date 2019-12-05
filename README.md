# RMIT MParking Prediction

Parking slot prediction model for MParking project

### Models:
#### 1.DatasetConvertor.py
Convert dataset from data collection team. The input file should be the parking slot occupancy status of a proid(e.g. 1 min or 5 min, ...), which is handled by Bruce.
The dataset format:  
| StreetMarker | Day(MM/DD/YYYY) | 00:01 | 00:02 | 00:03 | 00:.. |  
|--------------|-----------------|-------|-------|-------|-------|  
| SOO1         | 01/01/2017      | 1     | 1     | 0     | 1     |  
| SOO1         | 01/02/2017      | 0     | 1     | 1     | 0     |  
| S001         | ../../2017      | 1     | 1     | 0     | 1     |  
| S001         | 12/30/2017      | 0     | 1     | 1     | 1     |  
| S002         | 01/01/2017      | 1     | 1     | 1     | 1     |  

The output file could be train/test datasets in a special date range, or the whole year occupancy status(a proid) of all parking slot.

Convert to the dataset of the whole year occupancy status
Output dataset format:  
| id    | data(whole year parking occupancy state, default hex) |  
|-------|------------------------------------------|  
| SOO1  | ff0ffa1ff0ffa1ff0ffa1ff0ffa1ff0ffa1...                               |   
| SOO2  | f000ffff000ffff000abbccf000ffff000fff....                              |  
| S00.. | f000ffff000ffff000ffff000ffff000ffff000....                              |  

The whole year occupancy status data use binrary to store each occupancy status, then convert to hex string to save in csv data, if use binrary file to save, the file size could be reduce a lot, from 4GB to 400MB. If the sample rate is 1 minute, each whole year occupancy status data of parking slot is around 64KB.

Convert to training/test dataset
training/test_x dataset:  
| Id   | t0 | t1 | t2 | t3 | ... | tn |  
|------|----|----|----|----|-----|----|  
| SOO1 | 1  | 1  | 0  | 1  | ... | 1  |  
| SOO1 | 0  | 1  | 1  | 0  | ... | 1  |  
| S001 | 1  | 1  | 0  | 1  | ... | 0  |  
| S001 | 0  | 1  | 1  | 1  | ... | 1  |  
| S002 | 1  | 1  | 1  | 1  | ... | 0  |  

The training/test_y dataset:  
| Id   | t |  
|------|---|   
| SOO1 | 1 |  
| SOO1 | 0 |  
| S001 | 1 |  
| S001 | 0 |  
| S002 | 1 |  

Demo code:
```python
import DatasetConvertor

#Convert to the dataset of the whole year occupancy status
DatasetConvertor.convertParkingData(input = "car_parking_2017_1mins_point.csv",
                     output="output.csv",
                     year="2017")

#convert to training/test dataset
DatasetConvertor.convertParkingData(input ="car_parking_2017_1mins_point.csv",
                                    output="output.csv",
                                    year="2017", #the year of input dataset
                                    convert = "extract", 
                                    start = "03/01/2017", # the start day of the range 2017-3-1:00:00:00
                                    end = "05/01/2017", # the end day of the range, 2017-5-1:00:00:00
                                    units = 30, # how many prediction inputs, t0,t1,t2...tn
                                    interval = 1 # here is 1, means the input dataset sample rate is 1 minute per occupancy status, if it's 5, means 5 minute per occupancy status.
                                    )

#the output file should be output_x.csv(training/test_x dataset) and output_y.csv(training/test_y dataset)
```

Known Issue in input file, current the file from data collection team(Bruce) is missing some parking data because there is not in orignal data. This module can avoid this problem.