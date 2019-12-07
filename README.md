# RMIT MParking Prediction

Parking slot prediction model for MParking project

### Models:
#### 1.DatasetConvertor.py
Convert dataset from data collection team. The input file should be the parking slot occupancy status of a proid(e.g. 1 min or 5 min, ...), which is handled by Bruce.
The dataset format:  


| StreetMarker   | Day(MM/DD/YYYY)   | 00:01  | 00:02   | 00:03  | 00:..   |  
| -------------- | ----------------- |------- | ------- | -------| ------- |  
| SOO1           | 01/01/2017        | 1      | 1       | 0      | 1       |  
| SOO1           | 01/02/2017        | 0      | 1       | 1      | 0       |  
| S001           | ../../2017        | 1      | 1       | 0      | 1       |  
| S001           | 12/30/2017        | 0      | 1       | 1      | 1       |  
| S002           | 01/01/2017        | 1      | 1       | 1      | 1       |  

The output file could be train/test datasets in a special date range, or the whole year occupancy status(a proid) of all parking slot.

Convert to the dataset of the whole year occupancy status
Output dataset format:  


| id      | data(whole year parking occupancy state, default hex) |  
| ------- | ------------------------------------------ |  
| SOO1  | ff0ffa1ff0ffa1ff0ffa1ff0ffa1ff0ffa1...       |   
| SOO2  | f000ffff000ffff000abbccf000ffff000fff....    |  
| S00.. | f000ffff000ffff000ffff000ffff000ffff000....  |  



The whole year occupancy status data use binrary to store each occupancy status, then convert to hex string to save in csv data, if use binrary file to save, the file size could be reduce a lot, from 4GB to 400MB. If the sample rate is 1 minute, each whole year occupancy status data of parking slot is around 64KB.

Here is the training/test_x/y dataset structure:


|  	|         	|         	|         	|training/test_x|    	| 	\| 	|    	|    	|      	|training/test_y |   
| -----------------	| ---------	| ---------	| ---------	| -------	| ----	| ---	| -----------------	| ----	| ----	| ------	| ----	|   
| Id              	| t-(n-1) 	| t-(n-2) 	| t-(n-3) 	| t-... 	| t0 	| \| 	| t1              	| t2 	| t3 	| t... 	| tX 	|   
| SOO1            	| 0       	| 0       	| 0       	| 1     	| 0  	| \| 	| 1               	| 1  	| 0  	| 0    	| 0  	|   
| SOO1            	| 1       	| 0       	| 0       	| 1     	| 1  	| \| 	| 1               	| 1  	| 1  	| 0    	| 0  	|   
| SOO1            	| 0       	| 0       	| 0       	| 1     	| 0  	| \| 	| 1               	| 1  	| 0  	| 0    	| 1  	|   
| SOO2            	| 0       	| 0       	| 0       	| 1     	| 0  	| \| 	| 1               	| 1  	| 0  	| 0    	| 1  	|   


The training/test_y with t serial(Exponentially sparse) dataset:


|                      | 0\-30 mins        | 31\-60 mins          | 61\-120 mins          | 121\-240 mins            | 241\-480 mins            | 481\-24\*60               |    
| ---------------------- | ------------------- | ---------------------- | ----------------------- | -------------------------- | -------------------------- | --------------------------- |
| Interval             | 1                 | 2                    | 4                     | 8                        | 16                       | 32                        |    
| Number of t\(point\) | 30                | 15                   | 15                    | 15                       | 15                       | 30                        |     
| Example of t serial  | t1,t2,t3\.\.\.t30 | t31,t33,t35\.\.\.t59 | t61,t65,t69\.\.\.t117 | t121,t129,t137\.\.\.t233 | t241,t257,t273\.\.\.t465 | t481,t513,t545\.\.\.t1409 |    

Here is the serial array in code:
```python
#convert to training/test_y with serial dataset
DatasetConvertor.convertParkingData(....
                                    y_oneDaySerial = np.arange(1, 31, 1).tolist() + \
                                                    np.arange(31, 60, 2).tolist() + \
                                                    np.arange(61, 120, 4).tolist() + \
                                                    np.arange(121, 240, 8).tolist() + \
                                                    np.arange(241, 480, 16).tolist() + \
                                                    np.arange(481, 24*60+1, 32).tolist()
                                    # y_oneDaySerial = [1, 2, 3, 4, 5, ... , 29, 30,
                                    #                   31, 33, 35, ... , 59,
                                    #                   61, 65, 69, ... , 113, 117,
                                    #                   121, 129, 137, ... , 225, 233,
                                    #                   241, 257, 273, ... , 449, 465,
                                    #                   481, 513, 545, ..., 1377, 1409]
                                    )
```

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
                                    interval = 1, # here is 1, means the input dataset sample rate is 1 minute per occupancy status, if it's 5, means 5 minute per occupancy status.
                                    maxParkingSlots = 50, # the max number of parking slots of output dataset, if it is 0, means all parking slots.
                                    y_oneDaySerial = np.arange(1, 31, 1).tolist() # [1,2,3,...,30], the defualt value is False, which means no serial t in _y dataset, only one t1 after _x. 
                                    )

#the output file should be output_x.csv(training/test_x dataset) and output_y.csv(training/test_y dataset)
```

Known Issue in input file, current the file from data collection team(Bruce) is missing some parking data because there is not in orignal data. This module can avoid this problem.