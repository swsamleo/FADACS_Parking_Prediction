import Trier as trier
import os

expPath = "./experimentdir"
exp = trier.Experiment()

if os.path.exists(expPath+'/exp.json') == False:
    
    exp.setup(path = expPath,    
            locations = {
                "MelbCity" : {
                    "start" : "02/01/2017", 
                    "end" : "02/28/2017", 
                    "number" : 50,  #park lots number
                    "random":True,  #random training data
                    "randomParkingIDs":False, #random choose park lots id
                    "ParkingIDMetric":"median" #median, or mode
                },
                "Mornington" : {
                    "start" : "02/01/2020", 
                    "end" : "02/28/2020", 
                    "number" : 50, 
                    "random":True,
                    "randomParkingIDs":False,
                    "ParkingIDMetric":"median"
                }
            },
            medianSource = "MelbCity", #
            baseUnit = 6, #time window number, 6 x5min
            interval = 5, #time window, 5min 
            features = ['Temp', 'Wind', 'Humidity', 'Barometer', 'Extreme_weather','num_of_poi1.0', 'num_of_open_poi1.0','num_of_poi0.5', 'num_of_open_poi0.5','min_dis1.0', 'num_of_poi1.0', 'Hour','DayOfWeek', 'DayOfMonth','availability'],
            reLoadExistDir = False)

    # add experiments:
    exp.add({
            "model": "LSTM", #model name
            "location": "Mornington", #location
            "trainWithParkingData":False,  # whether include parking data in train dataset,if include plz remove this line!
            "parameters": {
#             "batch_size": 100,
#             "epochs": 4000,
#             "hidden_size": 48,
#             "learningRate": 0.000005,
#             "monitor": "val_loss",
#             "mode": "min",
#             "nesterov": False,
#             "loss": "mae",
#             "verbose": 1
                }
        })
    exp.add({
            "model": "convLSTM",
            "location": "Mornington",
            "trainWithParkingData":False, # whether include parking data in train dataset
            "parameters": { # add any parameters if need, see ./models/convLSTM.py
            }
        })

    exp.add({
            "model": "LSTM", #model name
            "location": "MelbCity", #location
            "trainWithParkingData":False,  # whether include parking data in train dataset
            "parameters": {# add any parameters if need, see ./models/LSTM.py
            }
        })
    exp.add({
            "model": "convLSTM",
            "location": "MelbCity",
            "trainWithParkingData":False, # whether include parking data in train dataset
            "parameters": {
            }
        })

    exp.add({
            "model": "LSTM", #model name
            "location": "Mornington", #location
            "parameters": {
            }
        })

    exp.add({
            "model": "convLSTM",
            "location": "Mornington",
            "parameters": {
            }
        })

    exp.add({
            "model": "LSTM", #model name
            "location": "MelbCity", #location
            "parameters": {
            }
        })
    exp.add({
            "model": "convLSTM",
            "location": "MelbCity",
            "parameters": {
            }
        })

    exp.add({
        "model": "ADDA",
        "source": "MelbCity",
        "target": "Mornington",
        "trainWithParkingData":False, # whether include parking data in train dataset, if include plz remove this line!
        "parameters": { # add any parameters if need, see ./models/ADDA.py
            "encoder": "MLP", # Normal ADDA
            "batchSize": 10000,
            "num_epochs": 100,
            "num_epochs_pre": 100,
            "d_learning_rate": 1e-05,
            "c_learning_rate": 1e-05,
            "e_input_dims": 90, # 6 * (0+15), 
            "e_hidden_dims": 45,
            "e_output_dims": 24,
            "r_input_dims": 24,
            "d_input_dims": 24,
            "d_hidden_dims": 12
        }
    })

    exp.add({
        "model": "ADDA",
        "source": "MelbCity",
        "target": "Mornington",
        "parameters": {
            "encoder": "MLP", # Normal ADDA
            "batchSize": 10000,
            "num_epochs": 100,
            "num_epochs_pre": 100,
            "d_learning_rate": 1e-05,
            "c_learning_rate": 1e-05,
            "e_input_dims": 96, # 6 * (1+15), 1 is parking data, 15: features number, 6 is time windows number, baseUnit
            "e_hidden_dims": 48, 
            "e_output_dims": 24,
            "r_input_dims": 24,
            "d_input_dims": 24,
            "d_hidden_dims": 12
        }
    })

    exp.add({
        "model": "ADDA",
        "source": "MelbCity",
        "target": "Mornington",
        "trainWithParkingData":False, # whether include parking data in train dataset, if include plz remove this line!
        "parameters": {
            "encoder": "convLSTM", # ADDA with convLSTM
            "batchSize": 32,
            "num_epochs": 100,
            "num_epochs_pre": 100,
            "d_learning_rate": 1e-05,
            "c_learning_rate": 1e-05,
            "e_input_dims": 90, # 6 * (0+15), 
            "e_hidden_dims": 48,
            "e_output_dims": 24,
            "r_input_dims": 24,
            "d_input_dims": 24,
            "d_hidden_dims": 12
        }
    })

    exp.add({
        "model": "ADDA",
        "source": "MelbCity",
        "target": "Mornington",
        "trainWithParkingData":False, # whether include parking data in train dataset, if include plz remove this line!
        "parameters": {
            "encoder": "convLSTM", # ADDA with convLSTM
            "batchSize": 32,
            "num_epochs": 100,
            "num_epochs_pre": 100,
            "d_learning_rate": 1e-05,
            "c_learning_rate": 1e-05,
            "e_input_dims": 90, # 6 * (0+15), 
            "e_hidden_dims": 48,
            "e_output_dims": 24,
            "r_input_dims": 24,
            "d_input_dims": 24,
            "d_hidden_dims": 12
        }
    })

    #exp.showConfig()
    exp.prepareTTDatasets()             # generate datasets

    exp.run()                           # run all experiments(no result only)

else:
    exp.loadConfig(expPath+"/exp.json") # load experiment file
    
    #exp.showConfig()                           # show all experiments configs
    
    #exp.showFeatureList()                      # show all possible features

    exp.run()                                    # run all experiments(no result only)

    #exp.run("2b39d340-b946-11ea-b691-0242ac110003") #run the experiment (UUID:2b39d340-b946-11ea-b691-0242ac110003)
