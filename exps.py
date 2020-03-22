import Trier as trier
import os


## Experiment 1
exp = trier.Experiment()

if os.path.exists("./exps/t0"+'/exp.json'):
    exp.setup(path = "./exps/t0",    
             start = "01/09/2017",    
             end = "01/13/2017",
             testStart = "02/13/2017",
             testEnd = "02/16/2017",
             number = 5,                #How many parking lots
             location = "MelbCity",     #MelbCity or Mornington(not ready yet)
         features = ['Temp', 'Wind', 'Humidity', 'Barometer', 'Extreme_weather', 'min_dis0.05', 'num_of_poi0.05', 'num_of_open_poi0.05', 'mean_dis0.05', 'min_dis0.1', 'num_of_poi0.1', 'num_of_open_poi0.1', 'mean_dis0.1', 'min_dis0.2', 'num_of_poi0.2', 'num_of_open_poi0.2', 'mean_dis0.2', 'min_dis0.3', 'num_of_poi0.3', 'num_of_open_poi0.3', 'mean_dis0.3', 'min_dis0.4', 'num_of_poi0.4', 'num_of_open_poi0.4', 'mean_dis0.4', 'min_dis0.5', 'num_of_poi0.5', 'num_of_open_poi0.5', 'mean_dis0.5', 'min_dis0.8', 'num_of_poi0.8', 'num_of_open_poi0.8', 'mean_dis0.8', 'min_dis1.0', 'num_of_poi1.0', 'num_of_open_poi1.0', 'mean_dis1.0', 'availability', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'DayOfMonth', 'DayOfYear'],
             reLoadExistDir = True)     #If True and a experiment exists in the path, it will reload it.
                                        #If False, it would create/recreate a experiment in the path.
     
    exp.prepareTTDatasets()
    
    exp.add({
        "model":"LightGBM"
    })

    exp.add({
        "model":"FNN",
        "parameters":{
            "batch_size" : 1000,
            "epochs" : 100,
            "learningRate" : 0.01,
            "momentum" : 0.9,
            "decay" : 0.01,
            "nesterov" : False,
            "loss" :'mean_squared_error',
            "activation" : "relu",
            "verbose" : 1
        }
    })

    exp.add({
        "model":"LSTM",
        "parameters":{
            "batch_size" : 1000,
            "epochs" : 100,
            "learningRate" : 0.01,
            "monitor" : 'val_loss',
            "mode" : "min",
            "nesterov" : False,
            "loss" :'mae',
            "verbose" : 1
        }
    })
    exp.run()
else:
    exp.loadConfig("./exps/t0/exp.json")
    exp.run()
    

## Experiment 2
exp = trier.Experiment()

if os.path.exists("./exps/t1"+'/exp.json'):
    exp.setup(path = "./exps/t1",    
             start = "01/09/2017",    
             end = "02/09/2017",
             testStart = "02/10/2017",
             testEnd = "03/10/2017",
             number = 100,                #How many parking lots
             location = "MelbCity",     #MelbCity or Mornington(not ready yet)
             features = ['Temp', 'Wind', 'Humidity', 'Barometer', 'Extreme_weather', 'min_dis0.05', 'num_of_poi0.05', 'num_of_open_poi0.05', 'mean_dis0.05', 'min_dis0.1', 'num_of_poi0.1', 'num_of_open_poi0.1', 'mean_dis0.1', 'min_dis0.2', 'num_of_poi0.2', 'num_of_open_poi0.2', 'mean_dis0.2', 'min_dis0.3', 'num_of_poi0.3', 'num_of_open_poi0.3', 'mean_dis0.3', 'min_dis0.4', 'num_of_poi0.4', 'num_of_open_poi0.4', 'mean_dis0.4', 'min_dis0.5', 'num_of_poi0.5', 'num_of_open_poi0.5', 'mean_dis0.5', 'min_dis0.8', 'num_of_poi0.8', 'num_of_open_poi0.8', 'mean_dis0.8', 'min_dis1.0', 'num_of_poi1.0', 'num_of_open_poi1.0', 'mean_dis1.0', 'availability', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'DayOfMonth', 'DayOfYear'],
             reLoadExistDir = False)     #If True and a experiment exists in the path, it will reload it.
                                        #If False, it would create/recreate a experiment in the path.
     
    exp.prepareTTDatasets()
    
    exp.add({
        "model":"LightGBM"
    })

    exp.add({
        "model":"FNN",
        "parameters":{
            "batch_size" : 1000,
            "epochs" : 100,
            "learningRate" : 0.01,
            "momentum" : 0.9,
            "decay" : 0.01,
            "nesterov" : False,
            "loss" :'mean_squared_error',
            "activation" : "relu",
            "verbose" : 1
        }
    })

    exp.add({
        "model":"LSTM",
        "parameters":{
            "batch_size" : 1000,
            "epochs" : 100,
            "learningRate" : 0.01,
            "monitor" : 'val_loss',
            "mode" : "min",
            "nesterov" : False,
            "loss" :'mae',
            "verbose" : 1
        }
    })
else:
    exp.loadConfig("./exps/t1/exp.json")
    exp.run()