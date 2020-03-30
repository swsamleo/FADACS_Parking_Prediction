import Trier as trier
import os


## Experiment 2
exp = trier.Experiment()

if os.path.exists("./eptest/t2"+'/exp.json') == False:
    exp.setup(path = "./eptest/t2",    
            locations = {
                "MelbCity" : {
                    "start" : "01/01/2017",
                    "end" : "01/15/2017",
                    "testStart" : "02/02/2017",
                    "testEnd" : "02/16/2017",
                    "number" : 50,
                },
                "Mornington" : {
                    "start" : "02/01/2020",
                    "end" : "02/15/2020",
                    "testStart" : "02/11/2020",
                    "testEnd" : "02/26/2020",
                    "number" : 50,
                }
            },
            baseUnit = 6,
            interval = 5,
            features = ['Temp', 'Wind', 'Humidity', 'Barometer', 'Extreme_weather','num_of_poi1.0', 'num_of_open_poi1.0','num_of_poi0.5', 'num_of_open_poi0.5','min_dis1.0', 'num_of_poi1.0', 'Day', 'Hour','DayOfWeek', 'DayOfMonth','availability'],
            reLoadExistDir = False)

    exp.add({
            "model":"LightGBM",
            "location":"MelbCity",
             "parameters":{
                "learningRate" : 0.01,
                "num_leaves" : 50,
                "n_estimators" : 50,
                "verbose" : True,
                "early_stopping_rounds" : 5
            }
    })

    exp.add({
            "model":"DANN",
            "source":"MelbCity",
            "target":"Mornington",
            "parameters":{
                "batchSize":1000,
                "nepoch" : 100,
                "learningRate" : 1e-2,
                "gamma" : .5,
                "input_dim" : 50,
                "hidden_dim" : 25,
                "featureSize" : 102,
                "outFeatureSize" : 50,
                }
    })
    
    exp.add({
            "model":"ADDA",
            "source":"MelbCity",
            "target":"Mornington",
            "parameters":{
                "batchSize":1000,
                "num_epochs" : 100,
                "num_epochs_pre" : 1000,
                "d_learning_rate" : 1e-4,
                "c_learning_rate" : 1e-4,
                "d_hidden_dims" : 100,
                "d_input_dims" : 50,
                "e_input_dims" : 102,
                "e_hidden_dims" : 100,
                }
    })

    exp.add({
        "model":"FNN",
        "location":"MelbCity",
        "parameters":{
            "batch_size" : 1000,
            "epochs" : 100,
            "learningRate" : 0.000001,
            "loss" :'mean_squared_error',
            "activation" : "relu",
            "verbose" : 1
        }
    })

    exp.add({
        "model":"LSTM",
        "location":"MelbCity",
        "parameters":{
            "batch_size" : 1000,
            "epochs" : 100,
            "learningRate" : 0.001,
            "monitor" : 'val_loss',
            "mode" : "min",
            "nesterov" : False,
            "loss" :'mae',
            "verbose" : 1
        }
    })
    exp.prepareTTDatasets()
    exp.run()
else:
    exp.loadConfig("./eptest/t2/exp.json")
    exp.run()
