# @Time     : Jul. 10, 2020 19:45
# @Author   : Zhen Zhang
# @Email    : david.zhen.zhang@gmail.com
# @FileName : run_FADACS.py
# @Version  : 1.0
# @IDE      : VSCode

import keras
import Trier as trier
import os

exp = trier.Experiment()
exp.loadConfig("./experiments/FADACS/exp.json")
#exp.prepareTTDatasets()
#exp.loadConfig("./eptest/median-melb-50/exp.json")

# exp.add({
#         "model": "LSTM",
#         "location": "Mornington",
#         "trainWithParkingData":False,
#         "parameters": {
#         }
#     })
# exp.add({
#         "model": "convLSTM",
#         "location": "Mornington",
#         "trainWithParkingData":False,
#         "parameters": {
#         }
#     })

# exp.add({
#         "model": "ARIMA",
#         "location": "Mornington",
#         "testStart":"02/06/2017",
#         "testEnd":"02/07/2017"
#     })

# exp.add({
#         "model": "ARIMA",
#         "location": "Mornington",
#         "testStart":"02/06/2020",
#         "testEnd":"02/07/2020"
#     })

# exp.add({
#     "model": "ADDA",
#     "source": "MelbCity",
#     "target": "Mornington",
#     "trainWithParkingData":False,
#     "parameters": {
#         "encoder": "MLP",
#         "batchSize": 10000,
#         "num_epochs": 100,
#         "num_epochs_pre": 100,
#         "d_learning_rate": 1e-05,
#         "c_learning_rate": 1e-05,
#         "e_input_dims": 96,
#         "e_hidden_dims": 48,
#         "e_output_dims": 24,
#         "r_input_dims": 24,
#         "d_input_dims": 24,
#         "d_hidden_dims": 12
#     }
# })

#exp.run("f0cd689a-b92a-11ea-a733-8307e46c3f58")
#exp.run("d742a6b4-bb79-11ea-8ba9-0242ac110002")
#exp.run("7f5c7a1e-b92c-11ea-aabc-1b1a91e842ya")
#exp.run("7f5c7a1e-b92c-11ea-aabc-1b1a91e842wa")

exp.run()