import sys
sys.path.append("..")

import TTRunner

parkingMinsFile = '/run/user/1000/gvfs/smb-share:server=ds-tokyo.local,share=downloads/pp/rmit-parking-prediction/car_parking_2017_5mins_point.csv'
outputDir = '/run/user/1000/gvfs/smb-share:server=ds-tokyo.local,share=downloads/pp/'

#2day 5 parking slots
TTRunner.generateTTDataSetAndRun(parkingMinsFile = parkingMinsFile,outputDir =outputDir, parkingSlotsNum=5,interval=5,
                        tts=["SVR","LightGBM"],
                        trainStart="01/01/2017",
                        trainEnd="01/03/2017",
                        testStart="01/08/2017",
                        testEnd="01/11/2017")