import pmdarima as pm
from . import ModelUtils
import numpy as np
import time

def evaluate(predictOffest,trainY, testY, params):
    print("ARIMA evaluation start!")
    start = time.time()

    m = {
    "mae" :0,
    "rmse"  : 0,
    "mase" : 0,
    "r2" : 0,
    }

    order=(1, 1, 2)
    seasonal_order=(0, 1, 1, 12)

    if params is not None:
        if "order" in params: order = tuple(params["order"])
        if "seasonal_order" in params: order = tuple(params["seasonal_order"])

    arima = pm.ARIMA(order=order, seasonal_order=seasonal_order)
    arima.fit(trainY)

    count = len(testY) //predictOffest - 1
    print("ARIMA count:{} predictOffest:{} len(testY):{}".format(count,predictOffest,len(testY)))

    for i in range((len(testY) //predictOffest) - 1):
        start1 = time.time()
        forecasts = arima.predict(predictOffest)
        
        forecasts = [0 if a_ < 0.01 else a_ for a_ in forecasts]
        forecasts = [1 if a_ > 1 else a_ for a_ in forecasts]
        
        updateT = None
        if i*predictOffest+predictOffest < len(testY):
            updateT = testY[i*predictOffest:i*predictOffest+predictOffest]
        elif i*predictOffest+predictOffest >= (len(testY) -1):
            updateT = testY[i*predictOffest:len(testY) -1]
            predictLen = len(testY) - 1 - i*predictOffest
        
        trainY = np.concatenate((trainY, updateT), axis=None)
        
        arima.update(updateT)
        
        _m = ModelUtils.getMetrics(updateT,forecasts,trainY)
        
        m["mae"] = _m["mae"]
        m["rmse"] = _m["rmse"]
        m["rmse"] = _m["mase"]
        m["r2"] = _m["r2"]
        end1 = time.time()
        print("{}/{}".format(i,count) + (' ARIMA sub MAE: %.2f' % m["mae"])+",spent: %.4fs" % (end1 - start1))

    xlen = len(testY) // predictOffest 
    print("ARIMA xlen:{} predictOffest:{}".format(xlen,predictOffest))

    m["mae"] = m["mae"]/xlen
    m["rmse"] = m["rmse"]/xlen
    m["rmse"] = m["mase"]/xlen
    m["r2"] = m["r2"]/xlen

    end = time.time()
    print((' ARIMA MAE: %.2f' % m["mae"])+",spent: %.4fs" % (end - start))

    return m
