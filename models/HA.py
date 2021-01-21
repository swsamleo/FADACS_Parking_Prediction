# @Time     : Jul. 10, 2020 19:45
# @Author   : Zhen Zhang
# @Email    : david.zhen.zhang@gmail.com
# @FileName : HA.py
# @Version  : 1.0
# @IDE      : VSCode

from . import ModelUtils
import time


def evaluate(data):
    print(data["loghead"] + data["col"] + " start!")
    start = time.time()

    x = data["x"]
    y = data["y"]
    tIndex = data["tIndex"]

    ha = x.mean(axis=1)

    number_of_data = y.shape[0]//y['id'].nunique()
    col_list = y.iloc[:,tIndex:tIndex+1].columns.values

    rst =  y['id'].to_frame().assign(**{col:ha for col in col_list})

    metrics = ModelUtils.getMetrics(y.iloc[:,tIndex:tIndex+1].values, rst.iloc[:,1:].values, None)

    end = time.time()
    print(data["loghead"] + data["col"] + (' HA MAE: %.2f' % metrics["mae"])+",spent: %.2fs" % (end - start))

    return {data["col"]:metrics}