# @Time     : Jul. 10, 2020 19:45
# @Author   : Zhen Zhang
# @Email    : david.zhen.zhang@gmail.com
# @FileName : EPUtils.py
# @Version  : 1.0
# @IDE      : VSCode

import pyecharts.options as opts
from pyecharts.charts import Line
import uuid
import arrow
import json
import numpy as np

def drawMetricChart(title,timeAxis,lines):
    chart = (Line(init_opts=opts.InitOpts(width="1400px", height="660px"))
             .set_global_opts(
#                  datazoom_opts=opts.DataZoomOpts(range_start = 5,
#                                                 range_end = 10),
                 title_opts=opts.TitleOpts(title=title),
                 legend_opts=opts.LegendOpts(is_show=True,
                                                   orient = "vertical",
                                                   pos_left = "10%",
                                                   pos_top = "20%",
                                                   type_ = "scroll"),
            )
        )
    
    chart.add_xaxis(timeAxis)
    for lineName in lines:
        chart.add_yaxis(lineName, lines[lineName],is_smooth=True,is_symbol_show=False,is_selected = True)
    
    return chart
    

def drawResult(filePath):
    config = None
    
    with open(filePath) as json_file:
        config = json.load(json_file)
    
    if config is not None:
        timeAxis = [ arrow.get("2017-01-01 00:00:00").shift(seconds=+(config["predictionOffests"][x]*config["interval"]*60)).format("HH:mm") for x in np.arange(len(config["predictionOffests"])).tolist()]
        
        mae_lines = {}
        rmse_lines = {}
        r2_lines = {}
        
        for uuid in config["results"]:
            model = config["experiences"][uuid]["model"]
            mae = []
            rmse = []
            r2 = []
            
            b = {}
            for e in range(len(config["predictionOffests"])):
                ts = dict(config["results"][uuid][e])
                for x in ts.keys():
                    #print(x)
                    b[x] = ts[x]
            
            #print(b)
            
            for i in config["predictionOffests"]:
                print(b["t"+str(i)])
                mae.append(b["t"+str(i)]["mae"])
                rmse.append(b["t"+str(i)]["rmse"])
                #if "r2" in config["results"][uuid]["t"+str(i)]:
                r2.append(b["t"+str(i)]["r2"])
                
            mae_lines[model+" ["+uuid[:8]+"]"]=mae
            rmse_lines[model+" ["+uuid[:8]+"]"]=rmse
            #if len(r2) > 0:
            r2_lines[model+" ["+uuid[:8]+"]"]=r2
        
        return drawMetricChart("MAE",timeAxis,mae_lines),drawMetricChart("RMSE",timeAxis,rmse_lines),drawMetricChart("R2",timeAxis,r2_lines)
        
        
    