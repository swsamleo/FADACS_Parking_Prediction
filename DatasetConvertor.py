import csv
import arrow
import numpy as np

DATE_FORMAT = 'MM/DD/YYYY'

def diffDays(yesterday,today):
    y = arrow.get(yesterday, DATE_FORMAT)
    t = arrow.get(today, DATE_FORMAT)
    return (t - y).days


def getValueOfBit(binraryInt,index,blen):
    return (binraryInt >> (blen - index - 1))&1


# Return a parking event string(0001110, 0:no car, 1:has car) for days if comparing yesterday date and today date.
# eventNum: how many parking event a day. if it's one minute ,it should be 60x24
def getMissingDaysData(yesterday, today,eventNum):
    diffdays = diffDays(yesterday,today)
    dataStr=""
    if diffdays > 1:
        for n in range(1,diffdays):
           dataStr += "0" * eventNum
    return dataStr


def split(word):
    return [char for char in word] 


def getSubParkingData(yearData,startIndex,endIndex,opt):
    if opt == "str" :
        return yearData[startIndex:endIndex]
    elif opt == "arr" :
        return split(yearData[startIndex:endIndex])
    elif opt == "int" :
        return int("1"+yearData[startIndex:endIndex],2)


def getSubParkingDataSerial(yearData,startIndex,inputSerial,opt):
    rowArray = list(yearData)
    serial = [x+startIndex for x in inputSerial]
    if opt == "str" :
        return "".join([rowArray[i] for i in serial])
    elif opt == "arr" :
        return [rowArray[i] for i in serial]
    elif opt == "int" :
        return int("1"+("".join([rowArray[i] for i in serial])),2)


def getSubParkingDataSet(id,yearData,startIndex,endIndex,tNum):
    length = len(yearData)
    if length < startIndex or length < endIndex:
        print("Error, the index is out of range!")
        return ""
    num = endIndex - startIndex - tNum
    res = ""
    for i in range(1, num+1):
        res += id+","+(",".join(getSubParkingData(yearData,startIndex+i,startIndex+i+tNum,"arr")))+"\n"
    return res


def getSubParkingDataSerialSet(id,yearData,startIndex,endIndex,serial):
    length = len(yearData)
    if length < (startIndex + serial[len(serial) -1]) or length < (endIndex + serial[len(serial) -1]):
        print("Error, the index is out of range!")
        return ""
    num = endIndex - startIndex - serial[len(serial) -1]
    res = ""
    for i in range(0, num):
        res += id+","+(",".join(getSubParkingDataSerial(yearData,startIndex+i,serial,"arr")))+"\n"
    return res


def saveToFiles(convert,outputCSV,outputCSV_x,outputCSV_y,parkingSlotID,parkingEventsYearStr,y_oneDaySerial,startIndex,endIndex,units):
    if convert == "convert":
        outputCSV.write(parkingSlotID+","+format(int(parkingEventsYearStr,2),"x")+"\n")
    elif convert == "extract": 
        if isinstance(y_oneDaySerial,list):
            outputCSV_x.write(getSubParkingDataSet(parkingSlotID,parkingEventsYearStr,startIndex,endIndex - y_oneDaySerial[len(y_oneDaySerial) - 1],units))
            outputCSV_y.write(getSubParkingDataSerialSet(parkingSlotID,parkingEventsYearStr,startIndex+units,endIndex,y_oneDaySerial))
        else:
            outputCSV_x.write(getSubParkingDataSet(parkingSlotID,parkingEventsYearStr,startIndex,endIndex,units))
            outputCSV_y.write(getSubParkingDataSet(parkingSlotID,parkingEventsYearStr,startIndex+units,endIndex+1,1))

##
def convertParkingData(*args, **kwargs):
    parkingMinsDataFile = kwargs["input"]
    outputFile = kwargs["output"]
    year = kwargs["year"]

    parkingSlotNum = 0
    maxParkingSlotsNum = 0

    y_oneDaySerial = False

    if "y_oneDaySerial" in kwargs:
        y_oneDaySerial = kwargs["y_oneDaySerial"]

    if "maxParkingSlots" in kwargs:
        maxParkingSlotsNum = kwargs["maxParkingSlots"]

    START_DATE_OF_YEAR = "01/01/"+year
    END_DATE_OF_YEAR = "12/31/"+year

    convert = "convert" #convert:only convert dataset, do not extract training/test dataset
                        #extract: extract training/test dataset in a range
    startDay = START_DATE_OF_YEAR
    endDay = END_DATE_OF_YEAR
    units = 30
    interval = 1
    startIndex = 1
    endIndex = 1

    if "convert" in kwargs:
        convert = kwargs["convert"]
        startDay = kwargs["start"]
        endDay = kwargs["end"]
        units = kwargs["units"]
        interval = int(kwargs["interval"])
        startIndex = (arrow.get(startDay, DATE_FORMAT) - arrow.get(START_DATE_OF_YEAR, DATE_FORMAT)).days * 60 * 24 // interval
        endIndex = (arrow.get(endDay, DATE_FORMAT) - arrow.get(START_DATE_OF_YEAR, DATE_FORMAT)).days * 60 *24 // interval

    print("checking line number of the input file ...")
    count = len(open(parkingMinsDataFile).readlines())
    #print("file lines:"+str(count))
    outputCSV = open(outputFile,'w')
    outputCSV_x =  outputCSV
    outputCSV_y =  outputCSV

    ## prepare output files
    if convert == "convert":
        outputCSV.write("id,data\n")
    elif convert == "extract": 
        print("startIndex:"+str(startIndex)+" endIndex:"+str(endIndex))
        outputCSV_x = open(outputFile[:-4] + '_x' + outputFile[-4:],'w')
        outputCSV_y = open(outputFile[:-4] + '_y' + outputFile[-4:],'w')
        outputCSV_x.write("id" + "".join([",t"+str(i) for i in range(1 - units, 1)])+"\n")
        if isinstance(y_oneDaySerial,list):
            outputCSV_y.write("id" + "".join([",t"+str(i) for i in y_oneDaySerial])+"\n")
        else:
            outputCSV_y.write("id,t\n")

    with open(parkingMinsDataFile, "r") as f:
        reader = csv.reader(f, delimiter=",")
        yesterday = ""
        parkingSlotID = ""
        parkingEventsYearStr = "1"
        for i, line in enumerate(reader):
            log_head = ("{:.2f}%("+str(parkingSlotNum)+")["+str(i)+"/"+str(count)+"]["+line[0]+"]").format(100*i/count)
            parkingRecordLen = len(line)
            parkingEventsNum = parkingRecordLen - 2
            parkingEvents = line[2:parkingRecordLen]
            parkingEventsStr = ''.join(parkingEvents)
            #print(log_head+"parkingEventsLen:"+str(parkingEventsNum))
            if len(parkingSlotID) == 0 and line[0] != "StreetMarker" :
                parkingSlotID = line[0]
                parkingSlotNum +=1

            elif len(yesterday) > 0 and parkingSlotID != line[0] : # goto the next parking slot
                if yesterday != END_DATE_OF_YEAR : #check if the last parking slot record date is not the last day of the year
                    #print(log_head+" "+parkingSlotID+" missing the last day of 2017")
                    today = arrow.get(END_DATE_OF_YEAR, DATE_FORMAT).shift(days=+1).format(DATE_FORMAT)
                    parkingEventsYearStr += getMissingDaysData(yesterday,today,parkingEventsNum)
                    saveToFiles(convert,outputCSV,outputCSV_x,outputCSV_y,parkingSlotID,parkingEventsYearStr,y_oneDaySerial,startIndex,endIndex,units)
                    print(log_head+" "+parkingSlotID+" saved@[1], the parkingEventsYearStr length:"+str(len(parkingEventsYearStr)-1))

                #goto the next parking slot, reset yesterday, and set parkingSlotID to current line.
                yesterday = ""
                parkingSlotID = line[0]
                parkingEventsYearStr = "1"
                parkingSlotNum +=1

                if maxParkingSlotsNum > 0 and maxParkingSlotsNum < parkingSlotNum : break

                if line[1] != START_DATE_OF_YEAR: #check if the first parking slot record date is not the first day of the year
                    yesterday = arrow.get(START_DATE_OF_YEAR, DATE_FORMAT).shift(days=-1).format(DATE_FORMAT)
                    #print(log_head+" missing the first day of 2017")
                
            if line[0] == "StreetMarker":
                continue
            
            #uncomment following line to debug the data of a sepcial parking slot ID
            #if parkingSlotID != "10001S":break

            parkingEventsYearStr += parkingEventsStr

            if len(yesterday) > 0 : #check date and make up the missing events(with 0) for days
                parkingEventsYearStr += getMissingDaysData(yesterday,line[1],parkingEventsNum)

            if line[1] == END_DATE_OF_YEAR : #meets the last record then save the parking slot data.
                saveToFiles(convert,outputCSV,outputCSV_x,outputCSV_y,parkingSlotID,parkingEventsYearStr,y_oneDaySerial,startIndex,endIndex,units)
                print(log_head+" "+parkingSlotID+" saved@[0], the parkingEventsYearStr length:"+str(len(parkingEventsYearStr)-1))

            if i == count-1 and line[1] != END_DATE_OF_YEAR: # the last line of input file
                today = arrow.get(END_DATE_OF_YEAR, DATE_FORMAT).shift(days=+1).format(DATE_FORMAT)
                parkingEventsYearStr += getMissingDaysData(line[1],today,parkingEventsNum)
                saveToFiles(convert,outputCSV,outputCSV_x,outputCSV_y,parkingSlotID,parkingEventsYearStr,y_oneDaySerial,startIndex,endIndex,units)
                print(log_head+" "+parkingSlotID+" saved@[2], the parkingEventsYearStr length:"+str(len(parkingEventsYearStr)-1))

            #print(log_head+" processed "+line[1])
            yesterday = line[1]
    
    if convert == "convert":
        outputCSV.close()
    elif convert == "extract": 
        outputCSV_x.close()
        outputCSV_y.close()
