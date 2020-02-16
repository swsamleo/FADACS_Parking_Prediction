
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import re


def time_12_to_24(datetime):
    """Change 12-hour clock to 24-hour clock.

    Return 24-hour clock of the given datetime. 

    For example,
        if given date is "05/07/2019 1:48:38 PM", 
        then return "05/07/2019 13:48:38"

    Args:
        datetime: A string, format like "MM/DD/YY h:m:s AM(PM)".

    Returns:
        A string, format like "MM/DD/YY h:m:s" in 24-hour clock.
    """

    # split datetime into ["MM/DD/YY", "h:m:s", "AM/PM"]
    datetime_list = datetime.split(" ")
    time_list = datetime_list[1].split(":")    # ["h", "m", "s"]

    if datetime_list[2] == "AM":    # 12 AM(midnight), 1-11 AM
        if time_list[0] == "12":
            time_list[0] = "00"
            datetime_list[1] = ":".join(time_list)
            return " ".join(datetime_list[:2])
        else:
            return " ".join(datetime_list[:2])

    elif datetime_list[2] == "PM":    # 12 PM(noon), 1-11 PM
        if time_list[0] == "12":
            return " ".join(datetime_list[:2])
        else:
            time_list[0] = str(int(time_list[0]) + 12)
            datetime_list[1] = ":".join(time_list)
            return " ".join(datetime_list[:2])


def date_upbound(date):
    """The upbound the given date.

    Return the midnight of the given date. 

    For example,
        if given date is "05/07/2019 13:48:38", 
        then return "05/08/2019 00:00:00"

    Args:
        date: A string, format like "MM/DD/YY h:m:s".

    Returns:
        A string, format like "MM/DD/YY 00:00:00".
    """

    datatime_time = datetime.strptime(date.split()[0], "%m/%d/%Y")

    return (datatime_time + timedelta(days=1)).strftime("%m/%d/%Y %H:%M:%S")


def data_clean_car_parking_data(read_file_path, write_file_path, data_year=None):
    """Clean the car parking sensor data.

    Load the data, cleaning including remove outlier, remove no vehicle present data

    Args:
        read_file_path: The reading path of file.
        write_file_path: The writing path of file.
        data_year: optional, given the year of the data, name into file. 
                   Otherwise, use the current datatime.

    Raises:
        pd.errors.ParserError: Exception that is raised by an error encountered in pd.read_csv.
    """

    if not data_year:
        if re.findall(r"\d+", read_file_path) != list():
            data_year = re.findall(r"\d+", read_file_path)[0]
        else:
            data_year = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    try:
        ####################
        # 1. Load the Data #
        ####################

        dtype = {
            "DeviceId": np.int32,
        }    # To reduce usage of memory

        # Read CSV from read_file_path
        df_car_parking = pd.read_csv(
            read_file_path,
            dtype=dtype,
        )

        print("Loading Done!")

    except pd.errors.ParserError:
        print("!! ParserError. Stop Loading")

    else:
        #####################
        # 2. Clean the Data #
        #####################

        # Clean 1:
        # Check the relation between column "StreetId" and "StreetName",
        # if 1-to-1 relation, then de-duplicates and save into a new CSV,
        # Also drop column "StreetName" from original DataFrame.

        df_streetId_2_name = df_car_parking[[
            "StreetId", "StreetName"]].drop_duplicates()
        if df_streetId_2_name.shape[0] == df_car_parking[["StreetId"]].drop_duplicates().shape[0] and \
                df_streetId_2_name.shape[0] == df_car_parking[["StreetName"]].drop_duplicates().shape[0]:

            df_car_parking = df_car_parking.drop(
                ["StreetName"], axis=1)    # Drop the col "StreetName"

            df_streetId_2_name.to_csv(
                write_file_path + "/street_id_2_name_" + data_year + ".csv", index=False)

        else:
            del df_streetId_2_name

        print("... Cleaning 1 Done!")

        # Clean 2:
        # Remove the records that the "Vehicle Present" is False.

        # Extract the Vehicle Present
        df_car_parking = df_car_parking[df_car_parking["Vehicle Present"] == True]
        df_car_parking = df_car_parking.drop(
            "Vehicle Present", axis=1)    # Drop the col "Vehicle Present"

        print("... Cleaning 2 Done!")

        # Clean 3:
        # Remove the records that have the non-positive "DurationSeconds".

        df_car_parking = df_car_parking[df_car_parking.DurationSeconds > 0]

        print("... Cleaning 3 Done!")

        # Clean 4:
        # Remove the records that both "DepartureTime" and "ArrivalTime" are in the midnight.

        # Convert the "ArrivalTime" and "DepartureTime" from 12-hour to 24-hour
        df_car_parking["ArrivalTime"] = df_car_parking["ArrivalTime"].apply(
            time_12_to_24)
        df_car_parking["DepartureTime"] = df_car_parking["DepartureTime"].apply(
            time_12_to_24)

        df_car_parking = df_car_parking.drop(df_car_parking[
            df_car_parking["ArrivalTime"].str.contains("00:00:00") &
            df_car_parking["DepartureTime"].str.contains("00:00:00")
        ].index, axis=0)

        print("... Cleaning 4 Done!")

        # Clean 5:
        # Remove the records that the "DepartureTime" is over the midnight of "ArrivalTime".

        df_car_parking["DepartureTime_upbond"] = df_car_parking["ArrivalTime"].apply(
            date_upbound)    # Set the the upbound of DepartureTime
        df_car_parking = df_car_parking[df_car_parking["DepartureTime_upbond"]
                                        >= df_car_parking["DepartureTime"]]    # Remove Wrong recording

        # Drop the new col "DepartureTime_upbond"
        df_car_parking = df_car_parking.drop("DepartureTime_upbond", axis=1)

        print("... Cleaning 5 Done!")

        # Clean 6:
        # Remove the records that overlap with other records

        # Sort by "StreetMarker" first, then by "ArrivalTime"
        df_car_parking = df_car_parking.sort_values(
            ["StreetMarker", "ArrivalTime"])
        df_car_parking = df_car_parking.reset_index(drop=True)

        # Adding 4 columns to assist check overlap
        df_car_parking["Last_StreetMarker"] = pd.concat(
            [pd.Series("-1"), df_car_parking["StreetMarker"].head(-1)]
        ).reset_index(drop=True)
        df_car_parking["Next_StreetMarker"] = df_car_parking["StreetMarker"].append(
            pd.Series("-1"), ignore_index=True)[1:].tolist()

        df_car_parking["Last_DepartureTime"] = pd.concat(
            [df_car_parking["DepartureTime"].head(
                1), df_car_parking["DepartureTime"].head(-1)]
        ).reset_index(drop=True)
        df_car_parking["Next_ArrivalTime"] = df_car_parking["ArrivalTime"].append(
            pd.Series([df_car_parking["ArrivalTime"]
                       [df_car_parking["ArrivalTime"].shape[0]-1]]),
            ignore_index=True
        )[1:].tolist()

        # Check the overlap by the following:
        # IF StreetMarker==LastStreetMarker & DepartureTime>NextArrivalTime:
        #     THEN Overlap
        # OR
        # IF StreetMarker==NextStreetMarker & ArrivalTime<LastDepartureTime:
        #     THEN Overlap
        df_car_parking["overlap"] = \
            (
                (df_car_parking["Next_StreetMarker"] == df_car_parking["StreetMarker"]) &
                (df_car_parking["DepartureTime"] >
                 df_car_parking["Next_ArrivalTime"])
        ) | \
            (
                (df_car_parking["Last_StreetMarker"] == df_car_parking["StreetMarker"]) &
                (df_car_parking["ArrivalTime"] <
                 df_car_parking["Last_DepartureTime"])
        )

        df_car_parking = df_car_parking[df_car_parking["overlap"] == False]
        df_car_parking = df_car_parking.drop(
            ["Last_StreetMarker", "Next_StreetMarker",
                "Last_DepartureTime", "Next_ArrivalTime", "overlap"],
            axis=1
        )

        print("... Cleaning 6 Done!")

        # Clean 7:
        # Keep the Lexicographical order of column "BetweenStreet1" and "BetweenStreet2",
        # such that "BetweenStreet1" < "BetweenStreet2".

        # True, if need change
        df_car_parking["BetweenStChange"] = df_car_parking["BetweenStreet1"] > df_car_parking["BetweenStreet2"]

        col1 = df_car_parking.loc[df_car_parking[df_car_parking["BetweenStChange"]
                                                 ].index, "BetweenStreet2"]
        col2 = df_car_parking.loc[df_car_parking[df_car_parking["BetweenStChange"]
                                                 ].index, "BetweenStreet1"]

        df_car_parking.loc[df_car_parking[df_car_parking["BetweenStChange"]
                                          ].index, "BetweenStreet2"] = col2
        df_car_parking.loc[df_car_parking[df_car_parking["BetweenStChange"]
                                          ].index, "BetweenStreet1"] = col1

        df_car_parking = df_car_parking.drop(
            "BetweenStChange", axis=1)    # Drop the two new columns

        del col1
        del col2

        print("... Cleaning 7 Done!")

        print("Cleaning Done!")

        ####################
        # 3. Save the Data #
        ####################

        if write_file_path[-1] == "/":
            file_name = "car_parking_" + str(data_year) + ".csv"
        else:
            file_name = "/car_parking_" + str(data_year) + ".csv"

        # Write the cleand data
        df_car_parking.to_csv(
            write_file_path + file_name,
            index=False
        )

        print("Saving Done!")


def slice_point_car_parking_data(read_file_path, write_file_path, interval):
    """Check the occupied or not at time point.

    Check the occupied or not of each parking slot every interval of each day.

    Args:
        read_file_path: The reading path of file.
        write_file_path: The writing path of file.
        interval: The interval used to slice the data in each interval.
    """

    ####################
    # 1. Load the Data #
    ####################

    dtype = {
        "DeviceId": np.int32,
    }  # To reduce usage of memory

    df_car_parking = pd.read_csv(
        read_file_path,
        dtype=dtype,
    )

    # Remove some useless columns
    df_car_parking = df_car_parking[
        ["ArrivalTime", "DepartureTime", "StreetMarker"]
    ]

    # Split the ArrivalTime and DepartureTime into the Date ("MM/DD/YY") and Time ("h:m:s")
    df_car_parking["ArrivalTime_Date"] = \
        df_car_parking["ArrivalTime"].apply(lambda x: x.split()[0])
    df_car_parking["ArrivalTime_Time"] = \
        df_car_parking["ArrivalTime"].apply(lambda x: x.split()[1])
    df_car_parking["DepartureTime_Time"] = \
        df_car_parking["DepartureTime"].apply(
            lambda x: x.split()[1]).replace("00:00:00", "24:00:00")

    print("Load Done!")

    ########################
    # 2. Slice each record #
    ########################

    # Given interval, slice 24 hours into each time point
    # Check occupied or not of each record at these time points

    # Generate compare time list
    candidates = []
    for H in range(0, 24):
        for M in range(0, 60, interval):
            candidates.append("{:02d}:{:02d}:00".format(H, M))

    # Check in each time, occupied or not
    for label in candidates:

        # If T1 <= label and T2 > label, then 1 (occupied); else 0 (unoccupied)
        df_car_parking[label] = (
            (df_car_parking["ArrivalTime_Time"] <= label) & (
                df_car_parking["DepartureTime_Time"] > label) + 0
        ).astype("int8")

    # Remove some useless columns
    df_car_parking = df_car_parking.drop(
        [
            "ArrivalTime", "DepartureTime", "ArrivalTime_Time", "DepartureTime_Time"
        ], axis=1)

    print("Slice Done!")

    ############################
    # 3. Merge the same record #
    ############################

    # Sum the occupied of each parking slot at the same date,
    # Loop / iterate several columns in order to avoid memory issues,
    # group by ["StreetMarker", "ArrivalTime_Date"]

    columns_num = 60 // interval

    df_car_parking_point_slice = df_car_parking.iloc[:, :2+columns_num]\
        .groupby(["StreetMarker", "ArrivalTime_Date"])\
        .sum()\
        .reset_index()

    for h in range(1, 24):
        col_index = [0, 1]
        col_index.extend(
            [2 + i + columns_num * h for i in range(columns_num)]
        )

        tmp = df_car_parking.iloc[:, col_index]\
            .groupby(["StreetMarker", "ArrivalTime_Date"])\
            .sum()\
            .reset_index()

        df_car_parking_point_slice = df_car_parking_point_slice.merge(
            tmp,
            how="left",
            on=["StreetMarker", "ArrivalTime_Date"]
        )

    # Check the occupied value must be 1 or 0
    if (df_car_parking_point_slice.iloc[:, 2:] > 1).sum(axis=1).sum() != 0:

        # Convert the value over 1 to 1.
        for label in df_car_parking_point_slice.columns.tolist()[2:]:

            df_car_parking_point_slice.loc[df_car_parking_point_slice[label] > 1, label] \
                = [1]*df_car_parking_point_slice[df_car_parking_point_slice[label] > 1].shape[0]

            df_car_parking_point_slice[label] = df_car_parking_point_slice[label].astype(
                "int8")

    print("Merge Done!")

    ####################
    # 4. Save the Data #
    ####################

    file_name = "car_parking_" + str(interval) + "_point.csv"

    # Save into file
    df_car_parking_point_slice.to_csv(
        write_file_path,
        index=False
    )
    print("Saving Done!")


def slice_interval_car_parking_data(read_file_path, write_file_path, interval):
    """Check the occupied or not during time period.

    Check the occupied or not of each parking slot each period of each day.

    Args:
        read_file_path: The reading path of file.
        write_file_path: The writing path of file.
        interval: The interval used to slice the data in each interval.
    """

    ####################
    # 1. Load the Data #
    ####################

    dtype = {
        "DeviceId": np.int32,
    }  # To reduce usage of memory

    df_car_parking = pd.read_csv(
        read_file_path,
        dtype=dtype,
    )

    # Remove some useless columns
    df_car_parking = df_car_parking[
        ["ArrivalTime", "DepartureTime", "StreetMarker"]
    ]

    # Split the ArrivalTime and DepartureTime into the Date ("MM/DD/YY") and Time ("h:m:s")
    df_car_parking["ArrivalTime_Date"] = \
        df_car_parking["ArrivalTime"].apply(lambda x: x.split()[0])
    df_car_parking["ArrivalTime_Time"] = \
        df_car_parking["ArrivalTime"].apply(lambda x: x.split()[1])
    df_car_parking["DepartureTime_Time"] = \
        df_car_parking["DepartureTime"].apply(
            lambda x: x.split()[1]).replace("00:00:00", "24:00:00")

    print("Load Done!")

    ########################
    # 2. Slice each record #
    ########################

    # Given interval, slice 24 hours into each time period
    # Check occupied or not of each record at these time period

    # Generate compare time list
    candidates = []
    for H in range(0, 24):
        for M in range(0, 60, interval):
            candidates.append("{:02d}:{:02d}:00".format(H, M))

    candidates.append("24:00:00")

    # Check each record in each time intervals
    for i in range(len(candidates) - 1):

        C1 = candidates[i]
        C2 = candidates[i+1]

        label = "{:s}-{:s}".format(C1, C2)  # generate label

        # if T1 < C2 and T2 > C1, then 1 (occupied); else 0 (unoccupied)
        df_car_parking[label] = (
            (df_car_parking["ArrivalTime_Time"] < C2) & (
                df_car_parking["DepartureTime_Time"] > C1) + 0
        ).astype("int8")

    # Remove some useless columns
    df_car_parking = df_car_parking.drop(
        [
            "ArrivalTime", "DepartureTime", "ArrivalTime_Time", "DepartureTime_Time"
        ], axis=1)
    
    print("Slice Done!")

    ############################
    # 3. Merge the same record #
    ############################

    # Sum the occupied of each parking slot in each day,
    # Loop / iterate several columns in order to avoid memory issues,
    # group by ["StreetMarker", "ArrivalTime_Date"]

    columns_num = 60 // interval

    df_car_parking_interval_slice = df_car_parking.iloc[:, :2+columns_num]\
        .groupby(["StreetMarker", "ArrivalTime_Date"])\
        .sum()\
        .reset_index()

    for h in range(1, 24):
        col_index = [0, 1]
        col_index.extend(
            [2 + i + columns_num * h for i in range(columns_num)]
        )

        tmp = df_car_parking.iloc[:, col_index]\
            .groupby(["StreetMarker", "ArrivalTime_Date"])\
            .sum()\
            .reset_index()

        df_car_parking_interval_slice = df_car_parking_interval_slice.merge(
            tmp,
            how="left",
            on=["StreetMarker", "ArrivalTime_Date"]
        )

    # convert the occupied value over 1 to 1.
    for label in df_car_parking_interval_slice.columns.tolist()[2:]:

        df_car_parking_interval_slice.loc[df_car_parking_interval_slice[label] > 1, label] \
            = [1]*df_car_parking_interval_slice[df_car_parking_interval_slice[label] > 1].shape[0]

        df_car_parking_interval_slice[label] = df_car_parking_interval_slice[label].astype(
            "int8")

    print("Merge Done!")

    ####################
    # 4. Save the Data #
    ####################

    file_name = "car_parking_" + str(interval) + "_interval.csv"

    # Save into file
    df_car_parking_interval_slice.to_csv(
        write_file_path,
        index=False
    )
    print("Saving Done!")
