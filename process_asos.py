"""
This module processes ASOS data. It reads
ASOS 6406 and 6405 files, combines their data,
and produces one netCDF4_CLASSIC file.

Dean Meyer 2021
"""

import io
import os
import datetime as dt
import netCDF4 as nc4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def asos_6406_to_df(file):
    """
    Read an ASOS 64060 file into a Pandas DataFrame.
    """
    # read the file and process it
    with open(file, "r") as f:
        lines = list(f)  # read each line of the file into a list

    # remove brackets if they exist
    lines = [l.replace("[", " ") for l in lines]
    lines = [l.replace("]", " ") for l in lines]

    # determine number of columns
    ncols = len(lines[0].split())

    # define column names
    if ncols == 9:
        cols = [
            "StationID",
            "DateTime",
            "Precip",
            "Unknown",
            "PrecipAmt",
            "FrzPrcpSnsrFreq",
            "Press1",
            "T",
            "Td",
        ]
    if ncols == 10:
        cols = [
            "StationID",
            "DateTime",
            "Precip",
            "Unknown",
            "PrecipAmt",
            "FrzPrcpSnsrFreq",
            "Press1",
            "Press2",
            "T",
            "Td",
        ]
    if ncols == 11:
        cols = [
            "StationID",
            "DateTime",
            "Precip",
            "Unknown",
            "PrecipAmt",
            "FrzPrcpSnsrFreq",
            "Press1",
            "Press2",
            "Press3",
            "T",
            "Td",
        ]

    # initialize the DataFrame
    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        names=cols,
        na_values=["M"],
        delim_whitespace=True,
        error_bad_lines=False,
    )

    # format datetimes and set them as the index
    df["DateTime"] = df["DateTime"].apply(format_asos_datetime)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.set_index("DateTime")

    # convert numeric columns
    df["PrecipAmt"] = pd.to_numeric(df["PrecipAmt"])
    df["FrzPrcpSnsrFreq"] = pd.to_numeric(df["FrzPrcpSnsrFreq"])
    df["Press1"] = pd.to_numeric(df["Press1"])
    df["Press2"] = pd.to_numeric(df["Press2"])
    df["T"] = pd.to_numeric(df["T"])
    df["Td"] = pd.to_numeric(df["Td"])

    return df


def asos_6405_to_df(file):
    """
    Read an ASOS 64050 file into a Pandas DataFrame.
    """
    # read the file and process it
    with open(file, "r") as f:
        lines = list(f)  # read each line of the file into a list

    # remove brackets if they exist
    lines = [l.replace("[", " ") for l in lines]
    lines = [l.replace("]", " ") for l in lines]

    # define column names
    cols = [
        "StationID",
        "DateTime",
        "ExtinctCoef",
        "DayNight",
        "Wdir2Min",
        "Wspd2Min",
        "Wdir5Sec",
        "Wspd5Sec",
    ]

    # initialize the DataFrame
    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        names=cols,
        index_col=False,
        na_values=["M"],
        delim_whitespace=True,
        error_bad_lines=False,
    )

    # format datetimes and set them as the index
    df["DateTime"] = df["DateTime"].apply(format_asos_datetime)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.set_index("DateTime")

    # convert numeric columns
    df["ExtinctCoef"] = pd.to_numeric(df["ExtinctCoef"])
    df["Wdir2Min"] = pd.to_numeric(df["Wdir2Min"])
    df["Wspd2Min"] = pd.to_numeric(df["Wspd2Min"])
    df["Wdir5Sec"] = pd.to_numeric(df["Wdir5Sec"])
    df["Wspd5Sec"] = pd.to_numeric(df["Wspd5Sec"])

    return df


def format_asos_datetime(x):
    """
    Format datetime columns in ASOS data.

    Retrieves the date and local(!!) time from the datetime column string.
    """
    return x[3:15]


def asos_to_nc4_classic(file_6405, file_6406, save_dir=None):
    """
    Read and combine ASOS 6405 and 6406 files into one DataFrame
    and then write a netCDF4_CLASSIC file.

    Method assumes data files share the same exact timeline.

    save_dir is the path to the directory the file will save in.

    Returns nothing and saves a .nc file.
    """
    # read files into DataFrames
    df1 = asos_6405_to_df(file_6405)
    df2 = asos_6406_to_df(file_6406)

    # ensure same station ID
    if df1["StationID"][0] != df2["StationID"][0]:
        raise ValueError("Station ID of files do not match!")

    # combine DFs into one
    # df = pd.merge(df1, df2, how='outer', on=df1.index)
    df = df1.join(df2, how="outer", lsuffix="_")

    # find filename from given file
    ncFilename = os.path.basename(file_6405).split(".")[0][5:]
    ncFilename = "".join([ncFilename, ".nc"])

    if save_dir is not None:
        fullpath = "/".join([save_dir, ncFilename])
    else:
        fullpath = ncFilename

    # open a netCDF4 file
    with nc4.Dataset(fullpath, "w", format="NETCDF4_CLASSIC") as f:

        # create dimensions
        time_dim = f.createDimension(
            "time", None
        )  # None means unlimited length

        # create variables
        time = f.createVariable("epochTime", np.float64, ("time"))
        time.units = "seconds"
        time.long_name = "Seconds Since 00 UTC 1970 01 01"

        ext = f.createVariable("extinct", np.float64, ("time"))
        ext.units = "unitless"
        ext.long_name = "Visibility extinction coefficient"

        dayNight = f.createVariable("DayNight", "S1", ("time"))
        dayNight.units = "unitless"
        dayNight.long_name = "Day/Night sensor: Day=D Night=N"

        wdir2min = f.createVariable("Wdir2Min", np.float64, ("time"))
        wdir2min.units = "degrees"
        wdir2min.long_name = "Direction of 2-minute average wind"

        wspd2min = f.createVariable("Wspd2Min", np.float64, ("time"))
        wspd2min.units = "knots"
        wspd2min.long_name = "Speed of 2-minute average wind"

        wdir5sec = f.createVariable("Wdir5Sec", np.float64, ("time"))
        wdir5sec.units = "degrees"
        wdir5sec.long_name = "Direction of max 5-second average wind"

        wspd5sec = f.createVariable("Wspd5Sec", np.float64, ("time"))
        wspd5sec.units = "knots"
        wspd5sec.long_name = "Speed of max 5-second average wind"

        precip = f.createVariable("Precip", "S1", ("time"))
        precip.units = "unitless"
        precip.long_name = "Precipitation ID: R=Rain S=Snow"

        precipAmt = f.createVariable("PrecipAmt", np.float64, ("time"))
        precipAmt.units = "hundredths of inches"
        precipAmt.long_name = "Precipitation amount"

        frzFreq = f.createVariable("FrzPrcpSnsrFreq", np.float32, ("time"))
        frzFreq.units = "frequency"
        frzFreq.long_name = "Frozen precipitation sensor frequency"

        press1 = f.createVariable("Pressure1", np.float32, ("time"))
        press1.units = "inches Hg"
        press1.long_name = "Station pressure from sensor 1"

        if "Press2" in df.columns:
            press2 = f.createVariable("Pressure2", np.float32, ("time"))
            press2.units = "inches Hg"
            press2.long_name = "Station pressure from sensor 2"

        if "Press3" in df.columns:
            press3 = f.createVariable("Pressure3", np.float32, ("time"))
            press3.units = "inches Hg"
            press3.long_name = "Station pressure from sensor 3"

        temp = f.createVariable("Temperature", np.int32, ("time"))
        temp.units = "degrees Fahrenheit"
        temp.long_name = "Average 1-minute dry bulb temperature"

        td = f.createVariable("Dewpoint Temperature", np.int32, ("time"))
        td.units = "degrees Fahrenheit"
        td.long_name = "Average 1-minute dewpoint temperature"

        # convert instrument time to seconds since the epoch 1970-01-01 0000Z
        epoch = dt.datetime(1970, 1, 1, 0, 0, 0)
        epoch = np.datetime64(epoch).astype("datetime64[s]")
        index_time = df.index.values.astype("datetime64[s]")
        sec_since_epoch = index_time - epoch

        # load data into variables
        time[:] = sec_since_epoch
        ext[:] = df["ExtinctCoef"]
        dayNight[:] = df["DayNight"]
        wdir2min[:] = df["Wdir2Min"]
        wspd2min[:] = df["Wspd2Min"]
        wdir5sec[:] = df["Wdir5Sec"]
        wspd5sec[:] = df["Wspd5Sec"]
        precip[:] = df["Precip"]
        precipAmt[:] = df["PrecipAmt"]
        frzFreq[:] = df["FrzPrcpSnsrFreq"]

        press1[:] = df["Press1"]
        if "Press2" in df.columns:
            press2[:] = df["Press2"]
        if "Press3" in df.columns:
            press3[:] = df["Press3"]

        temp[:] = df["T"]
        td[:] = df["Td"]

        # write metadata
        f.description = f"ASOS data for {df['StationID'][0]}"
        f.history = f"Created {dt.datetime.today()}"
