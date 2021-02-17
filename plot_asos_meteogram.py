"""
Plotting ASOS netCDF4_CLASSIC data.

Inspired by UNIDATA MetPy tutorial at 
https://unidata.github.io/MetPy/latest/examples/meteogram_metpy.html

Dean Meyer 2021
"""

import os
# import netCDF4 as nc4
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# Python standard open example
# with nc4.Dataset('/'.join([save_dir, 'KHSV201711.nc']),'r') as f:
#     var = 'epochTime'
#     print(f.variables.keys())
#     print('\n')
#     print(f.variables[var])
#     print('\n')
#     print(f.variables[var][:])

"""
UNIDATA METEOGRAM PLOT
"""
import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from metpy.calc import dewpoint_from_relative_humidity
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo
from metpy.units import units


def calc_mslp(t, p, h):
    return p * (1 - (0.0065 * h) / (t + 0.0065 * h + 273.15)) ** (-5.257)


# Make meteogram plot
class Meteogram:
    """Plot a time series of meteorological data from a particular station as a
    meteogram with standard variables to visualize, including thermodynamic,
    kinematic, and pressure. The functions below control the plotting of each
    variable.
    TO DO: Make the subplot creation dynamic so the number of rows is not
    static as it is currently."""

    def __init__(self, fig, dates, probeid, time=None, axis=0):
        """
        Required input:
            fig: figure object
            dates: array of dates corresponding to the data
            probeid: ID of the station
        Optional Input:
            time: Time the data is to be plotted
            axis: number that controls the new axis to be plotted (FOR FUTURE)
        """
        if not time:
            time = dt.datetime.utcnow()
        self.start = dates[0]
        self.fig = fig
        self.end = dates[-1]
        self.axis_num = 0
        self.dates = mpl.dates.date2num(dates)
        self.time = time.strftime("%Y-%m-%d %H:%M UTC")
        self.title = f"Latest Ob Time: {self.time}\nProbe ID: {probeid}"

    def plot_winds(self, ws, wd, wsmax, plot_range=None):
        """
        Required input:
            ws: Wind speeds (knots)
            wd: Wind direction (degrees)
            wsmax: Wind gust (knots)
        Optional Input:
            plot_range: Data range for making figure (list of (min,max,step))
        """
        # PLOT WIND SPEED AND WIND DIRECTION
        self.ax1 = fig.add_subplot(4, 1, 1)
        ln1 = self.ax1.plot(self.dates, ws, label="Wind Speed")
        self.ax1.fill_between(self.dates, ws, 0)
        self.ax1.set_xlim(self.start, self.end)
        if not plot_range:
            plot_range = [0, 40, 1]
        self.ax1.set_ylabel("Wind Speed (knots)", multialignment="center")
        self.ax1.set_ylim(plot_range[0], plot_range[1], plot_range[2])
        self.ax1.grid(
            b=True,
            which="major",
            axis="y",
            color="k",
            linestyle="--",
            linewidth=0.5,
        )
        ln2 = self.ax1.plot(
            self.dates, wsmax, ".r", label="5-sec Wind Speed Max"
        )

        ax7 = self.ax1.twinx()
        ln3 = ax7.plot(
            self.dates, wd, ".k", linewidth=0.5, label="Wind Direction"
        )
        ax7.set_ylabel("Wind\nDirection\n(degrees)", multialignment="center")
        ax7.set_ylim(0, 360)
        ax7.set_yticks(np.arange(45, 405, 90), ["NE", "SE", "SW", "NW"])
        lines = ln1 + ln2 + ln3
        labs = [line.get_label() for line in lines]
        ax7.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d/%H"))
        ax7.legend(
            lines,
            labs,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2),
            ncol=3,
            prop={"size": 12},
        )

    def plot_thermo(self, t, td, plot_range=None):
        """
        Required input:
            T: Temperature (deg F)
            TD: Dewpoint (deg F)
        Optional Input:
            plot_range: Data range for making figure (list of (min,max,step))
        """
        # PLOT TEMPERATURE AND DEWPOINT
        if not plot_range:
            plot_range = [10, 90, 2]
        self.ax2 = fig.add_subplot(4, 1, 2, sharex=self.ax1)
        ln4 = self.ax2.plot(self.dates, t, "r-", label="Temperature")
        self.ax2.fill_between(self.dates, t, td, color="r")

        self.ax2.set_ylabel("Temperature\n(F)", multialignment="center")
        self.ax2.grid(
            b=True,
            which="major",
            axis="y",
            color="k",
            linestyle="--",
            linewidth=0.5,
        )
        self.ax2.set_ylim(plot_range[0], plot_range[1], plot_range[2])

        ln5 = self.ax2.plot(self.dates, td, "g-", label="Dewpoint")
        self.ax2.fill_between(
            self.dates, td, self.ax2.get_ylim()[0], color="g"
        )

        ax_twin = self.ax2.twinx()
        ax_twin.set_ylim(plot_range[0], plot_range[1], plot_range[2])
        lines = ln4 + ln5
        labs = [line.get_label() for line in lines]
        ax_twin.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d/%H UTC"))

        self.ax2.legend(
            lines,
            labs,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2),
            ncol=2,
            prop={"size": 12},
        )

    def plot_rh(self, rh, plot_range=None):
        """
        Required input:
            RH: Relative humidity (%)
        Optional Input:
            plot_range: Data range for making figure (list of (min,max,step))
        """
        # PLOT RELATIVE HUMIDITY
        if not plot_range:
            plot_range = [0, 100, 4]
        self.ax3 = fig.add_subplot(4, 1, 3, sharex=self.ax1)
        self.ax3.plot(self.dates, rh, "g-", label="Relative Humidity")
        self.ax3.legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.22), prop={"size": 12}
        )
        self.ax3.grid(
            b=True,
            which="major",
            axis="y",
            color="k",
            linestyle="--",
            linewidth=0.5,
        )
        self.ax3.set_ylim(plot_range[0], plot_range[1], plot_range[2])

        self.ax3.fill_between(
            self.dates, rh, self.ax3.get_ylim()[0], color="g"
        )
        self.ax3.set_ylabel("Relative Humidity\n(%)", multialignment="center")
        self.ax3.xaxis.set_major_formatter(
            mpl.dates.DateFormatter("%d/%H UTC")
        )
        axtwin = self.ax3.twinx()
        axtwin.set_ylim(plot_range[0], plot_range[1], plot_range[2])

    def plot_pressure(self, p, plot_range=None):
        """
        Required input:
            P: Mean Sea Level Pressure (hPa)
        Optional Input:
            plot_range: Data range for making figure (list of (min,max,step))
        """
        # PLOT PRESSURE
        if not plot_range:
            plot_range = [970, 1030, 2]
        self.ax4 = fig.add_subplot(4, 1, 4, sharex=self.ax1)
        self.ax4.plot(self.dates, p, "m", label="Mean Sea Level Pressure")
        self.ax4.set_ylabel(
            "Mean Sea\nLevel Pressure\n(mb)", multialignment="center"
        )
        self.ax4.set_ylim(plot_range[0], plot_range[1], plot_range[2])

        axtwin = self.ax4.twinx()
        axtwin.set_ylim(plot_range[0], plot_range[1], plot_range[2])
        axtwin.fill_between(self.dates, p, axtwin.get_ylim()[0], color="m")
        axtwin.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d/%H UTC"))

        self.ax4.legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.2), prop={"size": 12}
        )
        self.ax4.grid(
            b=True,
            which="major",
            axis="y",
            color="k",
            linestyle="--",
            linewidth=0.5,
        )
        # OTHER OPTIONAL AXES TO PLOT
        # plot_irradiance
        # plot_precipitation


file = r"/nas/rstor/dmeyer/GRA/ASOS/KHSV201711.nc"
# file = r'/nas/rstor/dmeyer/GRA/ASOS/KDCU201711.nc'
# file = r'/nas/rstor/dmeyer/GRA/ASOS/KMSL201711.nc'

# xarray example
ds = xr.open_dataset(
    file, decode_timedelta=False
)  # dont decode times from s into ns(!!)

# convert xarray ds to df example
df = ds.to_dataframe()
df["time"] = pd.to_datetime(df["epochTime"], unit="s")
df = df.set_index("time", drop=True)

# cut DataFrame
df = df.loc["2017-11-18 16:30:00":"2017-11-18 18:30:00"]

# Height of the station to calculate MSLP.
hgt = 175.0  # meters? Average between Decatur and Huntsville AL

# Temporary variables for ease
temp = df["Temperature"]
td = df["Dewpoint Temperature"]
pres = df["Pressure1"]
ws = df["Wspd2Min"]
wsmax = df["Wspd5Sec"]
wd = df["Wdir2Min"]
date = df.index.values

# ID For Plotting on Meteogram
probe_id = ds.attrs["description"]

data = {
    "wind_speed": np.array(ws) * units("knots"),
    "wind_speed_max": np.array(wsmax) * units("knots"),
    "wind_direction": np.array(wd) * units("degrees"),
    "dewpoint": np.array(td) * units("degF"),
    "air_temperature": np.array(temp) * units("degF"),
    "mean_slp": calc_mslp(
        np.array(temp), (np.array(pres) * units("inHg")).to(units("hPa")), hgt
    )
    * units("hPa"),
    "relative_humidity": np.full_like(
        np.array(temp), np.nan
    ),  # array of NaNs as a placeholder
    "times": np.array(date),
}

fig = plt.figure(figsize=(15, 14))
meteogram = Meteogram(fig, data["times"], probe_id)
meteogram.plot_winds(
    data["wind_speed"],
    data["wind_direction"],
    data["wind_speed_max"],
    plot_range=[0, 60, 10],
)
meteogram.plot_thermo(
    data["air_temperature"], data["dewpoint"], plot_range=[50, 80, 2]
)
meteogram.plot_rh(data["relative_humidity"], plot_range=[0, 100, 20])
meteogram.plot_pressure(data["mean_slp"], plot_range=[994, 1007, 2])
fig.subplots_adjust(hspace=0.5)

fname = os.path.basename(file).split(".")[0]
savefile = "/".join(["/nas/rstor/dmeyer/GRA/ASOS", fname + "_meteogram.png"])

plt.suptitle(fname)

# plt.savefig(savefile, dpi=300)

# import hvplot.xarray
