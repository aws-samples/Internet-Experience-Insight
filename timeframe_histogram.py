import sys
sys.path.append("/Users/yemingcn/Desktop/my_code/timeframe_histogram/v1")


from copy import copy

import numpy as np
import pandas as pd
from dateutil import parser
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, ScalarFormatter
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

from utils import logger
from utils.util import (
    create_stat_function,
    format_ms,
    format_unix_timestamp,
    convert_datetime_field,
    convert_to_unix_timestamp,
    process_datetime_field,
    get_datetime_field,
    get_unix_time_field,
)

def common_histogram_setup(ax, title, xlabel, ylabel):
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(format_unix_timestamp)
    ax.yaxis.set_major_formatter(format_ms)

def plot_histogram(
    fig, ax, df, title, metric, ylim=None, yprecision=None, datetime_field=None, yscale="log"
):
    df = df.dropna(subset=[metric])

    datetime_field = datetime_field or get_unix_time_field(df.columns)

    df = process_datetime_field(df, convert_to_unix_timestamp, datetime_field)

    x = df[datetime_field].to_numpy()
    y = df[metric].to_numpy()

    # y-axis limits
    if not ylim:
        ylim = [0.001, min(100, np.max(y))]
    ax.set_ylim(ylim)

    if yprecision is not None:
        ynum = min((ylim[1] - ylim[0]) / yprecision, 1000)  # max 1000 ybins
    else:
        # No precision specified, just use 1000 ybins.
        # TODO: we may want to guess as to the precision of the data if possible.
        ynum = 1000
        # Set the precision to the min ylim to avoid hitting the precison floor below.
        # yprecision = ylim[0]
        # TODO - that seems to be broken.  We need better auto-precision.
        yprecision = 1
    ylogbins = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), num=int(ynum))
    # Now, we must take into account precision:
    # Since precision is typically fixed (precision does not
    # scale-up or scale-down depending on value measured) we have a "floor"
    # on bin size limited by the precision. Split the logbins into a
    # "linear" part and "log" part if the lower bound on precision is
    # violated:
    logpart = ylogbins[~(np.diff(ylogbins, prepend=[True]) < yprecision)]
    if logpart.size > 0:
        linpart = np.linspace(
            ylim[0],
            logpart.min() - yprecision,
            int(np.floor((logpart.min() - ylim[0]) / yprecision)),
        )
        ylogbins = np.concatenate([linpart, logpart])
    else:
        ylogbins = np.linspace(ylim[0], ylim[1], num=int(ynum))

    xnum = min(800, np.max(x) - np.min(x))
    xbins = np.linspace(np.min(x), np.max(x), num=int(xnum))

    # 2D histogram
    h, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ylogbins])
    cmap = copy(plt.cm.plasma)
    pcm = ax.pcolormesh(
        xedges, yedges, h.T, cmap=cmap, norm=LogNorm(vmin=ylim[0], vmax=ylim[1]), rasterized=True
    )
    try:
        fig.colorbar(pcm, ax=ax, pad=0)
    except Exception as e:
        logger.log_err(f"execption e = {e}")

    ax.set_yscale(yscale)
    common_histogram_setup(ax, title, xlabel="Time", ylabel=metric)

def plot_timestamp_vline(ax, ts, color="r"):
    vline = pd.to_datetime(ts).timestamp()
    ax.axvline(vline, color=color)

def plot_metrics(
    ax,
    df,
    metric,
    resample_size="1min",
    stats=["P50", "P90", "P99", "TM99", "n"],
    datetime_field=None,
    label=None,
    group_by=None,
):
    datetime_field = datetime_field or get_datetime_field(df.columns)
    df = process_datetime_field(df, convert_datetime_field, datetime_field)


    if group_by is not None:
        grouped_df = df.groupby(group_by)
    else:
        grouped_df = [("", df)]

    for group_name, group in grouped_df:
        current_label = f"{label} {group_name}" if label else group_name

        r = group.dropna(subset=[metric]).resample(resample_size, on=datetime_field)
        agg_dict = {metric: []}
        for stat in stats:
            if stat == "n":
                stat = "size"
            agg_dict[metric].append((stat.lower(), create_stat_function(stat)))

        tm99_data = r.agg(agg_dict)
        tm99_data.columns = tm99_data.columns.get_level_values(1)

        for stat in stats:
            if stat == "n":
                ax2 = ax.twinx()
                ax2.plot(
                    tm99_data.index,
                    tm99_data["size"],
                    color="aqua",
                    linewidth=3,
                    label=f"{current_label} Count",
                )
                ax2.set_ylim([0, np.max(tm99_data["size"]) + np.max(tm99_data["size"]) * 0.1])
                ax2.legend(loc="upper right")
                ax2.ticklabel_format(style="plain", axis="y")
            else:
                ax.plot(
                    tm99_data.index,
                    tm99_data[stat.lower()],
                    linewidth=3,
                    label=f"{current_label} {stat}",
                )
                ax.ticklabel_format(style="plain", axis="y")

    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M:%S"))

def plot_metrics_1(
    ax,
    df,
    metric,
    resample_size="1min",
    stats=["P50", "P90", "P99", "TM99", "n"],
    datetime_field=None,
    label=None,
    group_by=None,
):
    datetime_field = datetime_field or get_datetime_field(df.columns)
    df = process_datetime_field(df, convert_datetime_field, datetime_field)

    if group_by is not None:
        grouped_df = df.groupby(group_by)
    else:
        grouped_df = [("", df)]

    for group_name, group in grouped_df:
        current_label = f"{label} {group_name}" if label else group_name

        r = group.dropna(subset=[metric]).resample(resample_size, on=datetime_field)
        agg_dict = {metric: []}
        for stat in stats:
            if stat == "n":
                stat = "size"
            agg_dict[metric].append((stat.lower(), create_stat_function(stat)))

        tm99_data = r.agg(agg_dict)
        tm99_data.columns = tm99_data.columns.get_level_values(1)

        for stat in stats:
            if stat == "n":
                ax2 = ax.twinx()
                ax2.plot(
                    tm99_data.index,
                    tm99_data["size"],
                    color="aqua",
                    linewidth=3,
                    alpha=0.7,
                    label=f"{current_label} Count",
                )
                ax2.set_ylim([0, np.max(tm99_data["size"]) + np.max(tm99_data["size"]) * 0.1])
                ax2.legend(loc="upper right")
                ax2.ticklabel_format(style="plain", axis="y")
            else:
                ax.plot(
                    tm99_data.index,
                    tm99_data[stat.lower()],
                    linewidth=3,
                    alpha=0.7,
                    label=f"{current_label} {stat}",
                )
                # ax.ticklabel_format(style="plain", axis="y")

    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M:%S"))

def plot_metrics_2(fig,ax,df,metric,resample_size, stats=['P50','P90','P99','TM99','n']):
    # compute TM99 and other stats, then plot directly on the histogram.
    r = df.dropna(subset=[metric]).resample(resample_size, on='ts')
    tm99_data = r[metric].agg([tm99,p50,p90,p99,size])
    tm99_data['TimeEpoch'] = (pd.to_datetime(tm99_data.index) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
    for stat in stats:
        if stat == 'P50':
            ax.plot(tm99_data['TimeEpoch'],tm99_data['p50'], color='red',linewidth=3,label="P50")
        elif stat == 'P90':
            ax.plot(tm99_data['TimeEpoch'],tm99_data['p90'], color='orange',linewidth=3,label="P90")
        elif stat == 'P99':
            ax.plot(tm99_data['TimeEpoch'],tm99_data['p99'], color='yellow',linewidth=3,label="P99")
        elif stat == 'TM99':
            ax.plot(tm99_data['TimeEpoch'],tm99_data['tm99'],color='chartreuse',linewidth=3,label="TM99")
        elif stat == 'n':
            ax2 = ax.twinx()
            ax2.plot(tm99_data['TimeEpoch'],tm99_data['size'], color='aqua',linewidth=3,label="n")
            ax2.set_ylim([0,np.max(tm99_data['size'])])
            ax2.legend(loc='upper right')
    ax.legend(loc='upper left')