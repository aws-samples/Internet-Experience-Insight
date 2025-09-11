import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))


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

def calculate_precision_aware_bins(ylim, ynum, yprecision):
    """Calculate y-axis bins with precision awareness."""
    ylogbins = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), num=int(ynum))
    logpart = ylogbins[~(np.diff(ylogbins, prepend=[True]) < yprecision)]
    if logpart.size > 0:
        linpart = np.linspace(
            ylim[0],
            logpart.min() - yprecision,
            int(np.floor((logpart.min() - ylim[0]) / yprecision)),
        )
        return np.concatenate([linpart, logpart])
    else:
        return np.linspace(ylim[0], ylim[1], num=int(ynum))

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
    ylogbins = calculate_precision_aware_bins(ylim, ynum, yprecision)

    x_min, x_max = np.min(x), np.max(x)
    xnum = min(800, x_max - x_min)
    xbins = np.linspace(x_min, x_max, num=int(xnum))

    # 2D histogram
    h, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ylogbins])
    cmap = copy(plt.cm.plasma)
    pcm = ax.pcolormesh(
        xedges, yedges, h.T, cmap=cmap, norm=LogNorm(vmin=ylim[0], vmax=ylim[1]), rasterized=True
    )
    try:
        fig.colorbar(pcm, ax=ax, pad=0)
    except (ValueError, RuntimeError) as e:
        logger.log_err(f"Failed to create colorbar: {type(e).__name__}: {e}")

    ax.set_yscale(yscale)
    common_histogram_setup(ax, title, xlabel="Time", ylabel=metric)

def plot_timestamp_vline(ax, ts, color="r"):
    vline = pd.to_datetime(ts).timestamp()
    ax.axvline(vline, color=color)

def _process_group_data(group, metric, resample_size, datetime_field, stats):
    """Helper function to process group data and create aggregations."""
    r = group.dropna(subset=[metric]).resample(resample_size, on=datetime_field)
    agg_dict = {metric: []}
    for stat in stats:
        stat_name = "size" if stat == "n" else stat
        agg_dict[metric].append((stat_name.lower(), create_stat_function(stat_name)))
    
    tm99_data = r.agg(agg_dict)
    tm99_data.columns = tm99_data.columns.get_level_values(1)
    return tm99_data

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

    ax2 = None
    for group_name, group in grouped_df:
        current_label = f"{label} {group_name}" if label else group_name
        tm99_data = _process_group_data(group, metric, resample_size, datetime_field, stats)

        for stat in stats:
            if stat == "n":
                if ax2 is None:
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

def _prepare_aggregated_data(group, metric, resample_size, datetime_field, stats):
    """Helper function to prepare aggregated data for plotting."""
    r = group.dropna(subset=[metric]).resample(resample_size, on=datetime_field)
    agg_dict = {metric: []}
    for stat in stats:
        stat_name = "size" if stat == "n" else stat
        agg_dict[metric].append((stat_name.lower(), create_stat_function(stat_name)))
    
    tm99_data = r.agg(agg_dict)
    tm99_data.columns = tm99_data.columns.get_level_values(1)
    return tm99_data

def _plot_stat_data(ax, ax2, tm99_data, stat, current_label):
    """Helper function to plot individual stat data."""
    if stat == "n":
        if ax2 is None:
            ax2 = ax.twinx()
        ax2.plot(
            tm99_data.index,
            tm99_data["size"],
            color="aqua",
            linewidth=3,
            alpha=0.7,
            label=f"{current_label} Count",
        )
        max_size = np.max(tm99_data["size"])
        ax2.set_ylim([0, max_size + max_size * 0.1])
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
    return ax2

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

    ax2 = None
    for group_name, group in grouped_df:
        current_label = f"{label} {group_name}" if label else group_name
        tm99_data = _prepare_aggregated_data(group, metric, resample_size, datetime_field, stats)

        for stat in stats:
            ax2 = _plot_stat_data(ax, ax2, tm99_data, stat, current_label)

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