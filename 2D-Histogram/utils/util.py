import pandas as pd
import re
import psutil
from typing import List, Tuple, Union, cast
from dateutil import parser
from scipy import stats


def create_stat_function(stat):
    if stat.lower().startswith("p"):
        percentile = float(stat[1:]) / 100
        return lambda x: x.quantile(percentile)
    elif stat.lower().startswith("tm"):
        percentile = float(100 - float(stat[2:])) / 100
        return lambda x: stats.trim_mean(x, percentile)
    elif stat.lower() in ["mean", "sum", "max", "min", "median", "std", "var", "size"]:
        return stat.lower()
    else:
        raise ValueError(f"Unsupported statistic: {stat}")

def format_ms(y, pos=None):
    return "{:10.3f}".format(y)

def format_unix_timestamp(x, pos=None):
    ts = pd.to_datetime(x, unit="s")
    return ts.strftime("%Y-%m-%d %H:%M:%S")

def convert_datetime_field(timefield_data):
    if pd.api.types.is_numeric_dtype(timefield_data):
        return pd.to_datetime(timefield_data, unit="s")

    elif pd.api.types.is_string_dtype(timefield_data):
        return timefield_data.apply(lambda x: parser.parse(x))

    elif pd.api.types.is_datetime64_any_dtype(timefield_data):
        return timefield_data
        

def convert_to_unix_timestamp(timefield_data):
    if pd.api.types.is_numeric_dtype(timefield_data):
        return timefield_data

    elif pd.api.types.is_string_dtype(timefield_data):
        timefield_data = timefield_data.apply(lambda x: parser.parse(x))

    if not pd.api.types.is_datetime64_any_dtype(timefield_data):
        timefield_data = pd.to_datetime(timefield_data)

    return timefield_data.astype("int64") // 10**9

def process_datetime_field(df, convert_func, datetime_field=None):
    if datetime_field:
        df.loc[:, datetime_field] = convert_func(df[datetime_field])
    else:
        raise ValueError(
            "No suitable datetime field found in the DataFrame. "
            "Ensure that your DataFrame includes a column with datetime information. "
            "This column should contain dates/times in a recognizable format such as a Python datetime object, "
            "a string representing a date/time, or a Unix timestamp. "
            "Please pass a 'datetime_field' parameter as shown below when calling the function: "
            "function_you_are_calling(convert_func, datetime_field='your_datetime_column_name', other_parameters)"
        )
    return df

def get_datetime_field(columns):
    for field in ["DateTime", "timestamp", "TimeEpoch"]:
        if field in columns:
            return field
    return None

def get_unix_time_field(columns):
    for field in ["timestamp", "TimeEpoch", "DateTime"]:
        if field in columns:
            return field