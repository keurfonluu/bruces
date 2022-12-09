from datetime import datetime, timedelta

import numpy as np

from ..._catalog import Catalog
from ..._common import open_file


_required_columns = ["year", "month", "day", "hour", "minute", "second"]


def read(filename):
    """
    Read catalog from CSV file.

    Parameters
    ----------
    filename : str, pathlike or buffer
        Input file name or buffer.

    Returns
    -------
    :class:`bruces.Catalog`
        Earthquake catalog.

    """
    with open_file(filename, "r") as f:
        out = read_buffer(f)

    return out


def read_buffer(f):
    """Read CSV catalog."""
    # Check header
    columns = f.readline().strip().lower().split(",")
    if len(set(_required_columns).intersection(columns)) < 6:
        raise ValueError()

    # Read tabular data
    data = [
        [to_float(x) for x in line.strip().split(",")] for line in f
    ]
    data = {k: v for k, v in zip(columns, np.transpose(data))}

    # Origin times
    origin_times = [
        datetime(
            year=int(Y),
            month=int(M),
            day=int(D),
            hour=int(h),
            minute=int(m),
        ) + timedelta(seconds=s)
        for Y, M, D, h, m, s in zip(*[data[k] for k in _required_columns])
    ]

    return Catalog(
        origin_times=origin_times,
        latitudes=process_data(data, "latitude"),
        longitudes=process_data(data, "longitude"),
        eastings=process_data(data, "easting"),
        northings=process_data(data, "northing"),
        depths=process_data(data, "depth"),
        magnitudes=process_data(data, "magnitude"),
    )


def to_float(x):
    """Convert string to float."""
    try:
        return float(x.strip())

    except ValueError:
        return np.nan


def process_data(data, key):
    """Process optional catalog inputs."""
    x = data[key] if key in data else None

    return None if x is not None and np.isnan(x).any() else x
