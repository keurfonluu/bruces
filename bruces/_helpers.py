from datetime import datetime, timedelta

import numpy as np

__all__ = [
    "from_csep",
    "to_decimal_year",
    "to_datetime",
]


_datetime_like = (datetime, np.datetime64)


def from_csep(catalog):
    """
    Convert a :class:`csep.Catalog` to :class:`bruces.Catalog`.

    Parameters
    ----------
    catalog : :class:`csep.Catalog`
        Input catalog.

    Returns
    -------
    :class:`bruces.Catalog`
        Output catalog.

    """
    from ._catalog import Catalog

    return Catalog(
        origin_times=catalog.get_datetimes(),
        latitudes=catalog.get_latitudes(),
        longitudes=catalog.get_longitudes(),
        depths=catalog.get_depths(),
        magnitudes=catalog.get_magnitudes(),
    )


def to_decimal_year(dates):
    """
    Convert datetime_like to decimal year.

    Parameters
    ----------
    dates : datetime_like or sequence of datetime_like
        Date time or list of date times.

    Returns
    -------
    scalar or list
        Decimal year or list of decimal years.

    """

    def decimal_year(d):
        """Convert a datetime_like to decimal year."""
        if isinstance(d, datetime):
            year = d.year
        elif isinstance(d, np.datetime64):
            d = d.astype("M8[ms]").tolist()
            year = d.year
        else:
            raise TypeError()

        d1 = datetime(year, 1, 1)
        d2 = datetime(year + 1, 1, 1)

        return (
            year
            + (d.replace(tzinfo=None) - d1).total_seconds() / (d2 - d1).total_seconds()
        )

    ndim = np.ndim(dates)
    if ndim == 0:
        if not isinstance(dates, _datetime_like):
            raise TypeError()
    elif ndim == 1:
        if any(not isinstance(date, _datetime_like) for date in dates):
            raise TypeError()
    else:
        raise TypeError()

    return (
        decimal_year(dates)
        if ndim == 0
        else np.array([decimal_year(date) for date in dates])
    )


def to_datetime(dates):
    """
    Convert decimal year to datetime.

    Parameters
    ----------
    dates : scalar or array_like
        Decimal year or list of decimal years.

    Returns
    -------
    datetime or sequence of datetime
        Date time or list of date times.
    
    """
    from calendar import isleap

    def datetime_(d):
        """Convert a decimal year to datetime_like."""
        year = np.floor(d)
        days = (d - year) * (366.0 if isleap(year) else 365.0)

        return datetime(int(year), 1, 1) + timedelta(days=days)

    ndim = np.ndim(dates)
    
    return (
        datetime_(dates)
        if ndim == 0
        else np.array([datetime_(date) for date in dates])
    )
