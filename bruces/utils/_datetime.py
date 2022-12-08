from datetime import datetime, timedelta

import numpy as np

__all__ = [
    "to_datetime",
    "to_decimal_year",
]


_scalar_like = (int, np.int32, np.int64, float, np.float32, np.float64)


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
        if isinstance(d, datetime):
            return d

        elif isinstance(d, np.datetime64):
            return d.astype("M8[ms]")

        elif not isinstance(d, _scalar_like):
            raise TypeError()

        year = np.floor(d)
        days = (d - year) * (366.0 if isleap(year) else 365.0)

        return datetime(int(year), 1, 1) + timedelta(days=days)

    ndim = np.ndim(dates)

    return datetime_(dates) if ndim == 0 else [datetime_(date) for date in dates]


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

        elif isinstance(d, _scalar_like):
            return d

        else:
            raise TypeError()

        d1 = datetime(year, 1, 1)
        d2 = datetime(year + 1, 1, 1)

        return (
            year
            + (d.replace(tzinfo=None) - d1).total_seconds() / (d2 - d1).total_seconds()
        )

    ndim = np.ndim(dates)

    return (
        decimal_year(dates)
        if ndim == 0
        else np.array([decimal_year(date) for date in dates])
    )
