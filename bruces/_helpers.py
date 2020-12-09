from datetime import datetime

import numpy

__all__ = [
    "to_decimal_year",
]


_datetime_like = (datetime, numpy.datetime64)


def to_decimal_year(dates):
    """
    Convert :class:`datetime.datetime` or :class:`numpy.datetime64` to decimal year.
    
    Parameters
    ----------
    dates : datetime.datetime or numpy.datetime64, list or tuple
        Date time or list of date times.

    Returns
    -------
    scalar or list
        Decimal year or list of decimal years.
    
    """

    def decimal_year(d):
        """Convert a :class:`datetime.datetime` or :class:`numpy.datetime64` to decimal year."""
        if isinstance(d, datetime):
            year = d.year
        elif isinstance(d, numpy.datetime64):
            d = d.astype("M8[ms]").tolist()
            year = d.year
        else:
            raise TypeError()

        d1 = datetime(year, 1, 1)
        d2 = datetime(year + 1, 1, 1)

        return year + (d - d1).total_seconds() / (d2 - d1).total_seconds()

    ndim = numpy.ndim(dates)
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
        else numpy.array([decimal_year(date) for date in dates])
    )
