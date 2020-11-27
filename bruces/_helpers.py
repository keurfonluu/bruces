from datetime import datetime

import numpy

__all__ = [
    "to_decimal_year",
]


def to_decimal_year(dates):
    """
    Convert :class:`datetime.datetime` to decimal year.
    
    Parameters
    ----------
    dates : datetime.datetime, list or tuple
        Date time or list of date times.

    Returns
    -------
    scalar or list
        Decimal year or list of decimal years.
    
    """

    def decimal_year(d):
        """Convert a :class:`datetime.datetime` to decimal year."""
        if not isinstance(d, datetime):
            raise TypeError()

        year = d.year
        d1 = datetime(year, 1, 1)
        d2 = datetime(year + 1, 1, 1)

        return year + (d - d1).total_seconds() / (d2 - d1).total_seconds()

    if not isinstance(dates, (list, tuple, numpy.ndarray, datetime)):
        raise TypeError()
    if not isinstance(dates, datetime) and any(not isinstance(date, datetime) for date in dates):
        raise TypeError()

    return (
        decimal_year(dates)
        if isinstance(dates, datetime)
        else numpy.array([decimal_year(date) for date in dates])
    )
