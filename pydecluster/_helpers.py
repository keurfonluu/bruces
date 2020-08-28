from datetime import datetime

import numpy

__all__ = [
    "declusterize",
    "to_decimal_year",
]


_decluster_map = {}


def register(algorithm, declusterize):
    """Register a new declustering algorithm."""
    _decluster_map[algorithm] = declusterize


def declusterize(catalog, algorithm="nearest-neighbor", **kwargs):
    """
    Declusterize earthquake catalog.

    Parameters
    ----------
    catalog : pydecluster.Catalog
        Earthquake catalog.
    algorithm : str, optional, default 'nearest-neighbor'
        Declustering algorithm:
         - 'nearest-neighbor': nearest-neighbor algorithm (after Zaliapin and Ben-Zion, 2020).
    
    Other Parameters
    ----------------
    d : scalar, optional, default 1.5
        Only if ``algorithm = "nearest-neighbor"``. Fractal dimension of epicenter/hypocenter.
    w : scalar, optional, default 0.0
        Only if ``algorithm = "nearest-neighbor"``. Magnitude weighing factor (usually b-value).
    eta_0 : scalar, optional, default 0.1
        Only if ``algorithm = "nearest-neighbor"``. Initial cutoff threshold.
    alpha_0 : scalar, optional, default 0.1
        Only if ``algorithm = "nearest-neighbor"``. Cluster threshold.
    M : int, optional, default 100
        Only if ``algorithm = "nearest-neighbor"``. Number of reshufflings.

    Returns
    -------
    pydecluster.Catalog
        Declustered earthquake catalog.

    """
    return _decluster_map[algorithm](catalog, **kwargs)


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

    if not isinstance(dates, (list, tuple, datetime)):
        raise TypeError()
    if not isinstance(dates, datetime) and any(not isinstance(date, datetime) for date in dates):
        raise TypeError()

    return (
        decimal_year(dates)
        if isinstance(dates, datetime)
        else numpy.array([decimal_year(date) for date in dates])
    )
