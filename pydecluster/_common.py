from datetime import datetime

from numba import jit

import numpy

__all__ = [
    "to_decimal_year",
]


def jitted(*args, **kwargs):
    """Custom :func:`jit` with default options."""
    kwargs.update(
        {
            "nopython": True,
            "nogil": True,
            "fastmath": True,
            # "boundscheck": False,
            "cache": True,
        }
    )
    return jit(*args, **kwargs)


@jitted
def proximity(t, x, y, m, ti, xi, yi, d=1.5, w=0.0):
    """Calculate nearest-neighbor proximity."""
    N = len(t)

    eta_i = 1.0e20
    for j in range(N):
        t_ij = t[j] - ti

        # For each event, we are looking for its parent which corresponds
        # to the earliest event with the smallest proximity value
        if t_ij < 0.0:
            r_ij = ((x[j] - xi) ** 2 + (y[j] - yi) ** 2) ** 0.5

            # Skip events with the same epicenter
            if r_ij > 0.0:
                eta_ij = -t_ij * (r_ij * 1.0e-3) ** d
                if w > 0.0:
                    eta_ij *= 10.0 ** (-w * m[j])
                eta_i = min(eta_i, eta_ij)

    return eta_i


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

    if not isinstance(dates, (list, tuple)):
        raise TypeError()
    if any(not isinstance(date, datetime) for date in dates):
        raise TypeError()

    return (
        decimal_year(dates)
        if isinstance(dates, datetime)
        else numpy.array([decimal_year(date) for date in dates])
    )
