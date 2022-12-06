from datetime import datetime

import numpy as np

from ..utils import to_datetime, to_decimal_year

__all__ = [
    "poisson_process",
]


def poisson_process(times, rates):
    """
    Simulate a (non)-stationary Poisson process.
    
    """
    if np.ndim(times) != 1:
        raise TypeError()

    is_datetime = isinstance(times[0], (datetime, np.datetime64))
    times = to_decimal_year(times)

    ndim = np.ndim(rates)
    if ndim == 0 and len(times) != 2:
        raise ValueError()

    elif ndim == 1 and len(times) != len(rates):
        raise ValueError()

    t = []
    ti = 0.0
    t0 = times[0]
    times = np.asarray(times) - t0

    # Homogeneous
    if ndim == 0:
        while True:
            ti -= np.log(np.random.rand()) / rates

            if ti > times[-1]:
                break

            t.append(ti)

    # Non-stationary
    elif ndim == 1:
        rates = np.asarray(rates)
        L = rates.max()

        while True:
            ti -= np.log(np.random.rand()) / L

            if ti > times[-1]:
                break

            if np.random.rand() * L < np.interp(ti, times, rates):
                t.append(ti)

    else:
        raise TypeError()

    t = np.array(t) + t0

    return to_datetime(t) if is_datetime else t
