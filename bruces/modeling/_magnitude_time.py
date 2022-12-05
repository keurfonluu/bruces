import numpy as np

from .._common import jitted, set_seed
from ..utils import to_decimal_year

__all__ = [
    "magnitude_time",
]


@jitted
def rate2mag(dt, r, m, bm):
    """Generate magnitude samples for current time step."""
    dmdt = (m[1] - m[0]) * dt

    # Calculate rates per unit time per unit magnitude
    rtm = r * bm

    # Calculate total number of earthquakes
    neq = rtm.sum() * dmdt

    if neq > 0.0:
        nm = len(m)

        # Calculate cumulative distribution function
        cdf = np.empty(nm, dtype=np.float64)
        for i in range(nm):
            cdf[i] = 1.0 - rtm[i:].sum() * dmdt / neq

        # Sample neq magnitudes from CDF
        mout = np.interp(np.random.rand(int(neq)), cdf, m)

        return mout

    else:
        return np.empty(0)


@jitted
def rate2mag_vectorized(dt, r, m, b):
    """Generate magnitude samples for every time steps."""
    nm = len(m)
    nr = len(r)

    # Calculate Gutenberg-Richter rate scaling factor
    blg10 = b * np.log(10.0)
    bm = np.empty(nm, dtype=np.float64)
    for i in range(nm):
        bm[i] = blg10 * 10.0 ** (-b * (m[i] - m[0]))

    # Loop over time steps
    out = []
    for i in range(nr):
        out.append(rate2mag(dt[i], r[i], m, bm))

    return out


def magnitude_time(times, rates, m_bounds, n=50, b_value=1.0, seed=None):
    """
    Generate earthquake magnitude-time distribution.

    Parameters
    ----------
    times : sequence of scalar or sequence of datetime_like
        Dates for every seismicity rate samples (in years if scalar).
    rates : array_like
        Seismicity rates.
    m_bounds : array_like
        Magnitude bounds (support for CDF from which magnitude values are sampled).
    n : int, optional, default 50
        Number of magnitude points to define CDF.
    b_value : scalar, optional, default 1.0
        b-value.
    seed : int or None, optional, default None
        Seed for random number generator.

    Returns
    -------
    list
        List of magnitude samples for every time steps.

    """
    # Check seismicity rate
    r = np.asarray(rates)
    if r.ndim != 1 or len(times) != r.size:
        raise ValueError()

    # Check magnitude bounds
    if len(m_bounds) != 2 or m_bounds[0] - m_bounds[1] >= 0.0:
        raise ValueError()
    m = np.linspace(m_bounds[0], m_bounds[1], n)

    # Convert datetimes to decimal years
    t = to_decimal_year(times)

    # Convert times to time increments for integration
    dt = np.diff(t)
    dt = np.append(dt, dt[-1])

    # Set random seed
    if seed is not None:
        set_seed(seed)

    return rate2mag_vectorized(dt, r, m, b_value)
