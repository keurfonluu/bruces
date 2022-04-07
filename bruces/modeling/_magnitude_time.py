import numpy

from .._common import jitted
from .._helpers import to_decimal_year

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
        cdf = numpy.empty(nm, dtype=numpy.float64)
        for i in range(nm):
            cdf[i] = 1.0 - rtm[i:].sum() * dmdt / neq

        # Sample neq magnitudes from CDF
        mout = numpy.interp(numpy.random.rand(int(neq)), cdf, m)

        return mout

    else:
        return numpy.empty(0)


@jitted
def rate2mag_vectorized(dt, r, m, b, seed):
    """Generate magnitude samples for every time steps."""
    if seed is not None:
        numpy.random.seed(seed)

    nm = len(m)
    nr = len(r)

    # Calculate Gutenberg-Richter rate scaling factor
    blg10 = b * numpy.log(10.0)
    bm = numpy.empty(nm, dtype=numpy.float64)
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
    times : array_like
        Dates for every seismicity rate samples.
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
    r = numpy.asarray(rates)
    if r.ndim != 1 or len(times) != r.size:
        raise ValueError()

    # Check magnitude bounds
    if len(m_bounds) != 2 or m_bounds[0] - m_bounds[1] >= 0.0:
        raise ValueError()
    m = numpy.linspace(m_bounds[0], m_bounds[1], n)

    # Convert datetimes to decimal years
    t = to_decimal_year(times)

    # Convert times to time increments for integration
    dt = numpy.diff(t)
    dt = numpy.append(dt, dt[-1])

    return rate2mag_vectorized(dt, r, m, b_value, seed)
