import numpy as np
from numba.typed import List

from .._catalog import Catalog
from .._common import jitted, timedelta_to_day
from .._helpers import set_seed
from ..utils import to_decimal_year

__all__ = [
    "etas",
]


@jitted
def random_choice(p):
    """
    Draw a random sample from an array with probability.

    Note
    ----
    This is a custom jitted version of :func:`numpy.random.choice`.
    :mod:`numba` currently does not support its argument `p`.

    """
    probs = np.cumsum(p)

    return np.argmax(np.random.rand() * probs[-1] < probs)


@jitted
def aftershock_rate(t, m, mmin, theta, alpha, c, K):
    """Rate of aftershocks."""
    rho = K * np.power(10.0, alpha * (m - mmin))
    psi = theta * np.power(c, theta) / np.power(t + c, 1.0 + theta)

    return rho * psi


@jitted
def intensity(t, ta, ma, mmin, theta, alpha, c, K):
    """Conditional intensity."""
    l = 0.0
    for ti, mi in zip(ta, ma):
        l += aftershock_rate(t - ti, mi, mmin, theta, alpha, c, K)

    return l


@jitted
def aftershock_times(m, tmax, mmin, theta, alpha, c, K, b):
    """Generate aftershock times."""
    beta = 2.30258509 * b
    ti = 0.0
    ta = List([ti])
    ma = List([m])
    pa = List([aftershock_rate(ti, m, mmin, theta, alpha, c, K)])
    Lc = intensity(0.0, ta, ma, mmin, theta, alpha, c, K)

    while True:
        ti -= np.log(np.random.rand()) / Lc

        if ti > tmax:
            break

        Li = intensity(ti, ta, ma, mmin, theta, alpha, c, K)
        if np.random.rand() * Lc < Li:
            u = np.random.rand()
            u *= 1.0 - np.exp(-beta * (m - mmin))
            mi = mmin - np.log(1.0 - u) / beta
            pi = aftershock_rate(ti, mi, mmin, theta, alpha, c, K)

            ta.append(ti)
            ma.append(mi)
            pa.append(pi)

        Lc = Li

    return ta, ma, pa


@jitted
def aftershock_locations(pa, ma, mmin, alpha, d):
    """Generate aftershock locations."""
    n = len(pa)
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    for ia in range(1, n):
        # Randomly select parent of aftershock
        i = random_choice(pa[:ia])

        r = np.sqrt(
            -2.0 * np.log(np.random.rand()) * d * np.exp(-alpha * (ma[i] - mmin))
        )
        phi = 2.0 * np.pi * np.random.rand()
        x[ia] = x[i] + r * np.cos(phi)
        y[ia] = y[i] + r * np.sin(phi)

    return x, y


def aftershocks(m, tmax, mmin, theta, alpha, c, K, b, d):
    """Generate a sequence of aftershocks."""
    # Generate origin times of aftershocks
    ta, ma, pa = aftershock_times(m, tmax, mmin, theta, alpha, c, K, b)
    ta = np.array(ta, dtype=np.float64)
    ma = np.array(ma, dtype=np.float64)
    pa = np.array(pa, dtype=np.float64)

    # Generate locations of aftershocks
    x, y = aftershock_locations(pa, ma, mmin, alpha, d)

    return ta, x, y, ma


def etas(
    catalog,
    end_time=None,
    mc=0.0,
    theta=0.2,
    alpha=0.5,
    c=0.001,
    K=0.1,
    b=1.0,
    d=1.0,
    seed=None,
):
    """
    Epidemic-Type Aftershock Sequence.

    Parameters
    ----------
    catalog : :class:`bruces.Catalog`
        Earthquake catalog of background events.
    end_time : scalar, datetime_like or None, optional, default None
        Maximum simulation time. Default is origin time of last earthquake in input catalog.
    mc : scalar, optional, default 0.0
        Magnitude of completeness.
    theta : scalar, optional, default 0.2
        Omori's exponent (> 0.0).
    alpha : scalar, optional, default 0.5
        Aftershock productivity constant.
    c : scalar, timedelta_like or None, optional, default None
        Aftershock productivity constant (in days if scalar). Default is 0.001 days.
    K : scalar, optional, default 0.1
        Aftershock productivity constant.
    b : scalar, optional, default 1.0
        b-value.
    d : scalar, optional, default 1.0
        Characteristic length (in km).
    seed : int or None, optional, default None
        Seed for random number generator.

    Returns
    -------
    :class:`bruces.Catalog`
        Simulated catalog (background and aftershock events).

    Note
    ----
     - Origin times are generated following Helmstetter and Sornette (2002)
     - Locations are implemented following Ogata (1998)
     - Magnitudes are distributed following Gutenberg-Richter law

    """
    end_time = to_decimal_year(end_time) if end_time is not None else catalog[-1].year
    end_time -= catalog[0].year

    # Convert c to decimal years
    # c is usually defined in days in literature
    c = timedelta_to_day(c) if c is not None else 1.0e-3
    c /= 365.25

    # Set random seed
    if seed is not None:
        set_seed(seed)

    t, x, y, m = [], [], [], []
    for eq in catalog:
        dt, dx, dy, ma = aftershocks(
            eq.magnitude, end_time, mc, theta, alpha, c, K, b, d
        )

        t.append(eq.year + dt)
        x.append(eq.easting + dx)
        y.append(eq.northing + dy)
        m.append(ma)

    t = np.concatenate(t, dtype=np.float64)
    x = np.concatenate(x, dtype=np.float64)
    y = np.concatenate(y, dtype=np.float64)
    m = np.concatenate(m, dtype=np.float64)

    return Catalog(
        origin_times=t,
        eastings=x,
        northings=y,
        magnitudes=m,
    )
