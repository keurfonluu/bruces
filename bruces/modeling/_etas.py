import numpy as np
from numba.typed import List

from .._catalog import Catalog
from .._common import jitted, timedelta_to_day
from ..stats._sample_magnitude import grmag
from ..utils import to_decimal_year

__all__ = [
    "etas",
]


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
            mi = grmag(mmin, b=b)[0]
            pi = aftershock_rate(ti, mi, mmin, theta, alpha, c, K)

            ta.append(ti)
            ma.append(mi)
            pa.append(pi)
        
        Lc = Li

    return ta, ma, pa


@jitted
def aftershock_locations(ip, ma, mmin, alpha, d):
    """Generate aftershock locations."""
    n = len(ma)
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    for ia, i in enumerate(ip):
        r = np.sqrt(-2.0 * np.log(np.random.rand()) * d * np.exp(-alpha * (ma[i] - mmin)))
        phi = 2.0 * np.pi * np.random.rand()
        x[ia + 1] = x[i] + r * np.cos(phi)
        y[ia + 1] = y[i] + r * np.sin(phi)

    return x, y


def aftershocks(m, tmax, mmin, theta, alpha, c, K, b, d):
    """Generate a sequence of aftershocks."""
    # Generate origin times of aftershocks
    ta, ma, pa = aftershock_times(m, tmax, mmin, theta, alpha, c, K, b)
    ta = np.asarray(ta)
    ma = np.asarray(ma)
    pa = np.asarray(pa)

    # Randomly select parents of aftershocks
    # numba does not support yet option p of np.random.choice
    ip = [
        np.random.choice(i, p=pa[:i] / pa[:i].sum())
        for i in range(1, len(pa))
    ]
    ip = np.array(ip)

    # Generate locations of aftershocks
    if len(ip) > 0:
        x, y = aftershock_locations(ip, ma, mmin, alpha, d)

    else:
        x, y = np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)

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
    end_time = (
        to_decimal_year(end_time)
        if end_time is not None
        else catalog[-1].year
    )
    end_time -= catalog[0].year
    
    # Convert c to decimal years
    # c is usually defined in days in literature
    c = timedelta_to_day(c) if c is not None else 1.0e-3
    c /= 365.25

    t, x, y, m = [], [], [], []
    for eq in catalog:
        dt, dx, dy, ma = aftershocks(eq.magnitude, end_time, mc, theta, alpha, c, K, b, d)

        t.append(eq.year + dt)
        x.append(eq.easting + dx)
        y.append(eq.northing + dy)
        m.append(ma)

    t = np.concatenate(t)
    x = np.concatenate(x)
    y = np.concatenate(y)
    m = np.concatenate(m)

    return Catalog(
        origin_times=t,
        eastings=x,
        northings=y,
        magnitudes=m,
    )
