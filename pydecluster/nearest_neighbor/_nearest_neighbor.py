import numpy

from numba import prange

from .._common import jitted, proximity
from .._catalog import Catalog
from .._helpers import register, to_decimal_year


def decluster(catalog, d=1.5, w=0.0, eta_0=0.1, alpha_0=0.1, M=100):
    """
    Decluster earthquake catalog (after Zaliapin and Ben-Zion, 2020).

    Parameters
    ----------
    catalog : pydecluster.Catalog
        Earthquake catalog.
    d : scalar, optional, default 1.5
        Fractal dimension of epicenter/hypocenter.
    w : scalar, optional, default 0.0
        Magnitude weighing factor (usually b-value).
    eta_0 : scalar, optional, default 0.1
        Initial cutoff threshold.
    alpha_0 : scalar, optional, default 0.1
        Cluster threshold.
    M : int, optional, default 100
        Number of reshufflings.

    Returns
    -------
    pydecluster.Catalog
        Declustered earthquake catalog.

    """
    t = to_decimal_year(catalog.dates)  # Dates in years
    x = catalog.eastings
    y = catalog.northings
    z = catalog.depths
    m = catalog.magnitudes

    # Calculate nearest-neighbor proximities
    eta = _step1(t, x, y, m, d, w)

    # Calculate proximity vectors
    kappa = _step2(t, x, y, m, eta, d, w, eta_0, M)

    # Calculate normalized nearest-neighbor proximities
    alpha = _step3(eta, kappa)

    # Calculate retention probabilities and identify background events
    N = len(t)
    P = alpha * 10.0 ** alpha_0
    U = P > numpy.random.rand(N)
    bg = numpy.nonzero(U)[0]

    return Catalog(
        dates=[catalog.dates[i] for i in bg],
        eastings=x[bg],
        northings=y[bg],
        depths=z[bg],
        magnitudes=m[bg],
    )


@jitted(parallel=True)
def _step1(t, x, y, m, d, w):
    """Calculate nearest-neighbor proximity for each event."""
    N = len(t)

    eta = numpy.empty(N, dtype=numpy.float64)
    for i in prange(N):
        eta[i] = proximity(t, x, y, m, t[i], x[i], y[i], d, w)

    return eta


@jitted(parallel=True)
def _step2(t, x, y, m, eta, d, w, eta_0, M):
    """Calculate proximity vector for each event."""
    N = len(t)

    # Select N0 events that satisfy eta_i > eta_0
    ij = numpy.empty(N, dtype=numpy.int32)
    for i in range(N):
        ij[i] = 1 if eta[i] > eta_0 else 0
    N0 = ij.sum()

    # Initialize arrays
    xm = numpy.empty(N0)
    ym = numpy.empty(N0)
    mm = numpy.empty(N0)

    j = 0
    for i in range(N):
        if ij[i] == 1:
            xm[j] = x[i]
            ym[j] = y[i]
            mm[j] = m[i]
            j += 1

    # Loop over catalog
    kappa = numpy.empty((N, M), dtype=numpy.float64)

    tmin = t.min()
    tmax = t.max()
    for k in range(M):
        # Generate a randomized-reshuffled catalog
        tm = numpy.random.uniform(tmin, tmax, N0)
        mm[:] = m[numpy.random.permutation(N0)]

        # Calculate proximity vectors with respect to randomized catalog
        for i in prange(N):
            kappa[i, k] = proximity(tm, xm, ym, mm, t[i], x[i], y[i], d, w)

    return kappa


@jitted(parallel=True)
def _step3(eta, kappa):
    """Calculate normalized nearest-neighbor proximity for each event."""
    N = len(kappa)
    M = len(kappa[0, :])
    
    alpha = numpy.empty(N, dtype=numpy.float64)
    for i in prange(N):
        # Remove events without earlier events
        logk_sum = 0.0
        count = 0
        for j in range(M):
            k = kappa[i, j]
            if k < 1.0e20:
                logk_sum += numpy.log10(k)
                count += 1

        # First event has no earlier event
        if count == 0:
            alpha[i] = 0.0
        else:
            alpha[i] = eta[i] * 10.0 ** (-logk_sum / count)

    return alpha


register("nearest-neighbor", decluster)
