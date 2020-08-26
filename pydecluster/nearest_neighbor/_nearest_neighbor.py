import numpy

from .._common import jitted, proximity
from .._catalog import Catalog
from .._helpers import register


def declusterize(catalog, d, w, eta_0, alpha_0, M):
    t = catalog.times
    x = catalog.eastings
    y = catalog.northings
    z = catalog.depths
    m = catalog.magnitudes
    N = len(t)

    # Calculate nearest-neighbor proximities
    eta = _step1(t, x, y, m)

    # Calculate proximity vectors
    kappa = _step2(t, x, y, m, eta, eta_0, M)

    # Calculate normalized nearest-neighbor proximities
    alpha = _step3(eta, kappa)

    # Calculate retention probabilities and identify background events
    P = alpha * 10.0 ** alpha_0
    U = P > numpy.random.rand(N)
    bg = numpy.nonzero(U)[0]

    return Catalog(
        times=t[bg],
        eastings=x[bg],
        northings=y[bg],
        depths=z[bg],
        magnitudes=m[bg],
    )


@jitted
def _step1(t, x, y, m):
    """Calculate nearest-neighbor proximity for each event."""
    N = len(t)

    eta = numpy.empty(N, dtype=numpy.float64)
    for i in range(N):
        eta[i] = proximity(t, x, y, m, t[i], x[i], y[i])

    return eta


@jitted
def _step2(t, x, y, m, eta, eta_0, M):
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
    kappa = numpy.empty((M, N), dtype=numpy.float64)

    tmin = t.min()
    tmax = t.max()
    for k in range(M):
        # Generate a randomized-reshuffled catalog
        tm = numpy.random.uniform(tmin, tmax, N0)
        mm[:] = m[numpy.random.permutation(N0)]

        # Calculate proximity vectors with respect to randomized catalog
        for i in range(N):
            kappa[k, i] = proximity(tm, xm, ym, mm, t[i], x[i], y[i])

    return kappa


@jitted
def _step3(eta, kappa):
    """Calculate normalized nearest-neighbor proximity for each event."""
    N = len(eta)
    
    alpha = numpy.empty(N, dtype=numpy.float64)
    for i in range(N):
        # Remove events without earlier events
        k = kappa[[kappa[:, i] < 1.0e20], i]

        # First event has no earlier event
        if k.size == 0:
            alpha[i] = 0.0
        else:
            alpha[i] = eta[i] * 10.0 ** (-numpy.log10(k).mean())

    return alpha


register("nearest-neighbor", declusterize)
