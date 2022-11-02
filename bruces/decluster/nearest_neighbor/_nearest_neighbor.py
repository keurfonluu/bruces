import logging

import numpy as np
from numba import prange

from ..._common import jitted, set_seed, time_space_distances
from .._helpers import register


def decluster(
    catalog,
    return_indices=False,
    d=1.6,
    w=1.0,
    eta_0=None,
    alpha_0=1.5,
    use_depth=False,
    M=100,
    seed=None,
):
    """
    Decluster earthquake catalog (after Zaliapin and Ben-Zion, 2020).

    Parameters
    ----------
    catalog : :class:`bruces.Catalog`
        Earthquake catalog.
    return_indices : bool, optional, default False
        If `True`, return indices of background events instead of declustered catalog.
    d : scalar, optional, default 1.6
        Fractal dimension of epicenter/hypocenter.
    w : scalar, optional, default 1.0
        Magnitude weighting factor (usually b-value).
    eta_0 : scalar or None, optional, default None
        Initial cutoff threshold. If `None`, invoke :meth:`bruces.Catalog.fit_cutoff_threshold`.
    alpha_0 : scalar, optional, default 1.5
        Cluster threshold.
    use_depth : bool, optional, default False
        If `True`, consider depth in interevent distance calculation.
    M : int, optional, default 100
        Number of reshufflings.
    seed : int or None, optional, default None
        Seed for random number generator.

    Returns
    -------
    :class:`bruces.Catalog` or array_like
        Declustered earthquake catalog or indices of background events.

    """
    if seed is not None:
        np.random.seed(seed)
        set_seed(seed)

    if eta_0 is None:
        eta_0 = catalog.fit_cutoff_threshold(d, w)

        if eta_0 is None:
            logging.warn("Skipping nearest-neighbor declustering.")

            return catalog

    t = catalog.years
    x = catalog.eastings
    y = catalog.northings
    z = catalog.depths
    m = catalog.magnitudes

    # Calculate nearest-neighbor proximities
    eta = _step1(t, x, y, z, m, d, w, use_depth)

    # Calculate proximity vectors
    kappa = _step2(t, x, y, z, m, eta, d, w, eta_0, M, use_depth)

    # Calculate normalized nearest-neighbor proximities
    alpha = _step3(eta, kappa)

    # Calculate retention probabilities and identify background events
    U = alpha + alpha_0 > np.log10(np.random.rand(len(catalog)))
    bg = np.nonzero(U)[0]

    return bg if return_indices else catalog[bg]


@jitted
def proximity(t, x, y, z, m, ti, xi, yi, zi, d, w, use_depth):
    """Calculate nearest-neighbor proximity."""
    T, R = time_space_distances(t, x, y, z, m, ti, xi, yi, zi, d, w, use_depth)

    return T + R if not np.isnan(T) else 20.0


@jitted(parallel=True)
def _step1(t, x, y, z, m, d, w, use_depth):
    """Calculate nearest-neighbor proximity for each event."""
    N = len(t)

    eta = np.empty(N, dtype=np.float64)
    for i in prange(N):
        eta[i] = proximity(t, x, y, z, m, t[i], x[i], y[i], z[i], d, w, use_depth)

    return eta


@jitted
def _step2(t, x, y, z, m, eta, d, w, eta_0, M, use_depth):
    """Calculate proximity vector for each event."""
    N = len(t)

    # Select N0 events that satisfy eta_i > eta_0
    ij = eta > eta_0
    N0 = ij.sum()

    # Initialize arrays
    xm = x[ij]
    ym = y[ij]
    zm = z[ij]
    mm = np.empty(N0, dtype=np.float64)

    # Loop over catalog
    kappa = np.empty((N, M), dtype=np.float64)

    tmin = t.min()
    tmax = t.max()
    for k in range(M):
        # Generate a randomized-reshuffled catalog
        tm = np.random.uniform(tmin, tmax, N0)
        mm[:] = m[np.random.permutation(N0)]

        # Calculate proximity vectors with respect to randomized catalog
        # Generating random numbers is not thread safe
        # See <https://stackoverflow.com/questions/71351836/random-seeds-and-multithreading-in-numba>
        _step2_kappa(tm, xm, ym, zm, mm, t, x, y, z, d, w, use_depth, kappa[:, k])

    return kappa


@jitted(parallel=True)
def _step2_kappa(tm, xm, ym, zm, mm, t, x, y, z, d, w, use_depth, kappa):
    """Calculate kappa in parallel."""
    for i in prange(len(kappa)):
        kappa[i] = proximity(
            tm, xm, ym, zm, mm, t[i], x[i], y[i], z[i], d, w, use_depth
        )


@jitted(parallel=True)
def _step3(eta, kappa):
    """Calculate normalized nearest-neighbor proximity for each event."""
    N = len(kappa)
    M = len(kappa[0, :])

    alpha = np.empty(N, dtype=np.float64)
    for i in prange(N):
        # Remove events without earlier events
        k_sum = 0.0
        count = 0
        for j in range(M):
            k = kappa[i, j]
            if k < 20.0:
                k_sum += k
                count += 1

        # First event has no earlier event
        if count == 0:
            alpha[i] = -20.0

        else:
            alpha[i] = eta[i] - k_sum / count

    return alpha


register("nearest-neighbor", decluster)
