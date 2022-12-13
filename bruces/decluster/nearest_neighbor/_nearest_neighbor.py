import logging

import numpy as np
from numba import prange

from ..._common import jitted, time_space_distances
from ..._helpers import set_seed
from .._helpers import register


def decluster(
    catalog,
    return_indices=False,
    method="gaussian-mixture",
    d=1.6,
    w=1.0,
    eta_0=None,
    alpha_0=1.5,
    use_depth=False,
    M=16,
    seed=None,
):
    """
    Decluster earthquake catalog.

    Parameters
    ----------
    catalog : :class:`bruces.Catalog`
        Earthquake catalog.
    return_indices : bool, optional, default False
        If `True`, return indices of background events instead of declustered catalog.
    method : str, optional, default 'gaussian-mixture'
        Declustering method:
        
         - 'gaussian-mixture': use a 2D Gaussian Mixture classifier
         - 'thinning': random thinning (after Zaliapin and Ben-Zion, 2020)

    d : scalar, optional, default 1.6
        Fractal dimension of epicenter/hypocenter.
    w : scalar, optional, default 1.0
        Magnitude weighting factor (usually b-value).
    use_depth : bool, optional, default False
        If `True`, consider depth in interevent distance calculation.
    eta_0 : scalar or None, optional, default None
        Only if ``method = "thinning"`. Initial cutoff threshold. If `None`, invoke :meth:`bruces.Catalog.fit_cutoff_threshold`.
    alpha_0 : scalar, optional, default 1.5
        Only if ``method = "thinning"`. Cluster threshold.
    M : int, optional, default 16
        Only if ``method = "thinning"`. Number of reshufflings.
    seed : int or None, optional, default None
        Seed for random number generator.

    Returns
    -------
    :class:`bruces.Catalog` or array_like
        Declustered earthquake catalog or indices of background events.

    """
    if seed is not None:
        set_seed(seed)

    if method == "gaussian-mixture":
        try:
            from sklearn.mixture import GaussianMixture

        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Method '{method}' requires scikit-learn to be installed.")

        # Calculate rescaled time and space distances (as log10)
        T, R = catalog.time_space_distances(w, d, use_depth=use_depth, returns_log=True, prune_nans=False)

        # Set nans to max values (force events as background)
        idx = np.isnan(T)
        T[idx] = T[~idx].max()
        R[idx] = R[~idx].max()

        # Fit a mixture of two 2D Gaussian distributions
        gm = GaussianMixture(n_components=2)
        y_pred = gm.fit_predict(np.column_stack((T, R)))

        # Identify background events as those classified in class with largest mean
        sig0, sig1 = gm.means_.sum(axis=-1)
        bg = np.flatnonzero(y_pred == int(sig0 < sig1))

    elif method == "thinning":
        if eta_0 is None:
            eta_0 = catalog.fit_cutoff_threshold(d, w)

            if eta_0 is None:
                logging.warn("Skipping nearest-neighbor declustering.")

                return np.arange(len(catalog)) if return_indices else catalog

        t = catalog.years
        x = catalog.eastings
        y = catalog.northings
        z = catalog.depths
        m = catalog.magnitudes

        # Calculate nearest-neighbor proximities
        eta = np.empty(len(t), dtype=np.float64)
        proximity_catalog(t, x, y, z, m, t, x, y, z, d, w, use_depth, eta)

        # Calculate proximity vectors
        kappa = proximity_vector(t, x, y, z, m, eta, d, w, eta_0, M, use_depth)

        # Calculate normalized nearest-neighbor proximities
        alpha = normalize_proximity(eta, kappa)

        # Calculate retention probabilities and identify background events
        U = alpha + alpha_0 > np.log10(np.random.rand(len(catalog)))
        bg = np.flatnonzero(U)

    else:
        raise ValueError(f"Unknown method '{method}'.")

    return bg if return_indices else catalog[bg]


@jitted
def proximity(t, x, y, z, m, ti, xi, yi, zi, d, w, use_depth):
    """Calculate nearest-neighbor proximity."""
    T, R = time_space_distances(t, x, y, z, m, ti, xi, yi, zi, d, w, use_depth)

    return T + R if not np.isnan(T) else 20.0


@jitted(parallel=True)
def proximity_catalog(tc, xc, yc, zc, mc, t, x, y, z, d, w, use_depth, eta):
    """Calculate nearest-neighbor proximity for each eventin catalog."""
    for i in prange(len(eta)):
        eta[i] = proximity(tc, xc, yc, zc, mc, t[i], x[i], y[i], z[i], d, w, use_depth)


@jitted
def proximity_vector(t, x, y, z, m, eta, d, w, eta_0, M, use_depth):
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
        proximity_catalog(tm, xm, ym, zm, mm, t, x, y, z, d, w, use_depth, kappa[:, k])

    return kappa


@jitted(parallel=True)
def normalize_proximity(eta, kappa):
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
