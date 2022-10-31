import numpy as np
from numba import prange

from ..._common import jitted, dist3d
from .._helpers import register


def decluster(catalog):
    """
    Decluster earthquake catalog using Gardner-Knopoff's method.

    Parameters
    ----------
    catalog : :class:`bruces.Catalog`
        Earthquake catalog.

    Returns
    -------
    :class:`bruces.Catalog`
        Declustered earthquake catalog.

    """
    t = catalog.years * 365.25  # Days
    x = catalog.eastings
    y = catalog.northings
    z = catalog.depths
    m = catalog.magnitudes

    bg = _decluster(t, x, y, z, m)
    return catalog[bg]


@jitted(parallel=True)
def _decluster(t, x, y, z, m):
    """Gardner-Knopoff's method."""
    N = len(t)

    bg = np.ones(N, dtype=np.bool_)
    for i in prange(N):
        mag = m[i]

        # Calculate distance window length
        dr = 0.1238 * mag + 0.983
        dr = 10.0**dr

        # Calculate time window length
        dt = 0.032 * mag + 2.7389 if mag >= 6.5 else 0.5409 * mag - 0.547
        dt = 10.0**dt

        # Loop over catalog
        for j in range(N):
            if bg[j] and m[j] < mag and 0 < t[j] - t[i] < dt:
                r_ij = dist3d(x[i], y[i], z[i], x[j], y[j], z[j])
                if r_ij < dr:
                    bg[j] = False

    return bg


register("gardner-knopoff", decluster)