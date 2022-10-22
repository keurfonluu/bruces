import numpy as np
from numba import prange

from ..._common import jitted
from ..._helpers import to_decimal_year
from .._helpers import register


def decluster(catalog):
    """
    Decluster earthquake catalog using Gardner-Knopoff method.

    Parameters
    ----------
    catalog : bruces.Catalog
        Earthquake catalog.

    Returns
    -------
    :class:`bruces.Catalog`
        Declustered earthquake catalog.

    """
    t = to_decimal_year(catalog.dates) * 365.25  # Days
    x = catalog.eastings
    y = catalog.northings
    z = catalog.depths
    m = catalog.magnitudes

    bg = _decluster(t, x, y, z, m)
    return catalog[bg]


@jitted(parallel=True)
def _decluster(t, x, y, z, m):
    """Gardner-Knopoff method."""
    N = len(t)

    bg = np.ones(N, dtype=np.bool_)
    for i in range(N):
        mag = m[i]

        # Calculate distance window length
        dr = 0.1238 * mag + 0.983
        dr = 10.0**dr

        # Calculate time window length
        dt = 0.032 * mag + 2.7389 if mag >= 6.5 else 0.5409 * mag - 0.547
        dt = 10.0**dt

        # Loop over catalog
        for j in prange(N):
            if bg[j] and m[j] < mag and 0 < t[j] - t[i] < dt:
                r_ij = ((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2) ** 0.5
                if r_ij < dr:
                    bg[j] = False

    return bg


register("gardner-knopoff", decluster)
