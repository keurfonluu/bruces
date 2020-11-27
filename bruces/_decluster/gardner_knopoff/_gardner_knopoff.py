import numpy

from numba import prange

from .._helpers import register
from ..._common import jitted
from ..._helpers import to_decimal_year


def decluster(catalog):
    """
    Decluster earthquake catalog using Gardner-Knopoff method.

    Parameters
    ----------
    catalog : pydecluster.Catalog
        Earthquake catalog.

    Returns
    -------
    :class:`pydecluster.Catalog`
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

    bg = numpy.ones(N, dtype=numpy.bool_)
    for i in prange(N):
        mag = m[i]

        # Calculate distance window length
        dr = 0.1238 * mag + 0.983
        dr = 10.0 ** dr

        # Calculate time window length
        dt = (
            0.032 * mag + 2.7389
            if mag >= 6.5
            else 0.5409 * mag - 0.547
        )
        dt = 10.0 ** dt

        # Loop over catalog
        for j in range(N):
            if m[j] < mag:
                t_ij = numpy.abs(t[j] - t[i])
                if t_ij < dt:
                    r_ij = ((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2) ** 0.5
                    if r_ij < dr:
                        bg[i] = False

    return bg


register("gardner-knopoff", decluster)
