import numpy as np

from ..._common import dist3d, jitted
from .._helpers import register


def decluster(catalog, return_indices=False, window="default"):
    """
    Decluster earthquake catalog using Gardner-Knopoff's method.

    Parameters
    ----------
    catalog : :class:`bruces.Catalog`
        Earthquake catalog.
    return_indices : bool, optional, default False
        If `True`, return indices of background events instead of declustered catalog.
    window : str {'default', 'gruenthal', 'uhrhammer'}, optional, default 'default'
        Distance and time windows:

         - 'default': Gardner and Knopoff (1974)
         - 'gruenthal': personnal communication (see van Stiphout et al., 2012)
         - 'uhrhammer': Uhrhammer (1986)

    Returns
    -------
    :class:`bruces.Catalog` or array_like
        Declustered earthquake catalog or indices of background events.

    """
    if window not in {"default", "gruenthal", "uhrhammer"}:
        raise ValueError()

    t = catalog.years * 365.25  # Days
    x = catalog.eastings
    y = catalog.northings
    z = catalog.depths
    m = catalog.magnitudes

    bg = _decluster(t, x, y, z, m, window)

    return np.arange(len(catalog))[bg] if return_indices else catalog[bg]


@jitted
def _decluster(t, x, y, z, m, window):
    """Window method."""
    N = len(t)

    bg = np.ones(N, dtype=np.bool_)
    for i in range(N):
        mag = m[i]

        if window == "default":
            dr = np.power(10.0, 0.1238 * mag + 0.983)
            dt = (
                np.power(10.0, 0.032 * mag + 2.7389)
                if mag >= 6.5
                else np.power(10.0, 0.5409 * mag - 0.547)
            )

        elif window == "gruenthal":
            dr = np.exp(1.77 + (0.037 + 1.02 * mag) ** 0.5)
            dt = (
                np.exp(-3.95 + (0.62 + 17.32 * mag) ** 0.5)
                if mag < 6.5
                else np.power(10.0, 2.8 + 0.024 * mag)
            )

        elif window == "uhrhammer":
            dr = np.exp(-1.024 + 0.804 * mag)
            dt = np.exp(-2.87 + 1.235 * mag)

        # Loop over catalog
        for j in range(N):
            if bg[j] and m[j] < mag and 0 < t[j] - t[i] < dt:
                r_ij = dist3d(x[i], y[i], z[i], x[j], y[j], z[j])
                if r_ij < dr:
                    bg[j] = False

    return bg


register("gardner-knopoff", decluster)
