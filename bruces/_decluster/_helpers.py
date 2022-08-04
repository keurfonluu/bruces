__all__ = [
    "decluster",
]


_decluster_map = {}


def register(algorithm, decluster):
    """Register a new declustering algorithm."""
    _decluster_map[algorithm] = decluster


def decluster(catalog, algorithm="nearest-neighbor", **kwargs):
    """
    Decluster earthquake catalog.

    Parameters
    ----------
    catalog : :class:`bruces.Catalog`
        Earthquake catalog.
    algorithm : str, optional, default 'nearest-neighbor'
        Declustering algorithm:

         - 'gardner-knopoff': Gardner-Knopoff method
         - 'nearest-neighbor': nearest-neighbor algorithm (after Zaliapin and Ben-Zion, 2020)

    Other Parameters
    ----------------
    d : scalar, optional, default 1.6
        Only if ``algorithm = "nearest-neighbor"``. Fractal dimension of epicenter/hypocenter.
    w : scalar, optional, default 1.0
        Only if ``algorithm = "nearest-neighbor"``. Magnitude weighting factor (usually b-value).
    eta_0 : scalar, optional, default 0.1
        Only if ``algorithm = "nearest-neighbor"``. Initial cutoff threshold (as log10). If `None`, invoke :meth:`bruces.Catalog.fit_cutoff_threshold`.
    alpha_0 : scalar, optional, default 0.1
        Only if ``algorithm = "nearest-neighbor"``. Cluster threshold.
    M : int, optional, default 100
        Only if ``algorithm = "nearest-neighbor"``. Number of reshufflings.

    Returns
    -------
    :class:`bruces.Catalog`
        Declustered earthquake catalog.

    """
    return _decluster_map[algorithm](catalog, **kwargs)
