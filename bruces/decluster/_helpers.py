__all__ = [
    "decluster",
]


_decluster_map = {}


def register(algorithm, decluster):
    """Register a new declustering algorithm."""
    _decluster_map[algorithm] = decluster


def decluster(catalog, algorithm="nearest-neighbor", return_indices=False, **kwargs):
    """
    Decluster earthquake catalog.

    Parameters
    ----------
    catalog : :class:`bruces.Catalog`
        Earthquake catalog.
    algorithm : str, optional, default 'nearest-neighbor'
        Declustering algorithm:

         - 'gardner-knopoff': Gardner-Knopoff's method (after Gardner and Knopoff, 1974)
         - 'nearest-neighbor': nearest-neighbor algorithm (after Zaliapin and Ben-Zion, 2020)
         - 'reasenberg': Reasenberg's method (after Reasenberg, 1985)

    return_indices : bool, optional, default False
        If `True`, return indices of background events instead of declustered catalog.

    Other Parameters
    ----------------
    window : str {'default', 'gruenthal', 'uhrhammer'}, optional, default 'default'
        Only if ``algorithm = "gardner-knopoff"``. Distance and time windows:

         - 'default': Gardner and Knopoff (1974)
         - 'gruenthal': personnal communication (see van Stiphout et al., 2012)
         - 'uhrhammer': Uhrhammer (1986)

    d : scalar, optional, default 1.6
        Only if ``algorithm = "nearest-neighbor"``. Fractal dimension of epicenter/hypocenter.
    w : scalar, optional, default 1.0
        Only if ``algorithm = "nearest-neighbor"``. Magnitude weighting factor (usually b-value).
    eta_0 : scalar or None, optional, default None
        Only if ``algorithm = "nearest-neighbor"``. Initial cutoff threshold. If `None`, invoke :meth:`bruces.Catalog.fit_cutoff_threshold`.
    alpha_0 : scalar, optional, default 1.5
        Only if ``algorithm = "nearest-neighbor"``. Cluster threshold.
    use_depth : bool, optional, default False
        Only if ``algorithm = "nearest-neighbor"``. If `True`, consider depth in interevent distance calculation.
    M : int, optional, default 100
        Only if ``algorithm = "nearest-neighbor"``. Number of reshufflings.
    seed : int or None, optional, default None
        Only if ``algorithm = "nearest-neighbor"``. Seed for random number generator.
    rfact : int, optional, default 10
        Only if ``algorithm = "reasenberg"``. Number of crack radii surrounding each earthquake within which to consider linking a new event into a cluster.
    xmeff : scalar or None, optional, default None
        Only if ``algorithm = "reasenberg"``. "Effective" lower magnitude cutoff for catalog. If `None`, use minimum magnitude in catalog.
    xk : scalar, optional, default 0.5
        Only if ``algorithm = "reasenberg"``. Factor by which ``xmeff`` is raised during clusters.
    tau_min : scalar, timedelta_like or None, optional, default None
        Only if ``algorithm = "reasenberg"``. Look ahead time for non-clustered events (in days if scalar). Default is 1 day.
    tau_max : tau_min : scalar, timedelta_like or None, optional, default None
        Only if ``algorithm = "reasenberg"``. Maximum look ahead time for clustered events (in days if scalar). Default is 10 days.
    p : scalar, optional, default 0.95
        Only if ``algorithm = "reasenberg"``. Confidence of observing the next event in the sequence.

    Returns
    -------
    :class:`bruces.Catalog` or array_like
        Declustered earthquake catalog or indices of background events.

    """
    return _decluster_map[algorithm](catalog, return_indices, **kwargs)
