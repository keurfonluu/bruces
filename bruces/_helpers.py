import numpy as np

from ._common import set_seed as set_seed_numba

__all__ = [
    "from_csep",
    "set_seed",
]


def from_csep(catalog):
    """
    Convert a :class:`csep.Catalog` to :class:`bruces.Catalog`.

    Parameters
    ----------
    catalog : :class:`csep.Catalog`
        Input catalog.

    Returns
    -------
    :class:`bruces.Catalog`
        Output catalog.

    """
    from ._catalog import Catalog

    return Catalog(
        origin_times=catalog.get_datetimes(),
        latitudes=catalog.get_latitudes(),
        longitudes=catalog.get_longitudes(),
        depths=catalog.get_depths(),
        magnitudes=catalog.get_magnitudes(),
    )


def set_seed(seed):
    """
    Set random number generator seed.

    Parameters
    ----------
    seed : int
        Random number generator seed.

    """
    np.random.seed(seed)
    set_seed_numba(seed)
