from datetime import datetime

import numpy

from ._helpers import decluster

__all__ = [
    "Catalog",
]


def is_arraylike(arr, size):
    """Check input array."""
    return isinstance(arr, (list, tuple, numpy.ndarray)) and numpy.size(arr) == size


class Catalog:
    def __init__(self, dates, eastings, northings, depths, magnitudes):
        """
        Earthquake catalog.

        Parameters
        ----------
        dates : list of datetime.datetime
            Origin times.
        eastings : array_like
            Easting coordinates (in km).
        northings : array_like
            Northing coordinates (in km).
        depths : array_like
            Depths (in km).
        magnitudes : array_like
            Magnitudes.
        
        """
        if not isinstance(dates, (list, tuple)):
            raise TypeError()
        if any(not isinstance(time, datetime) for time in dates):
            raise TypeError()
        nev = len(dates)

        for arr in (eastings, northings, depths, magnitudes):
            if not is_arraylike(arr, nev):
                raise TypeError()
            if len(arr) != nev:
                raise ValueError()

        self._dates = dates
        self._eastings = eastings
        self._northings = northings
        self._depths = depths
        self._magnitudes = magnitudes

    def decluster(self, algorithm="nearest-neighbor", **kwargs):
        """
        Decluster earthquake catalog.

        Parameters
        ----------
        algorithm : str, optional, default 'nearest-neighbor'
            Declustering algorithm:
            - 'nearest-neighbor': nearest-neighbor algorithm (after Zaliapin and Ben-Zion, 2020).
        
        Other Parameters
        ----------------
        d : scalar, optional, default 1.5
            Only if ``algorithm = "nearest-neighbor"``. Fractal dimension of epicenter/hypocenter.
        w : scalar, optional, default 0.0
            Only if ``algorithm = "nearest-neighbor"``. Magnitude weighing factor (usually b-value).
        eta_0 : scalar, optional, default 0.1
            Only if ``algorithm = "nearest-neighbor"``. Initial cutoff threshold.
        alpha_0 : scalar, optional, default 0.1
            Only if ``algorithm = "nearest-neighbor"``. Cluster threshold.
        M : int, optional, default 100
            Only if ``algorithm = "nearest-neighbor"``. Number of reshufflings.

        Returns
        -------
        pydecluster.Catalog
            Declustered earthquake catalog.

        """
        return decluster(self, algorithm, **kwargs)

    @property
    def dates(self):
        """Return origin times."""
        return self._dates

    @property
    def eastings(self):
        """Return easting coordinates."""
        return self._eastings

    @property
    def northings(self):
        """Return northing coordinates."""
        return self._northings

    @property
    def depths(self):
        """Return depths."""
        return self._depths

    @property
    def magnitudes(self):
        """Return magnitudes."""
        return self._magnitudes
