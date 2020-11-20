from collections import namedtuple
from datetime import datetime

import numpy

from ._common import rescaled_time_distance
from ._decluster import decluster
from ._helpers import to_decimal_year

__all__ = [
    "Catalog",
]


Earthquake = namedtuple("Earthquake", ["date", "easting", "northing", "depth", "magnitude"])


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

    def __len__(self):
        """Return number of earthquakes in catalog."""
        return len(self._dates)

    def __iter__(self):
        """Iterate over earthquake in catalog as namedtuples."""
        self._it = 0

        return self

    def __next__(self):
        """Return next earthquake in catalog."""
        if self._it < len(self):
            eq = Earthquake(
                self.dates[self._it],
                self.eastings[self._it],
                self.northings[self._it],
                self.depths[self._it],
                self.magnitudes[self._it],
            )
            self._it += 1

            return eq
        
        else:
            raise StopIteration

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
        :class:`pydecluster.Catalog`
            Declustered earthquake catalog.

        """
        return decluster(self, algorithm, **kwargs)

    def get_rescaled_time_distance(self, d=1.5, w=0.0):
        """
        Get rescaled time and distance for each earthquake in the catalog.

        Parameters
        ----------
        d : scalar, optional, default 1.5
            Fractal dimension of epicenter/hypocenter.
        w : scalar, optional, default 0.0
            Magnitude weighing factor (usually b-value).

        Returns
        -------
        array_like
            Rescaled times (column 0) and distances (column 1).
        
        """
        t = to_decimal_year(self.dates)  # Dates in years
        x = self.eastings
        y = self.northings
        m = self.magnitudes

        return numpy.array(
            [
                rescaled_time_distance(t, x, y, m, t[i], x[i], y[i], d, w)
                for i in range(len(self))
            ]
        )

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
