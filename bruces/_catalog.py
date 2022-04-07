from collections import namedtuple
from datetime import datetime, timedelta

import numpy as np

from ._common import time_space_distances
from ._decluster import decluster
from ._helpers import to_decimal_year

__all__ = [
    "Catalog",
]


Earthquake = namedtuple(
    "Earthquake", ["date", "easting", "northing", "depth", "magnitude"]
)


def is_arraylike(arr, size):
    """Check input array."""
    return isinstance(arr, (list, tuple, np.ndarray)) and np.size(arr) == size


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
        if not isinstance(dates, (list, tuple, np.ndarray)):
            raise TypeError()
        if any(not isinstance(time, (datetime, np.datetime64)) for time in dates):
            raise TypeError()
        nev = len(dates)

        for arr in (eastings, northings, depths, magnitudes):
            if not is_arraylike(arr, nev):
                raise TypeError()
            if len(arr) != nev:
                raise ValueError()

        self._dates = np.asarray(dates)
        self._eastings = np.asarray(eastings)
        self._northings = np.asarray(northings)
        self._depths = np.asarray(depths)
        self._magnitudes = np.asarray(magnitudes)

    def __len__(self):
        """Return number of earthquakes in catalog."""
        return len(self._dates)

    def __getitem__(self, islice):
        """Slice catalog."""
        t = self.dates[islice]
        x = self.eastings[islice]
        y = self.northings[islice]
        z = self.depths[islice]
        m = self.magnitudes[islice]

        return Catalog(t, x, y, z, m) if np.ndim(t) > 0 else Earthquake(t, x, y, z, m)

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
        seed : int or None, optional, default None
            Only if ``algorithm = "nearest-neighbor"``. Seed for random number generator.

        Returns
        -------
        :class:`bruces.Catalog`
            Declustered earthquake catalog.

        """
        return decluster(self, algorithm, **kwargs)

    def time_space_distances(self, d=1.5, w=0.0):
        """
        Get rescaled time and space distances for each earthquake in the catalog.

        Parameters
        ----------
        d : scalar, optional, default 1.5
            Fractal dimension of epicenter/hypocenter.
        w : scalar, optional, default 0.0
            Magnitude weighing factor (usually b-value).

        Returns
        -------
        array_like
            Rescaled time distances.
        array_like
            Rescaled space distances.

        """
        t = to_decimal_year(self.dates)  # Dates in years
        x = self.eastings
        y = self.northings
        m = self.magnitudes

        return np.transpose(
            [
                time_space_distances(t, x, y, m, t[i], x[i], y[i], d, w)
                for i in range(len(self))
            ]
        )

    def seismicity_rate(self, tbins):
        """
        Get seismicity rate.

        Parameters
        ----------
        tbins : datetime.timedelta, np.timedelta64 or sequence of datetime_like
            If `tbins` is a :class:`datetime.timedelta` or a :class:`np.timedelta64`, it defines the width of each bin.
            If `tbins` is a sequence of datetime_like, it defines a monotonically increasing list of bin edges.

        Returns
        -------
        array_like
            Seismicity rate (in events/year).
        array_like
            Bin edges.

        """
        if isinstance(tbins, (timedelta, np.timedelta64)):
            tmin = min(self.dates)
            tmax = max(self.dates)
            tbins = np.arange(tmin, tmax, tbins, dtype="M8[ms]").tolist()

        elif isinstance(tbins, (list, tuple, np.ndarray)):
            if any(not isinstance(t, (datetime, np.datetime64)) for t in tbins):
                raise TypeError()
            tbins = tbins

        else:
            raise TypeError()

        ty = to_decimal_year(self.dates)
        tybins = to_decimal_year(tbins)
        hist, _ = np.histogram(ty, bins=tybins)

        return hist / np.diff(tybins), tbins

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
