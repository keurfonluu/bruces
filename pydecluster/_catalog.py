from datetime import datetime

import numpy

from ._helpers import declusterize

__all__ = [
    "Catalog",
]


def is_arraylike(arr, size):
    """Check input array."""
    return isinstance(arr, (list, tuple, numpy.ndarray)) and numpy.size(arr) == size


class Catalog:
    def __init__(self, dates, eastings, northings, depths, magnitudes):
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

    def declusterize(self, algorithm="nearest-neighbor", **kwargs):
        return declusterize(self, algorithm, **kwargs)

    @property
    def dates(self):
        return self._dates

    @property
    def eastings(self):
        return self._eastings

    @property
    def northings(self):
        return self._northings

    @property
    def depths(self):
        return self._depths

    @property
    def magnitudes(self):
        return self._magnitudes
