import numpy

from ._helpers import declusterize

__all__ = [
    "Catalog",
]


class Catalog:
    def __init__(self, times, eastings, northings, depths, magnitudes):
        self._times = times
        self._eastings = eastings
        self._northings = northings
        self._depths = depths
        self._magnitudes = magnitudes

    def declusterize(self, algorithm="nearest-neighbor", **kwargs):
        return declusterize(self, algorithm, **kwargs)

    @property
    def times(self):
        return self._times

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
