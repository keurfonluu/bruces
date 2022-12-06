from .utils import to_datetime, to_decimal_year


class Earthquake:
    def __init__(
        self,
        origin_time,
        latitude=None,
        longitude=None,
        easting=None,
        northing=None,
        depth=None,
        magnitude=None,
    ):
        """
        Earthquake object.

        """
        self._origin_time = to_datetime(origin_time)
        self._latitude = latitude
        self._longitude = longitude
        self._easting = easting
        self._northing = northing
        self._depth = depth
        self._magnitude = magnitude

    @property
    def origin_time(self):
        """Return origin time."""
        return self._origin_time

    @property
    def latitude(self):
        """Return latitude."""
        return self._latitude

    @property
    def longitude(self):
        """Return longitude."""
        return self._longitude

    @property
    def easting(self):
        """Return easting coordinate."""
        return self._easting

    @property
    def northing(self):
        """Return northing coordinate."""
        return self._northing

    @property
    def depth(self):
        """Return depth."""
        return self._depth

    @property
    def magnitude(self):
        """Return magnitude."""
        return self._magnitude

    @property
    def year(self):
        """Return origin time in decimal year."""
        return to_decimal_year(self._origin_time)
