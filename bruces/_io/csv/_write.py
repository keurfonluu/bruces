import numpy as np

from ..._common import open_file
from ...utils import to_datetime


def write(filename, catalog):
    """
    Write catalog to CSV file.

    Parameters
    ----------
    filename : str, pathlike or buffer
        Output file name or buffer.
    catalog : :class:`bruces.Catalog`
        Earthquake catalog to export.

    """
    buffer = write_buffer(catalog)

    with open_file(filename, "w") as f:
        for record in buffer:
            f.write(f"{record}\n")


def write_buffer(catalog):
    """Write CSV catalog."""
    out = [
        "year,month,day,hour,minute,second,latitude,longitude,easting,northing,depth,magnitude"
    ]

    for eq in catalog:
        date = to_datetime(eq.origin_time)

        Y = date.year
        M = date.month
        D = date.day
        h = date.hour
        m = date.minute
        s = np.round(date.second + date.microsecond * 1.0e-6, 6)
        lat = eq.latitude if not np.isnan(eq.latitude) else ""
        lon = eq.longitude if not np.isnan(eq.longitude) else ""
        x = eq.easting
        y = eq.northing
        z = eq.depth if not np.isnan(eq.depth) else ""
        mag = eq.magnitude if not np.isnan(eq.magnitude) else ""

        out += [f"{Y},{M},{D},{h},{m},{s},{lat},{lon},{x},{y},{z},{mag}"]

    return out
