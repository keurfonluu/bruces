import pathlib
import tempfile

import pandas as pd

import bruces


def comcat(use_utm=False):
    """
    Catalog downloaded using :mod:`pycsep`.

    Note
    ----
    The following code has been used to download the catalog:

    .. code-block::

        from datetime import datetime
        import csep

        catalog = csep.query_comcat(
            start_time=datetime(2016, 1, 1),
            end_time=datetime(2017, 1, 1),
            min_magnitude=3.0,
            min_latitude=35.0,
            max_latitude=37.0,
            min_longitude=-99.5,
            max_longitude=-96.0,
        )

    """
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "support_files" / "comcat.csv"

    df = pd.read_csv(filename)
    kwargs = {
        "origin_times": pd.to_datetime(df["Origin time"], utc=False).to_numpy(),
        "depths": df["Depth"].to_numpy(),
        "magnitudes": df["Magnitude"].to_numpy(),
    }

    if use_utm:
        kwargs["eastings"] = df["Easting"].to_numpy()
        kwargs["northings"] = df["Northing"].to_numpy()

    else:
        kwargs["latitudes"] = df["Latitude"].to_numpy()
        kwargs["longitudes"] = df["Longitude"].to_numpy()

    return bruces.Catalog(**kwargs)


def tempdir(filename=""):
    temp_dir = pathlib.Path(tempfile.mkdtemp())

    return temp_dir / filename


def write_read(filename, obj, writer, reader, writer_kws=None, reader_kws=None):
    writer_kws = writer_kws if writer_kws else {}
    reader_kws = reader_kws if reader_kws else {}

    filepath = tempdir(filename)
    if obj is not None:
        writer(filepath, obj, **writer_kws)

    else:
        writer(filepath, **writer_kws)

    return reader(filepath, **reader_kws)
