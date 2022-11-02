import pathlib

import pandas as pd

import bruces


def comcat():
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
    origin_times = pd.to_datetime(df["Origin time"], utc=False).to_numpy()
    eastings = df["Easting"].to_numpy()
    northings = df["Northing"].to_numpy()
    depths = df["Depth"].to_numpy()
    magnitudes = df["Magnitude"].to_numpy()

    return bruces.Catalog(
        origin_times=origin_times,
        eastings=eastings,
        northings=northings,
        depths=depths,
        magnitudes=magnitudes,
    )
