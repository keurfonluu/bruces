import os
from datetime import timedelta

import helpers
import matplotlib.pyplot as plt
import numpy as np
import pytest

if not os.environ.get("DISPLAY", ""):
    plt.switch_backend("Agg")

cat = helpers.comcat()


def test_iter():
    easting = 0.0
    northing = 0.0
    depth = 0.0
    magnitude = 0.0

    for eq in cat:
        easting += eq.easting
        northing += eq.northing
        depth += eq.depth
        magnitude += eq.magnitude

    assert np.allclose(easting, cat.eastings.sum())
    assert np.allclose(northing, cat.northings.sum())
    assert np.allclose(depth, cat.depths.sum())
    assert np.allclose(magnitude, cat.magnitudes.sum())


def test_time_space_distances():
    T, R = cat.time_space_distances(d=1.6, w=1.0, use_depth=False, prune_nans=True)

    assert np.allclose(T.sum(), -2684.88968071)
    assert np.allclose(R.sum(), -913.38699270)


def test_fit_cutoff_threshold():
    eta_0 = cat.fit_cutoff_threshold(d=1.6, w=1.0, use_depth=False)

    assert np.allclose(eta_0, -4.58877803)


@pytest.mark.parametrize("kde", [True, False])
def test_plot_time_space_distances(kde, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    cat.plot_time_space_distances(
        d=1.6,
        w=1.0,
        use_depth=False,
        eta_0=-4.6,
        eta_0_diag=-4.6,
        kde=kde,
        bins=50,
    )


@pytest.mark.parametrize(
    "tbins, return_cumulative, sum_ref",
    [
        (timedelta(days=30), False, 7734.8),
        (timedelta(days=30), True, 4881.0),
        (
            np.arange(
                cat.origin_times[0].replace(tzinfo=None),
                cat.origin_times[-1].replace(tzinfo=None),
                timedelta(days=30),
                dtype="M8[ms]",
            ),
            True,
            4881.0,
        ),
    ],
)
def test_seismicity_rate(tbins, return_cumulative, sum_ref):
    s, _ = cat.seismicity_rate(tbins, return_cumulative)

    assert np.allclose(s.sum(), sum_ref)
