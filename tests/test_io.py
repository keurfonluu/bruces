import helpers
import numpy as np
import pytest

import bruces


@pytest.mark.parametrize(
    "filename, file_format",
    [("catalog.csv", "csv")],
)
def test_catalog(filename, file_format):
    filename = helpers.tempdir(filename)
    cat_ref = helpers.comcat()
    cat_ref.write(filename, file_format=file_format)
    cat = bruces.read_catalog(filename, file_format=file_format)

    assert np.allclose(cat.years, cat_ref.years)
    assert np.allclose(cat.latitudes, cat_ref.latitudes)
    assert np.allclose(cat.longitudes, cat_ref.longitudes)
    assert np.allclose(cat.eastings, cat_ref.eastings)
    assert np.allclose(cat.northings, cat_ref.northings)
    assert np.allclose(cat.depths, cat_ref.depths)
    assert np.allclose(cat.magnitudes, cat_ref.magnitudes)
