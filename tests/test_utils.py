import numpy as np
import pytest

import bruces


@pytest.mark.parametrize(
    "years_ref",
    [1971.5, np.random.uniform(1970.0, 1971.0, size=10)],
)
def test_datetime_conversion(years_ref):
    bruces.set_seed(42)

    dates = bruces.utils.to_datetime(years_ref)
    years = bruces.utils.to_decimal_year(dates)

    assert np.allclose(np.sum(years), np.sum(years_ref))
