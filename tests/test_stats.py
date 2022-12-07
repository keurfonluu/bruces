from datetime import datetime

import numpy as np
import pytest

import bruces


@pytest.mark.parametrize(
    "times, rates, mean_ref",
    [
        ((datetime(1991, 1, 1), datetime(1992, 1, 1)), 10.0, 1991.49976736),
        ((1991.0, 1992.0), 10.0, 1991.49976736),
        ((1991.0, 1992.0), (10.0, 10.0), 1991.35864531),
    ]
)
def test_poisson_process(times, rates, mean_ref):
    bruces.set_seed(42)

    out = bruces.stats.poisson_process(times, rates)
    out = bruces.utils.to_decimal_year(out)

    assert np.allclose(out.mean(), mean_ref)


@pytest.mark.parametrize("size, sum_ref", [(1, 0.14557157), (10, 3.18575312)])
def test_sample_magnitude(size, sum_ref):
    bruces.set_seed(42)

    out = bruces.stats.sample_magnitude(0.0, 4.2, 1.4, size=size)

    assert np.allclose(np.sum(out), sum_ref)
