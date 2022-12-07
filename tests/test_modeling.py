import datetime

import numpy as np

import bruces


def test_etas():
    bruces.set_seed(42)

    origin_times = bruces.stats.poisson_process((1970.0, 1971.0), 10.0)
    n = len(origin_times)

    cat = bruces.Catalog(
        origin_times=origin_times,
        eastings=np.random.uniform(-1.0, 1.0, n),
        northings=np.random.uniform(-1.0, 1.0, n),
        magnitudes=bruces.stats.sample_magnitude(0.0, 4.2, 1.4, n),
    )
    cat = bruces.modeling.etas(
        cat,
        end_time=1970.5,
        mc=0.0,
        theta=0.2,
        alpha=0.5,
        c=0.001,
        K=0.5,
        b=1.4,
        d=0.1,
    )

    assert len(cat) == 18


def test_seismicity_rate():
    n = 101
    t = np.linspace(0.0, 10.0, n)
    s = np.ones(n) * 1.0e-3
    s[t > 2.0] = 1.0
    s[t > 4.0] = 1.0e-1

    r = bruces.modeling.seismicity_rate(
        times=t,
        stress=s,
        stress_ini=1.0e-3,
        asigma=0.15,
        first_step=datetime.timedelta(days=1),
        max_step=datetime.timedelta(days=30),
        reduce_step_factor=4.0,
        rtol=1.0e-3,
    )

    assert np.allclose(r.sum(), 19337.18387050)


def test_magnitude_time():
    n = 101
    t = np.linspace(0.0, 10.0, n)
    r = 100.0 * np.exp(-((t - 2.0) ** 2))

    mags = bruces.modeling.magnitude_time(
        times=t,
        rates=r,
        m_bounds=[0.0, 2.0],
        n=50,
        b_value=1.0,
        seed=0,
    )

    assert np.allclose(np.concatenate(mags).sum(), 68.91649999)
