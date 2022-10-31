import helpers
import pytest

cat = helpers.comcat()


@pytest.mark.parametrize(
    "algorithm, decluster_kws, nev",
    [
        ("gardner-knopoff", {}, 154),
        (
            "nearest-neighbor",
            {
                "d": 1.6,
                "w": 1.0,
                "eta_0": -5.1,
                "alpha_0": 1.5,
                "use_depth": True,
                "M": 100,
                "seed": 0,
            },
            420,
        ),
        (
            "nearest-neighbor",
            {
                "d": 1.6,
                "w": 1.0,
                "eta_0": -5.1,
                "alpha_0": 1.5,
                "use_depth": False,
                "M": 100,
                "seed": 0,
            },
            381,
        ),
        (
            "reasenberg",
            {
                "rfact": 10,
                "xmeff": 3.0,
                "xk": 0.5,
                "taumin": 1.0,
                "taumax": 10.0,
                "p": 0.95,
            },
            442,
        ),
    ],
)
def test_decluster(algorithm, decluster_kws, nev):
    catd = cat.decluster(algorithm, **decluster_kws)

    assert len(catd) == nev
