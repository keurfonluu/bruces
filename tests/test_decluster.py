import helpers
import pytest

cat = helpers.comcat()


@pytest.mark.parametrize(
    "algorithm, decluster_kws, nev",
    [
        (
            "gardner-knopoff",
            {"window": "default"},
            154,
        ),
        (
            "gardner-knopoff",
            {"window": "gruenthal"},
            52,
        ),
        (
            "gardner-knopoff",
            {"window": "uhrhammer"},
            430,
        ),
        (
            "nearest-neighbor",
            {
                "d": 1.6,
                "w": 1.0,
                "eta_0": -5.1,
                "alpha_0": 2.0,
                "use_depth": True,
                "M": 100,
                "seed": 42,
            },
            477,
        ),
        (
            "reasenberg",
            {
                "rfact": 10,
                "xmeff": 3.0,
                "xk": 0.5,
                "tau_min": 1.0,
                "tau_max": 10.0,
                "p": 0.95,
            },
            442,
        ),
    ],
)
def test_decluster(algorithm, decluster_kws, nev):
    catd = cat.decluster(algorithm, **decluster_kws)

    assert len(catd) == nev
