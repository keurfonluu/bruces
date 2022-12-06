import numpy as np

from .._common import jitted

__all__ = [
    "sample_magnitude",
]


@jitted
def grmag(low=0.0, high=None, b=1.0, size=1):
    beta = np.log(10.0 ** b)
    u = np.random.rand(size)
    u *= 1.0 - np.exp(-beta * (high - low)) if high is not None else 1.0

    return low - np.log(1.0 - u) / beta


def sample_magnitude(low=0.0, high=None, b=1.0, size=1):
    """
    Draw magnitude samples following Gutenberg-Richter law.

    Parameters
    ----------
    low : scalar, optional, default 0.0
        Minimum magnitude.
    high : scalar or None, optional, default None
        Maximum magnitude.
    b : scalar, optional, default 1.0
        b-value.
    size : int, optional, default 1
        Number of samples.

    Returns
    -------
    scalar or array_like
        Sampled magnitudes.
    
    """
    m = grmag(low, high, b, size)

    return m[0] if size == 1 else m
