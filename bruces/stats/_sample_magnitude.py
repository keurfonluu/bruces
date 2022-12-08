import numpy as np

__all__ = [
    "sample_magnitude",
]


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
    beta = 2.30258509 * b  # np.log(10**b)
    u = np.random.rand(size) if size > 1 else np.random.rand()
    u *= 1.0 - np.exp(-beta * (high - low)) if high is not None else 1.0

    return low - np.log(1.0 - u) / beta
