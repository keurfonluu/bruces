from datetime import timedelta

import numpy as np
from numba import jit


_timedelta_like = (timedelta, np.timedelta64)
_scalar_like = (int, np.int32, np.int64, float, np.float32, np.float64)


def jitted(*args, **kwargs):
    """Custom :func:`jit` with default options."""
    kwargs.update(
        {
            "nopython": True,
            "nogil": True,
            # Disable fast-math flag "nnan" and "reassoc"
            # <https://llvm.org/docs/LangRef.html#fast-math-flags>
            "fastmath": {"ninf", "nsz", "arcp", "contract", "afn"},
            # "boundscheck": False,
            "cache": True,
        }
    )
    return jit(*args, **kwargs)


@jitted
def set_seed(seed):
    """Set random seed for numba."""
    np.random.seed(seed)


@jitted
def dist2d(x1, y1, x2, y2):
    """Euclidean distance in 2D."""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


@jitted
def dist3d(x1, y1, z1, x2, y2, z2):
    """Euclidean distance in 3D."""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5


@jitted
def time_space_distances(t, x, y, z, m, ti, xi, yi, zi, d=1.6, w=1.0, use_depth=False):
    """Calculate rescaled time and space distances."""
    N = len(t)

    eta_i = 20.0
    T_i = np.nan
    R_i = np.nan
    for j in range(N):
        t_ij = t[j] - ti

        # For each event, we are looking for its parent which corresponds
        # to the earliest event with the smallest proximity value
        # Inputs are not necessarily sorted w.r.t. time (e.g., step 2 of NND), thus we can't break here
        if t_ij >= 0.0:
            continue

        r_ij = (
            dist2d(xi, yi, x[j], y[j])
            if not use_depth
            else dist3d(xi, yi, zi, x[j], y[j], z[j])
        )

        # Skip events with the same epicenter
        if r_ij == 0.0:
            continue

        # Rewrite equations as log10 to avoid using power exponents
        # Computing log10 is much faster than power
        fac = -0.5 * w * m[j]
        T_ij = np.log10(-t_ij) + fac
        R_ij = d * np.log10(r_ij) + fac
        eta_ij = T_ij + R_ij

        if eta_ij < eta_i:
            eta_i = eta_ij
            T_i = T_ij
            R_i = R_ij

    return T_i, R_i


def timedelta_to_second(dt):
    """Convert timedelta_like to second."""
    if isinstance(dt, _timedelta_like):
        if isinstance(dt, np.timedelta64):
            dt = dt.astype("timedelta64[ms]").tolist()

        return dt.total_seconds()

    elif isinstance(dt, _scalar_like):
        return dt

    else:
        raise TypeError()


def timedelta_to_day(dt):
    """Convert timedelta_like to day."""
    if isinstance(dt, _timedelta_like):
        return timedelta_to_second(dt) / 86400.0

    elif isinstance(dt, _scalar_like):
        return dt

    else:
        raise TypeError()


def timedelta_to_year(dt):
    """Convert timedelta_like to year."""
    if isinstance(dt, _timedelta_like):
        return timedelta_to_second(dt) / 31557600.0

    elif isinstance(dt, _scalar_like):
        return dt

    else:
        raise TypeError()
