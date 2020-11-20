import numpy

from numba import jit


def jitted(*args, **kwargs):
    """Custom :func:`jit` with default options."""
    kwargs.update(
        {
            "nopython": True,
            "nogil": True,
            "fastmath": True,
            # "boundscheck": False,
            "cache": True,
        }
    )
    return jit(*args, **kwargs)


@jitted
def rescaled_time_distance(t, x, y, m, ti, xi, yi, d=1.5, w=0.0):
    """Calculate rescaled time and distance."""
    N = len(t)

    eta_i = 1.0e20
    T_i = numpy.nan
    R_i = numpy.nan
    for j in range(N):
        t_ij = t[j] - ti

        # For each event, we are looking for its parent which corresponds
        # to the earliest event with the smallest proximity value
        if t_ij < 0.0:
            r_ij = ((x[j] - xi) ** 2.0 + (y[j] - yi) ** 2.0) ** 0.5

            # Skip events with the same epicenter
            if r_ij > 0.0:
                fac = 10.0 ** (-0.5 * w * m[j])
                T_ij = -t_ij * fac
                R_ij = r_ij ** d * fac
                eta_ij = T_ij * R_ij

                if eta_ij < eta_i:
                    eta_i = eta_ij
                    T_i = T_ij
                    R_i = R_ij

    return T_i, R_i
