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
def proximity(t, x, y, m, ti, xi, yi, d=1.5, w=1.0):
    """Calculate nearest-neighbor proximity."""
    N = len(t)

    eta_i = 1.0e20
    for j in range(N):
        t_ij = t[j] - ti

        # For each event, we are looking for its parent which corresponds
        # to the earliest event with the smallest proximity value
        if t_ij < 0.0:
            r_ij = ((x[j] - xi) ** 2 + (y[j] - yi) ** 2) ** 0.5

            # Skip events with the same epicenter
            if r_ij > 0.0:
                eta_ij = -t_ij * (r_ij * 1.0e-3) ** d * 10.0 ** (-w * m[j])
                eta_i = min(eta_i, eta_ij)

    return eta_i
