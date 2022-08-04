import numpy as np
from numba import jit


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
def time_space_distances(t, x, y, m, ti, xi, yi, d=1.6, w=1.0):
    """Calculate rescaled time and space distances."""
    N = len(t)

    eta_i = 20.0
    T_i = np.nan
    R_i = np.nan
    for j in range(N):
        t_ij = t[j] - ti

        # For each event, we are looking for its parent which corresponds
        # to the earliest event with the smallest proximity value
        if t_ij < 0.0:
            r_ij = np.sqrt((x[j] - xi) ** 2.0 + (y[j] - yi) ** 2.0)

            # Skip events with the same epicenter
            if r_ij > 0.0:
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
