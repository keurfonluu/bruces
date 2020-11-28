import numpy
from numba import prange

from .._common import jitted
from .._helpers import to_decimal_year

__all__ = [
    "seismicity_rate",
]


_C = numpy.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 4.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0],
        [1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0],
        [439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0],
        [-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0],
    ],
    dtype=numpy.float64,
)

_W = numpy.array(
    [
        [25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4101.0, -1.0 / 5.0, 0.0],
        [16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0],
    ],
    dtype=numpy.float64,
)


@jitted
def rate_rhs(r, s, tci):
    """Right hand side of rate-and-state ODE (Segall and Lu, 2015)."""
    return r * tci * (s - r)


@jitted
def pdot(a, b, n):
    """Partial dot product."""
    out = 0.0
    for i in range(n):
        out += a[i] * b[i]

    return out


@jitted
def rate(t, s, s0i, tci, tmax, dt, dtmax, dtfac, rtol):
    """
    Solve rate-and-state ODE using Runge-Kutta-Fehlberg method.

    Note
    ----
    See <http://maths.cnam.fr/IMG/pdf/RungeKuttaFehlbergProof.pdf>.

    """
    nt = len(t)
    i, ti = 0, t[0]
    times, rates = [ti], [1.0]
    k = numpy.empty(6, dtype=numpy.float64)
    while ti < tmax:
        si = s[i] * s0i

        # Calculate derivatives
        for j, c in enumerate(_C):
            k[j] = dt * rate_rhs(rates[-1] + pdot(c, k, j), si, tci)

        # Calculate error
        ro4 = rates[-1] + pdot(_W[0], k, 5)
        ro5 = rates[-1] + pdot(_W[1], k, 6)
        eps = max(numpy.abs(ro5 - ro4) / ro5, 1.0e-16)

        # Check convergence
        if eps > rtol:
            dt /= dtfac

        else:
            rates.append(ro5)
            times.append(ti)
            ti += dt

            # Update step size
            dt *= 0.84 * (rtol / eps) ** 0.25
            dt = min(dt, dtmax, tmax - ti)

            # Check current stressing rate
            if i + 1 < nt:
                dti = t[i + 1] - ti
                if dt > dti:
                    dt = dti
                    i += 1

    return numpy.interp(t, times, rates)


@jitted(parallel=True)
def rate_vectorized(t, s, s0i, tci, tmax, dt, dtmax, dtfac, rtol):
    """Solve ODE for different integration points."""
    ns = len(s)
    out = numpy.empty(s.shape, dtype=numpy.float64)
    for i in prange(ns):
        out[i] = rate(t, s[i], s0i, tci, tmax, dt, dtmax, dtfac, rtol)

    return out


def seismicity_rate(
    times,
    stress,
    stress_ini,
    asigma,
    t_bound=None,
    first_step=None,
    max_step=None,
    reduce_step_factor=4.0,
    rtol=1.0e-5,
    ):
    """
    Rate-and-state modeling of seismicity rate.
    
    Parameters
    ----------
    times : array_like
        Dates for every stressing rate samples.
    stress : array_like
        Stressing rates or list of stressing rates for every integration points.
    stress_ini : scalar
        Background stressing rate.
    asigma : scalar
        Free parameter for rate-and-state constitutive model.
    t_bound : datetime.datetime or None, optional, default None
        Boundary time.
    first_step : scalar or None, optional, default None
        Initial time step size (in years). Default is 1 day.
    max_step : scalar or None, optional, default None
        Maximum allowed time step size (in years). Default is 1 month.
    reduce_step_factor : scalar, optional, default 4.0
        Factor by which time step size is reduced for unsuccessful time step.
    rtol : scalar, optional, default 1.0e-5
        Relative convergence tolerance.

    Returns
    -------
    array_like
        Seismicity rates or list of seismicity rates for every integration points.
    
    """
    # Convert datetimes to decimal years
    t = to_decimal_year(times)
    nt = len(t)

    # Check stressing rate
    s = numpy.asarray(stress) + stress_ini
    ndim = s.ndim
    npts = s.size if ndim == 1 else s.shape[1]
    if npts != nt:
        raise ValueError()

    # Set time stepping parameters
    tmax = to_decimal_year(t_bound) if t_bound is not None else t[-1]
    dt = first_step if first_step is not None else 1.0 / 365.25
    dtmax = max_step if max_step is not None else 1.0 / 12.0
    dtfac = reduce_step_factor

    # Use inverse to avoid repeated zero division check
    tci = stress_ini / asigma
    s0i = 1.0 / stress_ini

    # Solve ODE
    if ndim == 1:
        return rate(t, s, s0i, tci, tmax, dt, dtmax, dtfac, rtol)

    else:
        return rate_vectorized(t, s, s0i, tci, tmax, dt, dtmax, dtfac, rtol)
