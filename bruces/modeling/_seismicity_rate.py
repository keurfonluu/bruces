from datetime import datetime, timedelta

import numpy as np
from numba import prange

from .._common import jitted
from .._helpers import to_decimal_year

__all__ = [
    "seismicity_rate",
]


_C = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 4.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0],
        [1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0],
        [439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0],
        [-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0],
    ],
    dtype=np.float64,
)

_W = np.array(
    [
        [25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4101.0, -1.0 / 5.0, 0.0],
        [
            16.0 / 135.0,
            0.0,
            6656.0 / 12825.0,
            28561.0 / 56430.0,
            -9.0 / 50.0,
            2.0 / 55.0,
        ],
    ],
    dtype=np.float64,
)


@jitted
def rate(t, s, s0i, tci, tcrit, tmax, dt, dtmax, dtfac, rtol):
    """
    Solve rate-and-state ODE using Runge-Kutta-Fehlberg method.

    Note
    ----
    See <http://maths.cnam.fr/IMG/pdf/RungeKuttaFehlbergProof.pdf>.

    """
    nt = len(t)
    i, ti = 0, t[0]
    times, rates = [ti], [1.0]
    while ti < tmax:
        si = s[i] * s0i if ti >= tcrit else 1.0

        # Calculate derivatives
        r0 = rates[-1]
        k0 = dt * r0 * tci * (si - r0)
        r1 = rates[-1] + _C[1, 0] * k0
        k1 = dt * r1 * tci * (si - r1)
        r2 = rates[-1] + _C[2, 0] * k0 + _C[2, 1] * k1
        k2 = dt * r2 * tci * (si - r2)
        r3 = rates[-1] + _C[3, 0] * k0 + _C[3, 1] * k1 + _C[3, 2] * k2
        k3 = dt * r3 * tci * (si - r3)
        r4 = rates[-1] + _C[4, 0] * k0 + _C[4, 1] * k1 + _C[4, 2] * k2 + _C[4, 3] * k3
        k4 = dt * r4 * tci * (si - r4)
        r5 = (
            rates[-1]
            + _C[5, 0] * k0
            + _C[5, 1] * k1
            + _C[5, 2] * k2
            + _C[5, 3] * k3
            + _C[5, 4] * k4
        )
        k5 = dt * r5 * tci * (si - r5)

        # Calculate error
        ro4 = rates[-1] + _W[0, 0] * k0 + _W[0, 2] * k2 + _W[0, 3] * k3 + _W[0, 4] * k4
        ro5 = (
            rates[-1]
            + _W[1, 0] * k0
            + _W[1, 2] * k2
            + _W[1, 3] * k3
            + _W[1, 4] * k4
            + _W[1, 5] * k5
        )
        eps = max(np.abs(ro5 - ro4) / ro5, 1.0e-16)

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
                if 0.0 < dti < dt:
                    dt = dti
                    i += 1

    return np.interp(t, times, rates)


@jitted(parallel=True)
def rate_vectorized(t, s, s0i, tci, tcrit, tmax, dt, dtmax, dtfac, rtol):
    """Solve ODE for different integration points."""
    ns = len(s)
    out = np.empty(s.shape, dtype=np.float64)
    for i in prange(ns):
        out[i] = rate(t, s[i], s0i[i], tci[i], tcrit, tmax, dt, dtmax, dtfac, rtol)

    return out


def seismicity_rate(
    times,
    stress,
    stress_ini,
    asigma,
    t_crit=None,
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
    times : sequence of scalar or sequence of datetime_like
        Dates for every stressing rate samples (in years if scalar). Common to all integration points.
    stress : sequence of scalar
        Stressing rates or sequence for every integration points.
    stress_ini : scalar or sequence of scalar
        Background stressing rate or sequence for every integration points.
    asigma : scalar or sequence of scalar
        Free parameter for rate-and-state constitutive model or sequence for every integration points.
    t_crit : datetime_like or None, optional, default None
        Critical time. Default is `times[0]`.
    t_bound : datetime_like or None, optional, default None
        Boundary time.
    first_step : scalar, timedelta_like or None, optional, default None
        Initial time step size (in years if scalar). Default is 1 day.
    max_step : scalar, timedelta_like or None, optional, default None
        Maximum allowed time step size (in years if scalar). Default is 1 month.
    reduce_step_factor : scalar, optional, default 4.0
        Factor by which time step size is reduced for unsuccessful time step.
    rtol : scalar, optional, default 1.0e-5
        Relative convergence tolerance.

    Returns
    -------
    array_like
        Seismicity rates or list of seismicity rates for every integration points. Returned seismicity rates are relative to background seismicity rate.

    """

    def check_parameter(x, ndim, npts):
        """Check consistency of input parameter."""
        if np.ndim(x) == 0:
            x = x if ndim == 1 else np.full(npts, x, dtype=np.float64)

        else:
            x = np.asarray(x, dtype=np.float64)
            if x.size != npts:
                raise ValueError()

        return x

    # Check stressing rate
    s = np.copy(stress)
    ndim = s.ndim
    if len(times) != (s.size if ndim == 1 else s.shape[1]):
        raise ValueError()

    # Check consistency of parameters related to integration points
    npts = 1 if ndim == 1 else len(s)
    s0 = check_parameter(stress_ini, ndim, npts)
    asig = check_parameter(asigma, ndim, npts)

    # Convert datetimes to decimal years
    t = (
        to_decimal_year(times)
        if isinstance(times, (datetime, np.datetime64))
        else times
    )

    # Set time stepping parameters
    tcrit = to_decimal_year(t_crit) if t_crit is not None else t[0]
    tmax = to_decimal_year(t_bound) if t_bound is not None else t[-1]
    dt = first_step if first_step is not None else 1.0 / 365.25
    dtmax = max_step if max_step is not None else 1.0 / 12.0
    dtfac = reduce_step_factor

    if isinstance(dt, np.timedelta64):
        dt = dt.astype("timedelta64[ms]").tolist()
    if isinstance(dt, timedelta):
        dt = dt.total_seconds() / 3600.0 / 24.0 / 365.25

    if isinstance(dtmax, np.timedelta64):
        dtmax = dtmax.astype("timedelta64[ms]").tolist()
    if isinstance(dtmax, timedelta):
        dtmax = dtmax.total_seconds() / 3600.0 / 24.0 / 365.25

    # Use inverse to avoid repeated zero division check
    tci = s0 / asig
    s0i = 1.0 / s0

    # Add background stressing rate to stressing rate
    s += s0 if ndim == 1 else s0[:, None]

    # Solve ODE
    if ndim == 1:
        return rate(t, s, s0i, tci, tcrit, tmax, dt, dtmax, dtfac, rtol)

    else:
        return rate_vectorized(t, s, s0i, tci, tcrit, tmax, dt, dtmax, dtfac, rtol)
