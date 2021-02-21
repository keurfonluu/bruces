from functools import partial

import numpy

from disba import surf96, swegn96
from disba._helpers import resample

from .._common import jitted

__all__ = [
    "seismogram",
]


pow = {
    "velocity": 1.0,
    "acceleration": 2.0,
}


@jitted
def hpulse(w, t):
    wt = 0.5 * w * t

    return numpy.exp(-1.0j * 2.0 * wt) * 4.0 * numpy.sin(0.5 * wt) ** 2.0 * numpy.sin(wt) / wt ** 3.0


@jitted
def tpulse(w, t):
    wt = w * t

    return numpy.exp(1.0j * wt) * numpy.sinc(0.5 * wt / numpy.pi) ** 2.0


@jitted
def fill_spectrum(x, n):
    """Return hermitian symmetric spectrum."""
    out = numpy.empty(n, dtype=numpy.complex_)
    out[0] = x[0].real

    if n % 2:
        out[1 : n // 2 + 1] = x
        out[n // 2 + 1:] = numpy.conjugate(x)[::-1]

    else:
        out[1 : n // 2] = x[:-1]
        out[n // 2] = 2.0 * x[-1].real  # Aliased
        out[n // 2 + 1:] = numpy.conjugate(x[:-1])[::-1]

    return out


def seismogram(
    velocity_model,
    time_sampling,
    time_max,
    source,
    mode_max=0,
    wave="both",
    moment_tensor=None,
    pulse=None,
    time_shift=None,
    freq_max=None,
    taper_width=None,
    z=0.0,
    dc=0.005,
    dt=0.025,
    dh=1.0,
    coord="zrt",
    out="displacement",
):
    if wave not in {"rayleigh", "love", "both"}:
        raise ValueError()

    else:
        if wave == "both":
            wave = ["love", "rayleigh"]

        else:
            wave = [wave]

    if out not in {"displacement", "velocity", "acceleration"}:
        raise ValueError()

    if coord not in {"zne", "zrt"}:
        raise ValueError()

    # Velocity model
    thickness, velocity_p, velocity_s, density = numpy.transpose(velocity_model)

    # Source parameters
    r, phi, h = source
    phi = (phi - 90.0) * 360.0
    phi = numpy.deg2rad(phi)
    cosp = numpy.cos(phi)
    sinp = numpy.sin(phi)
    cosp2 = cosp ** 2.0
    sinp2 = sinp ** 2.0

    # Components of moment tensor
    # Rescale to account for mixed units (return cm if M0 is dyne.cm)
    Mxx, Mxy, Mxz, Myx, Myy, Myz, Mzx, Mzy, Mzz = numpy.ravel(
        (
            moment_tensor
            if moment_tensor is not None
            else numpy.eye(3, dtype=numpy.float64)
        )
    ) * 1.0e-19

    # Pulse
    if pulse is not None:
        if isinstance(pulse, str):
            tau = float(pulse[1:])

            if pulse.startswith("h"):
                pulse = partial(hpulse, t=tau)

            elif pulse.startswith("t"):
                pulse = partial(tpulse, t=tau)

            else:
                raise NotImplementedError()

        elif hasattr(pulse, "__call__"):
            pass

        else:
            raise ValueError()

    # Frequency axis
    sampling_rate = 1.0 / time_sampling
    nsamples = time_max * sampling_rate + 1
    df = sampling_rate / nsamples
    nf = nsamples // 2 + 1
    freq = df * numpy.arange(1, nf)

    # Cutoff frequency and taper width
    if freq_max is not None:
        taper_width = taper_width if taper_width is not None else 1.0
        freq_max += taper_width

    else:
        freq_max = freq[-1]

    # Loop over modes
    t = 1.0 / freq[::-1]
    d, a, b, rho = resample(thickness, velocity_p, velocity_s, density, dh)
    
    depth = d.cumsum() - d[0]
    dz = numpy.diff(depth)
    nf = int(nf)
    nsamples = int(nsamples)

    Uz = numpy.zeros(nf - 1, dtype=numpy.complex_)
    Ur = numpy.zeros(nf - 1, dtype=numpy.complex_)
    Ut = numpy.zeros(nf - 1, dtype=numpy.complex_)
    for wave in wave:
        iwave = 1 if wave == "love" else 3

        for mode in range(mode_max + 1):
            c = surf96(t, d, a, b, rho, mode, 0, iwave, dc, dt)
            U = surf96(t, d, a, b, rho, mode, 1, iwave, dc, dt)

            for i, T in enumerate(t):
                if U[i] > 0.0:
                    if T * freq_max >= 1.0:
                        om = 2.0 * numpy.pi / T
                        kn = om / c[i]
                        knr = kn * r

                        egn = swegn96(T, d, a, b, rho, mode, iwave, dc)
                        if wave == "rayleigh":
                            r1 = egn[:, 0]
                            r2 = egn[:, 1]
                            I1 = 0.5 * numpy.dot(d, rho * (r1 ** 2 + r2 ** 2))

                            dr1dz = numpy.diff(r1) / dz
                            dr2dz = numpy.diff(r2) / dz
                            dr1dz = numpy.append(dr1dz, dr1dz[-1])
                            dr2dz = numpy.append(dr2dz, dr2dz[-1])

                            r1z = numpy.interp(z, depth, r1)
                            r2z = numpy.interp(z, depth, r2)
                            r1h = numpy.interp(h, depth, r1)
                            r2h = numpy.interp(h, depth, r2)
                            dr1dzh = numpy.interp(h, depth, dr1dz)
                            dr2dzh = numpy.interp(h, depth, dr2dz)

                            cst = kn * r1h * (Mxx * cosp2 + (Mxy + Myx) * sinp * cosp + Myy * sinp2)
                            cst += 1.0j * dr1dzh * (Mxz * cosp + Myz * sinp)
                            cst += -1.0j * kn * r2h * (Mzx * cosp + Mzy * sinp)
                            cst += dr2dzh * Mzz
                            cst *= 0.125 / c[i] / U[i] / I1 * (2.0 / numpy.pi / knr) ** 0.5

                            Uz[i] += r2z * numpy.exp(1.0j * (knr + 0.25 * numpy.pi)) * cst
                            Ur[i] += r1z * numpy.exp(1.0j * (knr - 0.25 * numpy.pi)) * cst

                        else:
                            l1 = egn[:, 0]
                            I1 = 0.5 * numpy.dot(d, rho * l1 ** 2)

                            dl1dz = numpy.diff(l1) / dz
                            dl1dz = numpy.append(dl1dz, dl1dz[-1])

                            l1z = numpy.interp(z, depth, l1)
                            l1h = numpy.interp(h, depth, l1)
                            dl1dzh = numpy.interp(h, depth, dl1dz)

                            cst = 1.0j * kn * l1h * (Mxx * sinp * cosp - Myx * cosp2 + Mxy * sinp2 - Myy * sinp * cosp)
                            cst += -dl1dzh * (Mxz * sinp - Myz * cosp)
                            cst *= 0.125 / c[i] / U[i] / I1 * (2.0 / numpy.pi / knr) ** 0.5

                            Ut[i] += l1z * numpy.exp(1.0j * (knr + 0.25 * numpy.pi)) * cst

                else:
                    break

    # Working back with frequency axis
    Uz = Uz[::-1]
    Ur = Ur[::-1]
    Ut = Ut[::-1]

    # Convolve with source wavelet
    if pulse is not None:
        om = 2.0 * numpy.pi * freq
        wavelet = pulse(om)

        Uz *= wavelet
        Ur *= wavelet
        Ut *= wavelet

    # Apply time shift (useful for when pulse is wrapped around)
    if time_shift:
        om = 2.0 * numpy.pi * freq
        tshift = numpy.exp(1.0j * om * time_shift)
        Uz *= tshift
        Ur *= tshift
        Ut *= tshift

    # Apply lowpass filter with sin taper to limit ringing
    if taper_width:
        lpfilt = numpy.zeros(nf, dtype=numpy.float64)
        i1 = int(freq_max // df // 2)
        i2 = int(taper_width // df // 2)
        step = 0.5 * numpy.pi / i2

        lpfilt[:i1] = 1.0
        for i in range(i2):
            lpfilt[i1 + i] = numpy.sin(0.5 * numpy.pi - i * step) ** 2.0

        Uz *= lpfilt[1:]
        Ur *= lpfilt[1:]
        Ut *= lpfilt[1:]

    # Multiply by iw depending on output type (differentiate w.r.t. time)
    if out in pow:
        om = 2.0 * numpy.pi * freq
        iw = (1.0j * om) ** pow[out]

        Uz *= iw
        Ur *= iw
        Ut *= iw

    # Append mirrored spectrum for negative frequencies
    Uz = fill_spectrum(Uz, nsamples)
    Ur = fill_spectrum(Ur, nsamples)
    Ut = fill_spectrum(Ut, nsamples)

    # Transform signal to time domain
    uz = numpy.fft.ifft(Uz)[::-1].real
    ur = numpy.fft.ifft(Ur)[::-1].real
    ut = numpy.fft.ifft(Ut)[::-1].real

    if coord == "zrt":
        return uz, ur, ut

    elif coord == "zne":
        # R -> N, T -> E (rotate by phi + pi/2)
        un = -ur * cosp + ut * sinp
        ue = -ur * sinp - ut * cosp

        return uz, un, ue
