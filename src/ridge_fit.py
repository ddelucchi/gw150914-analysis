"""Direct-data chirp-mass estimate.

Two estimators are provided:

1. `fit_chirp_mass_zerocross` — naive zero-crossing instantaneous frequency.
   Pedagogical, fragile in the presence of broadband noise.

2. `fit_chirp_mass_ridge` — extract f(t) as the argmax of a Q-transform-like
   spectrogram of the whitened strain, then fit y = f^{-8/3}(t) linearly.
   This is what is *actually used* by figure 6 since it is robust enough
   to recover the GW150914 chirp.

In both cases the chirp mass follows from

    (G M_c / c^3)^{5/3} = (5/256) pi^{-8/3} (-slope)
    M_c = (c^3/G) * (G M_c / c^3)

This is a pedagogical estimator; it is NOT the LVC posterior."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# SI constants
C_SI = 2.99792458e8        # m/s
G_SI = 6.67430e-11         # m^3 kg^-1 s^-2
MSUN_KG = 1.98892e30


@dataclass
class ChirpFit:
    t: np.ndarray
    f: np.ndarray
    y: np.ndarray            # f^{-8/3}
    slope: float
    intercept: float
    slope_err: float
    chirp_mass_msun: float
    chirp_mass_err_msun: float


def zero_crossings(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return interpolated upward zero-crossing times."""
    s = np.sign(x)
    s[s == 0] = 1
    idx = np.nonzero(np.diff(s) > 0)[0]
    out = []
    for i in idx:
        x0, x1 = x[i], x[i + 1]
        t0, t1 = t[i], t[i + 1]
        if x1 == x0:
            out.append(0.5 * (t0 + t1))
        else:
            out.append(t0 - x0 * (t1 - t0) / (x1 - x0))
    return np.asarray(out)


def instantaneous_frequency(zc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Centered finite-difference inverse half-period."""
    t_mid = 0.5 * (zc[1:] + zc[:-1])
    f = 1.0 / (zc[1:] - zc[:-1])     # one full period = 2 successive up-crossings? -> use successive zero crossings of any sign for half-period
    return t_mid, f


def fit_chirp_mass(t: np.ndarray, x: np.ndarray,
                   f_min: float, f_max: float) -> ChirpFit:
    """Fit y = f^{-8/3} = a t + b on the requested frequency band."""
    zc = zero_crossings(t, x)
    if len(zc) < 4:
        raise RuntimeError("Too few zero crossings for chirp-mass fit")
    t_mid, f = instantaneous_frequency(zc)
    mask = (f >= f_min) & (f <= f_max) & np.isfinite(f)
    t_mid, f = t_mid[mask], f[mask]
    if len(f) < 4:
        raise RuntimeError("Too few in-band zero crossings")
    # Keep only the monotonically chirping segment (drop post-merger
    # ringdown / noise crossings whose f decreases relative to running max).
    keep = np.ones_like(f, dtype=bool)
    fmax_so_far = -np.inf
    for i, fi in enumerate(f):
        if fi >= fmax_so_far - 2.0:   # 2 Hz tolerance
            fmax_so_far = max(fmax_so_far, fi)
        else:
            keep[i] = False
    t_mid, f = t_mid[keep], f[keep]
    if len(f) < 4:
        raise RuntimeError("Too few monotonic in-band crossings")
    y = f ** (-8.0 / 3.0)
    A = np.vstack([t_mid, np.ones_like(t_mid)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = coef
    # error from residuals
    y_pred = A @ coef
    sigma2 = np.mean((y - y_pred) ** 2)
    cov = sigma2 * np.linalg.inv(A.T @ A)
    slope_err = float(np.sqrt(cov[0, 0]))

    # Convert slope -> chirp mass.
    #   d/dt(f^{-8/3}) = -(8/3) f^{-11/3} f_dot
    # => f^{-11/3} f_dot = -(3/8) * slope
    # Standard relation: (G M_c / c^3)^{5/3} = (5/96) pi^{-8/3} f^{-11/3} f_dot
    #                                        = (5/256) pi^{-8/3} * (-slope)
    # then  M_c = (c^3 / G) * [(5/256) pi^{-8/3} (-slope)]^{3/5}
    coef53 = (5.0 / 256.0) * np.pi ** (-8.0 / 3.0) * (-slope)
    if coef53 <= 0:
        raise RuntimeError("Non-physical (negative) chirp-mass slope")
    g_m_over_c3 = coef53 ** (3.0 / 5.0)                # seconds
    mc_kg = (C_SI ** 3 / G_SI) * g_m_over_c3            # kilograms
    # propagate slope uncertainty:  d M_c / d slope = -(3/5) M_c / slope
    mc_err_kg = (3.0 / 5.0) * mc_kg / abs(slope) * slope_err
    return ChirpFit(t=t_mid, f=f, y=y,
                    slope=float(slope), intercept=float(intercept),
                    slope_err=slope_err,
                    chirp_mass_msun=float(mc_kg / MSUN_KG),
                    chirp_mass_err_msun=float(mc_err_kg / MSUN_KG))


# ---------------------------------------------------------------------
# Ridge-based estimator (used by figure 6).  Uses a short-time FFT of
# the whitened strain to track the chirp argmax frequency over time.
# ---------------------------------------------------------------------
def stft_ridge(t: np.ndarray, x: np.ndarray, fs: float,
               t_window_s: float, t_step_s: float,
               f_min: float, f_max: float
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (t_mid, f_peak) — the argmax STFT frequency in [f_min, f_max]."""
    nperseg = int(round(t_window_s * fs))
    nstep = max(1, int(round(t_step_s * fs)))
    n = len(x)
    win = np.hanning(nperseg)
    half = nperseg // 2
    centers = np.arange(half, n - half, nstep)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    band = (freqs >= f_min) & (freqs <= f_max)
    fb = freqs[band]
    t_out, f_out = [], []
    for c in centers:
        seg = x[c - half:c - half + nperseg] * win
        spec = np.abs(np.fft.rfft(seg))
        if not band.any():
            continue
        ib = int(np.argmax(spec[band]))
        # parabolic refinement
        sb = spec[band]
        if 0 < ib < len(sb) - 1:
            a, b, cc = sb[ib - 1], sb[ib], sb[ib + 1]
            denom = a - 2 * b + cc
            if denom != 0:
                delta = 0.5 * (a - cc) / denom
                f_peak = fb[ib] + delta * (fb[1] - fb[0])
            else:
                f_peak = fb[ib]
        else:
            f_peak = fb[ib]
        t_out.append(t[c])
        f_out.append(f_peak)
    return np.asarray(t_out), np.asarray(f_out)


def fit_chirp_mass_ridge(t: np.ndarray, x: np.ndarray, fs: float,
                         t_window_s: float = 0.020,
                         t_step_s: float = 0.005,
                         f_min: float = 35.0,
                         f_max: float = 200.0) -> ChirpFit:
    """STFT-ridge estimator for f(t), then linear fit of f^{-8/3} vs t."""
    t_mid, f = stft_ridge(t, x, fs, t_window_s, t_step_s, f_min, f_max)
    if len(f) < 4:
        raise RuntimeError("Too few STFT ridge samples")
    # keep only monotonically rising portion (inspiral), discard ringdown
    keep = np.ones_like(f, dtype=bool)
    fmax_so_far = -np.inf
    for i, fi in enumerate(f):
        if fi >= fmax_so_far - 1.0:
            fmax_so_far = max(fmax_so_far, fi)
        else:
            keep[i] = False
    t_mid, f = t_mid[keep], f[keep]
    if len(f) < 4:
        raise RuntimeError("Too few monotonic ridge samples")
    y = f ** (-8.0 / 3.0)
    A = np.vstack([t_mid, np.ones_like(t_mid)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = coef
    y_pred = A @ coef
    sigma2 = np.mean((y - y_pred) ** 2)
    cov = sigma2 * np.linalg.inv(A.T @ A)
    slope_err = float(np.sqrt(cov[0, 0]))
    coef53 = (5.0 / 256.0) * np.pi ** (-8.0 / 3.0) * (-slope)
    if coef53 <= 0:
        raise RuntimeError("Non-physical (negative) chirp-mass slope")
    g_m_over_c3 = coef53 ** (3.0 / 5.0)
    mc_kg = (C_SI ** 3 / G_SI) * g_m_over_c3
    mc_err_kg = (3.0 / 5.0) * mc_kg / abs(slope) * slope_err
    return ChirpFit(t=t_mid, f=f, y=y,
                    slope=float(slope), intercept=float(intercept),
                    slope_err=slope_err,
                    chirp_mass_msun=float(mc_kg / MSUN_KG),
                    chirp_mass_err_msun=float(mc_err_kg / MSUN_KG))

