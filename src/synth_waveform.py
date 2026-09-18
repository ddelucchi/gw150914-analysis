"""SciPy-only TaylorF2 inspiral waveform + simple merger taper.

This module provides a fallback when lalsimulation / PyCBC / PESummary
are unavailable (e.g. native Windows Python).  It is *not* IMRPhenomPv2,
but it produces a phase-correct frequency-domain inspiral chirp from
component masses, with a soft amplitude taper near the ISCO frequency
that mimics merger/ringdown decay well enough for visual overlays and
for matched-filter SNR demonstrations.

References:
  Buonanno, Iyer, Ochsner, Pan, Sathyaprakash, PRD 80, 084043 (2009).
  Allen et al., PRD 85, 122006 (2012)  — matched filter formalism.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# SI constants (CODATA / IAU)
C_SI = 2.99792458e8
G_SI = 6.67430e-11
MSUN_KG = 1.98892e30
MSUN_S = G_SI * MSUN_KG / C_SI ** 3            # ~4.925e-6 s
PC_M = 3.0856775814913673e16
MPC_M = 1e6 * PC_M


def f_isco(m_total_msun: float) -> float:
    """Schwarzschild ISCO orbital frequency (GW frequency = 2 * f_orb)."""
    m_s = m_total_msun * MSUN_S
    return 1.0 / (6.0 ** 1.5 * np.pi * m_s)


def chirp_mass_msun(m1: float, m2: float) -> float:
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def taylorf2_strain(freqs: np.ndarray, m1_msun: float, m2_msun: float,
                    distance_mpc: float = 410.0,
                    inclination: float = 0.0,
                    f_low: float = 20.0,
                    f_high: float | None = None,
                    phi0: float = 0.0,
                    tc: float = 0.0) -> np.ndarray:
    """Stationary-phase TaylorF2 (3.5 PN phase, Newtonian amplitude),
    one-sided spectrum h~(f) for a non-spinning binary, plus a smooth
    high-frequency rolloff at f_isco to fake merger decay."""
    m1 = m1_msun * MSUN_KG; m2 = m2_msun * MSUN_KG
    M = m1 + m2; mu = m1 * m2 / M
    eta = mu / M
    mc = chirp_mass_msun(m1_msun, m2_msun) * MSUN_KG
    m_total_s = M * G_SI / C_SI ** 3

    fmax = f_high if f_high is not None else f_isco(m1_msun + m2_msun)
    out = np.zeros_like(freqs, dtype=np.complex128)
    band = (freqs >= f_low) & (freqs <= fmax)
    f = freqs[band]

    # Newtonian amplitude (sky-/inclination-averaged simplification)
    d = distance_mpc * MPC_M
    amp = (np.sqrt(5.0 / 24.0) / (np.pi ** (2.0 / 3.0))
           * (G_SI * mc / C_SI ** 3) ** (5.0 / 6.0)
           * C_SI / d) * f ** (-7.0 / 6.0)
    amp *= 0.5 * (1.0 + np.cos(inclination) ** 2)

    # 3.5 PN phase (Buonanno+ 2009 eqs. 3.18 — drop the log term that
    # introduces v[0] sensitivity; the 0–2.5 PN truncation suffices for
    # visual templates and matched-filter SNR demos).
    v = (np.pi * m_total_s * f) ** (1.0 / 3.0)
    psi = 3.0 / (128.0 * eta * v ** 5) * (
        1.0
        + (3715.0 / 756.0 + 55.0 / 9.0 * eta) * v ** 2
        - 16.0 * np.pi * v ** 3
        + (15293365.0 / 508032.0 + 27145.0 / 504.0 * eta
           + 3085.0 / 72.0 * eta ** 2) * v ** 4
    )
    phase = 2.0 * np.pi * f * tc - phi0 - np.pi / 4.0 + psi

    # Soft hi-f taper to imitate merger-ringdown amplitude decay
    f_taper = 0.6 * fmax
    taper = 1.0 / (1.0 + np.exp((f - f_taper) / (0.05 * fmax)))
    h = amp * taper * np.exp(-1j * phase)
    out[band] = h
    return out


def time_domain_chirp(fs: float, duration_s: float,
                      m1_msun: float, m2_msun: float,
                      f_low: float = 20.0,
                      tc: float | None = None,
                      distance_mpc: float = 410.0,
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse-FFT the TaylorF2 waveform onto a real time series of length
    fs*duration_s.  tc is the merger time relative to the segment start;
    defaults to 0.7*duration_s so the chirp is well inside the window."""
    n = int(fs * duration_s)
    if n % 2:
        n += 1
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    t_c = 0.7 * duration_s if tc is None else tc
    h_f = taylorf2_strain(freqs, m1_msun, m2_msun,
                          distance_mpc=distance_mpc,
                          f_low=f_low, tc=t_c)
    h_t = np.fft.irfft(h_f, n=n) * fs
    t = np.arange(n) / fs
    return t, h_t


# ---------------------------------------------------------------------
# Simple matched filter: psd-weighted complex correlation
# ---------------------------------------------------------------------
def matched_filter_snr_series(strain: np.ndarray, template: np.ndarray,
                              fs: float, psd_freqs: np.ndarray,
                              psd_vals: np.ndarray,
                              f_low: float = 15.0) -> np.ndarray:
    """Return |rho(t)|, a numerically standard PSD-weighted matched-filter
    SNR time series.  Both inputs are real time series of equal length."""
    n = len(strain)
    if len(template) < n:
        template = np.pad(template, (0, n - len(template)))
    elif len(template) > n:
        template = template[:n]

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    sf = np.fft.rfft(strain)
    hf = np.fft.rfft(template)

    psd = np.interp(freqs, psd_freqs, psd_vals,
                    left=psd_vals[0], right=psd_vals[-1])
    psd = np.where((freqs < f_low) | (psd <= 0), np.inf, psd)

    # Standard matched-filter SNR (Allen et al. 2012, eq. 3.4):
    #   sigma^2 = 4 / (fs * n) * sum |h_f|^2 / S(f)
    #   z(t)    = (4 / fs) * irfft( s_f h_f* / S(f) )
    #   rho(t)  = |z(t)| / sigma
    hh = 4.0 / (fs * n) * np.real(np.sum(hf * np.conj(hf) / psd))
    integrand = sf * np.conj(hf) / psd
    rho_t = (4.0 / fs) * np.fft.irfft(integrand, n=n)
    return np.abs(rho_t) / np.sqrt(max(hh, 1e-300))


# =====================================================================
# Time-domain Newtonian chirp + ringdown (visual fallback)
# =====================================================================
def newtonian_chirp(t: np.ndarray, tc: float, m1_msun: float, m2_msun: float,
                    f_low: float = 20.0, phi_c: float = 0.0,
                    amplitude: float = 1.0,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Leading-order PN inspiral on a real time grid.

        f_gw(t) = (1/pi) * (5 / (256*Mc_s))^{3/8} * (tc - t)^{-3/8}
        Phi(t)  = -2 * ((tc - t) / (5*Mc_s))^{5/8}
        h(t)    = A * (pi*Mc_s*f(t))^{2/3} * cos(Phi(t) + phi_c)

    Zeroed outside [f_low, f_isco] and for t >= tc.
    """
    mc_s = chirp_mass_msun(m1_msun, m2_msun) * MSUN_S
    f_max = f_isco(m1_msun + m2_msun)
    tau = tc - t
    h = np.zeros_like(t); f = np.zeros_like(t)
    valid = tau > 0
    if not np.any(valid):
        return h, f
    tau_v = tau[valid]
    f_v = (1.0 / np.pi) * (5.0 / (256.0 * mc_s)) ** (3.0 / 8.0) * tau_v ** (-3.0 / 8.0)
    band = (f_v >= f_low) & (f_v <= f_max)
    if not np.any(band):
        return h, f
    phi = -2.0 * (tau_v / (5.0 * mc_s)) ** (5.0 / 8.0)
    h_v = amplitude * (np.pi * mc_s * f_v) ** (2.0 / 3.0) * np.cos(phi + phi_c)
    h_v = np.where(band, h_v, 0.0)
    f_v = np.where(band, f_v, 0.0)
    idx = np.nonzero(valid)[0]
    h[idx] = h_v
    f[idx] = f_v
    return h, f


def ringdown_tail(t: np.ndarray, tc: float,
                  f_rd: float = 250.0, tau_ms: float = 4.0,
                  amplitude: float = 1.0, phi_rd: float = 0.0) -> np.ndarray:
    out = np.zeros_like(t)
    after = t >= tc
    if not np.any(after):
        return out
    dt = t[after] - tc
    out[after] = (amplitude * np.exp(-dt / (tau_ms * 1e-3))
                  * np.cos(2.0 * np.pi * f_rd * dt + phi_rd))
    return out


def imr_waveform(t: np.ndarray, tc: float, m1_msun: float, m2_msun: float,
                 f_low: float = 20.0, amplitude: float = 1.0,
                 f_rd: float = 250.0, tau_rd_ms: float = 4.0) -> np.ndarray:
    """Inspiral (Newtonian) + smooth Hann ramp-out + damped-sinusoid
    ringdown stitched at tc with amplitude continuity."""
    h_in, _ = newtonian_chirp(t, tc, m1_msun, m2_msun,
                              f_low=f_low, amplitude=amplitude)
    pre = (t < tc) & (t > tc - 0.01)
    a_peak = float(np.max(np.abs(h_in[pre]))) if np.any(pre) else amplitude
    taper_t = 0.005
    pre_tap = (t < tc) & (t > tc - taper_t)
    if np.any(pre_tap):
        x = (tc - t[pre_tap]) / taper_t
        h_in[pre_tap] *= 0.5 * (1.0 - np.cos(np.pi * x))
    h_rd = ringdown_tail(t, tc, f_rd=f_rd, tau_ms=tau_rd_ms, amplitude=a_peak)
    return h_in + h_rd

