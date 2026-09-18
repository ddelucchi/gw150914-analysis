"""Conditioning routines: visualization-mode and analysis-mode.

Visualization mode (Discovery-paper / detector-noise-guide style)
----------------------------------------------------------------
    window -> whiten (ASD from long Welch average) -> bandpass [35,350] Hz
    plus narrow notches for instrumental lines.  This is FOR DISPLAY ONLY,
    not the LVC statistical pipeline.

Analysis mode (matched-filter style, after the GWpy SNR example)
----------------------------------------------------------------
    highpass at 15 Hz, estimate PSD with 4 s segments / 2 s overlap,
    crop to a short analysis segment, low-frequency cutoff 15 Hz.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, welch
from scipy.signal.windows import tukey


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load_strain(path: Path):
    from gwpy.timeseries import TimeSeries
    return TimeSeries.read(path, format="hdf5")


# ---------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------
def tukey_window(n: int, alpha: float) -> np.ndarray:
    return tukey(n, alpha=alpha)


def tukey_with_transitions(n: int, fs: float, transition_s: float) -> np.ndarray:
    """Tukey window whose taper region length matches `transition_s` seconds
    on each side (matches the data-guide 'Tukey with 0.5 s transitions')."""
    alpha = min(1.0, 2.0 * transition_s / (n / fs))
    return tukey(n, alpha=alpha)


# ---------------------------------------------------------------------
# Welch ASD on a long reference stretch
# ---------------------------------------------------------------------
def welch_asd(strain: np.ndarray, fs: float, seglen_s: float, step_s: float,
              tukey_alpha: float = 0.25,
              average: str = "mean") -> Tuple[np.ndarray, np.ndarray]:
    nper = int(seglen_s * fs)
    nover = int((seglen_s - step_s) * fs)
    win = tukey(nper, alpha=tukey_alpha)
    f, psd = welch(strain, fs=fs, window=win, nperseg=nper,
                   noverlap=nover, detrend=False, scaling="density",
                   average=average)
    return f, np.sqrt(psd)


# ---------------------------------------------------------------------
# Whitening (explicit, inspectable: FFT / ASD / iFFT)
# ---------------------------------------------------------------------
def whiten_explicit(seg: np.ndarray, fs: float,
                    asd_freqs: np.ndarray, asd_vals: np.ndarray,
                    rescale_unit_var: bool = True) -> np.ndarray:
    n = len(seg)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.fft.rfft(seg)
    asd_interp = np.interp(freqs, asd_freqs, asd_vals,
                           left=asd_vals[0], right=asd_vals[-1])
    asd_interp = np.where(asd_interp <= 0, np.inf, asd_interp)
    norm = np.sqrt(fs / 2.0)        # unit-variance normalization
    white_spec = spec / (asd_interp * norm)
    out = np.fft.irfft(white_spec, n=n)
    if rescale_unit_var:
        out = (out - out.mean()) / (out.std() + 1e-30)
    return out


# ---------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------
def bandpass_sos(fs: float, lo: float, hi: float, order: int = 8):
    nyq = 0.5 * fs
    return butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")


def apply_bandpass(x: np.ndarray, fs: float, lo: float, hi: float,
                   order: int = 8) -> np.ndarray:
    sos = bandpass_sos(fs, lo, hi, order)
    return sosfiltfilt(sos, x)


def apply_notches(x: np.ndarray, fs: float, freqs, q: float = 30.0) -> np.ndarray:
    y = x.copy()
    for f0 in freqs:
        b, a = iirnotch(f0 / (fs / 2), Q=q)
        # Use sosfiltfilt via second-order conversion
        from scipy.signal import tf2sos
        sos = tf2sos(b, a)
        y = sosfiltfilt(sos, y)
    return y


def highpass_sos(fs: float, fc: float, order: int = 4):
    return butter(order, fc / (0.5 * fs), btype="high", output="sos")


def apply_highpass(x: np.ndarray, fs: float, fc: float,
                   order: int = 4) -> np.ndarray:
    return sosfiltfilt(highpass_sos(fs, fc, order), x)


# ---------------------------------------------------------------------
# Composite: viz-mode pipeline on a short event segment
# ---------------------------------------------------------------------
def viz_condition(seg: np.ndarray, fs: float, cfg: Dict[str, Any],
                  asd_freqs: np.ndarray, asd_vals: np.ndarray) -> np.ndarray:
    win = tukey_window(len(seg), alpha=cfg["viz"]["tukey_alpha"])
    w = whiten_explicit(seg * win, fs, asd_freqs, asd_vals)
    bp = apply_bandpass(w, fs, cfg["viz"]["bandpass_low_hz"],
                        cfg["viz"]["bandpass_high_hz"],
                        cfg["viz"]["butter_order"])
    return apply_notches(bp, fs, cfg["viz"]["notches_hz"])


# ---------------------------------------------------------------------
# Composite: analysis-mode pre-filter (matched filter prep)
# ---------------------------------------------------------------------
def analysis_precondition(strain_ts, cfg: Dict[str, Any]):
    """Highpass at cfg.analysis.highpass_hz; returns gwpy.TimeSeries."""
    return strain_ts.highpass(cfg["analysis"]["highpass_hz"])


# ---------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------
def extract_segment(strain_ts, t_center: float, half_window_s: float):
    return strain_ts.crop(t_center - half_window_s, t_center + half_window_s)

