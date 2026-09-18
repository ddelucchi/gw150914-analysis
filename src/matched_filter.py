"""Single-detector matched-filter SNR.

Tries PyCBC first (per the official GWpy/PyCBC tutorial), then falls back
to a SciPy/NumPy implementation in src.synth_waveform that uses a
TaylorF2 inspiral template and an explicit PSD-weighted correlation
(Allen et al. 2012).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _matched_filter_pycbc(strain_ts, cfg, crop_gps, m1, m2):
    from pycbc.filter import matched_filter
    from pycbc.waveform import get_td_waveform
    a = cfg["analysis"]
    cond = strain_ts.highpass(a["highpass_hz"])
    psd = cond.psd(a["psd_segment_s"], a["psd_overlap_s"])
    seg = cond.crop(*crop_gps).to_pycbc()
    psd_pc = psd.to_pycbc()
    hp, _ = get_td_waveform(approximant=a["template_approximant"],
                            mass1=m1, mass2=m2,
                            delta_t=1.0 / float(strain_ts.sample_rate.value),
                            f_lower=a["template_f_low_hz"])
    hp.resize(len(seg))
    template = hp.cyclic_time_shift(hp.start_time)
    snr = matched_filter(template, seg, psd=psd_pc,
                         low_frequency_cutoff=a["matched_filter_low_freq_cutoff_hz"])
    snr = snr.crop(4, 4)
    return np.asarray(snr.sample_times), np.abs(np.asarray(snr))


def _matched_filter_scipy(strain_ts, cfg, crop_gps, m1, m2):
    from src.preprocess import welch_asd
    from src.synth_waveform import time_domain_chirp, matched_filter_snr_series

    a = cfg["analysis"]
    cond = strain_ts.highpass(a["highpass_hz"])
    fs = float(cond.sample_rate.value)

    # Median-PSD over the same long stretch as the GWpy/PyCBC tutorial
    # (32 s centred on event with 4 s segments; median statistic
    # suppresses transient outliers far better than mean Welch).
    t_c = 0.5 * (crop_gps[0] + crop_gps[1])
    half_psd = 16.0
    psd_seg = cond.crop(t_c - half_psd, t_c + half_psd).value
    f_asd, asd = welch_asd(psd_seg, fs, seglen_s=4.0, step_s=2.0,
                           average="median")
    psd = asd ** 2

    seg = cond.crop(*crop_gps)
    s = np.asarray(seg.value)
    duration = float(crop_gps[1] - crop_gps[0])

    # TaylorF2 inspiral template (well-behaved; FD-built then iFFT'd, so
    # no (t_c-t)^{-1/4} divergence to spoil the matched-filter norm).
    template_tc = 0.7 * duration
    _, h = time_domain_chirp(fs, duration, m1, m2,
                             f_low=a["template_f_low_hz"],
                             tc=template_tc)
    snr = matched_filter_snr_series(
        s, h, fs, f_asd, psd,
        f_low=a["matched_filter_low_freq_cutoff_hz"])
    # The cyclic FFT-correlation places the peak at an offset equal to
    # (signal_index - template_tc_index) mod n.  Roll the SNR series so
    # that the template-coalescence-aligns-with-signal lag corresponds
    # to t = (start_of_segment) + duration/2 -- i.e. centred near merger
    # for a segment that is symmetric about merger.
    n_shift = int(round(template_tc * fs))
    snr = np.roll(snr, n_shift)
    times = np.asarray(seg.times.value)
    return times, snr


def matched_filter_snr(strain_ts, cfg: Dict[str, Any],
                       crop_gps: Tuple[float, float],
                       m1: float | None = None,
                       m2: float | None = None):
    a = cfg["analysis"]
    m1 = a["template_mass1_msun"] if m1 is None else m1
    m2 = a["template_mass2_msun"] if m2 is None else m2
    try:
        return _matched_filter_pycbc(strain_ts, cfg, crop_gps, m1, m2)
    except Exception as exc:
        print(f"[info] PyCBC unavailable ({exc.__class__.__name__}); "
              f"using SciPy/TaylorF2 fallback.")
        return _matched_filter_scipy(strain_ts, cfg, crop_gps, m1, m2)

