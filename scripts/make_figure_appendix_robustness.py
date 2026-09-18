"""Appendix robustness sweep: vary windowing / PSD / bandpass and show
how the spectrogram and the Hilbert chirp-mass estimate move under
each choice.

Bottom panels use the SAME Hilbert instantaneous-frequency estimator
as Figure 6 (early-inspiral 35-100 Hz, t in [-0.06, -0.003] s, 4 ms
median binning, linear fit in f^-8/3 vs t).  This guarantees that
the robustness panels are consistent with the appendix Mc figure;
the previous STFT-ridge variant produced a code-path failure
(Mc ~ 4 M_sun) that contradicted Fig. 6.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sosfiltfilt

from _common import setup, load_event_strain, long_welch_asd
from src.preprocess import (welch_asd, whiten_explicit, apply_bandpass,
                            tukey_with_transitions, bandpass_sos)
from src.style import save_figure

from make_figure_06_chirp_mass_fit import fit_chirp_mass_hilbert


def _whitened(ts, cfg):
    """Whitened L1 segment (no bandpass yet) on the same 4 s window
    used for the Hilbert fit in Fig. 6."""
    pp = cfg["processing_pipeline"]
    fs = float(ts.sample_rate.value)
    t_c_seg = pp["segment_center_gps"]
    half = 0.5 * pp["segment_length_s"]
    seg = ts.crop(t_c_seg - half, t_c_seg + half).value
    n = len(seg)
    win = tukey_with_transitions(n, fs, pp["tukey_transition_s"])
    f_asd, asd = long_welch_asd(ts, cfg)
    seg_w = whiten_explicit(seg * win, fs, f_asd, asd)
    t_seg = t_c_seg - half + np.arange(n) / fs
    return t_seg, seg_w, fs


def _hilbert_mc(t_seg, seg_w, fs, t_c_event, lo_hz, hi_hz,
                f_lo_fit, f_hi_fit):
    sos = bandpass_sos(fs, lo_hz, hi_hz)
    seg_bp = sosfiltfilt(sos, seg_w)
    try:
        fit = fit_chirp_mass_hilbert(t_seg, seg_bp, fs, t_c_event,
                                     f_lo=f_lo_fit, f_hi=f_hi_fit,
                                     t_min_rel=-0.06, t_max_rel=-0.003)
        return fit.chirp_mass_msun, fit.chirp_mass_err_msun
    except Exception:
        return float("nan"), float("nan")


def main() -> None:
    cfg, paths = setup()
    rb = cfg["robustness"]
    t_c_event = cfg["event"]["gps_merger"]
    ts = load_event_strain(cfg, paths, "L1")
    fs = float(ts.sample_rate.value)
    t_seg, seg_w, _ = _whitened(ts, cfg)

    f_asd, asd = long_welch_asd(ts, cfg)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))

    # (a) Tukey alpha sweep on a 4 s segment ASD
    pp = cfg["processing_pipeline"]
    seg4 = ts.crop(t_c_event - pp["segment_length_s"]/2,
                   t_c_event + pp["segment_length_s"]/2).value
    for a in rb["tukey_alpha_grid"]:
        f, asd_a = welch_asd(seg4, fs, pp["welch_chunk_s"],
                             pp["welch_step_s"], a)
        axes[0, 0].loglog(f, asd_a, lw=0.9, label=fr"$\alpha={a}$")
    axes[0, 0].set_xlim(8, fs/2); axes[0, 0].set_xlabel("frequency (Hz)")
    axes[0, 0].set_ylabel(r"ASD (strain $/\sqrt{\rm Hz}$)")
    axes[0, 0].set_title(r"Tukey $\alpha$ sweep on 4 s segment")
    axes[0, 0].legend(fontsize=8)

    # (b) PSD seglen sweep
    for L in rb["psd_seglen_grid_s"]:
        sub = (seg4 if L <= pp["segment_length_s"]
               else ts.crop(t_c_event - L/2, t_c_event + L/2).value)
        f, asd_l = welch_asd(sub, fs, L, L/2)
        axes[0, 1].loglog(f, asd_l, lw=0.9, label=f"seglen={L} s")
    axes[0, 1].set_xlim(8, fs/2); axes[0, 1].set_xlabel("frequency (Hz)")
    axes[0, 1].set_ylabel(r"ASD (strain $/\sqrt{\rm Hz}$)")
    axes[0, 1].set_title("PSD segment-length sweep")
    axes[0, 1].legend(fontsize=8)

    # (c) Hilbert Mc vs conditioning bandpass low edge.  We hold the
    # Hilbert *fit* band fixed at [35, 100] Hz (Fig. 6 default) and
    # only vary the *signal-conditioning* bandpass low edge.  When the
    # conditioning low edge crawls above ~35 Hz it erases the bottom
    # of the fit band and the estimator degrades.
    HI_COND = 110.0
    F_FIT_LO = 35.0
    F_FIT_HI = 100.0
    mc, mc_err = [], []
    for lo in rb["bandpass_low_grid_hz"]:
        m, e = _hilbert_mc(t_seg, seg_w, fs, t_c_event, lo, HI_COND,
                           F_FIT_LO, F_FIT_HI)
        mc.append(m); mc_err.append(e)
    axes[1, 0].errorbar(rb["bandpass_low_grid_hz"], mc, yerr=mc_err,
                        fmt="o-", color="C0", lw=1.2, ms=6, capsize=3)
    axes[1, 0].axhspan(28, 35, color="0.85", alpha=0.5,
                       label=r"PRL/GWTC-1 $\mathcal{M}_c^{\rm det}\sim31\,M_\odot$")
    axes[1, 0].axvline(F_FIT_LO, color="0.4", ls=":", lw=0.8,
                       label=r"Hilbert fit-anchor low edge (35 Hz, fixed)")
    axes[1, 0].set_xlabel("conditioning bandpass low edge (Hz)")
    axes[1, 0].set_ylabel(r"$\mathcal{M}_c$ (M$_\odot$)")
    axes[1, 0].set_ylim(0, 80)
    axes[1, 0].set_title(r"Hilbert $\mathcal{M}_c$ vs conditioning low edge")
    axes[1, 0].legend(fontsize=8, loc="upper right")

    # (d) Hilbert Mc vs conditioning bandpass high edge.  Conditioning
    # low edge held at 30 Hz; only the high cutoff varies.  When the
    # conditioning high edge climbs much above the fit-band high edge
    # (100 Hz), the analytic-signal frequency is dragged toward the
    # mid-band noise floor and the chirp is lost.
    LO_COND = 30.0
    mc, mc_err = [], []
    for hi in rb["bandpass_high_grid_hz"]:
        m, e = _hilbert_mc(t_seg, seg_w, fs, t_c_event, LO_COND, hi,
                           F_FIT_LO, F_FIT_HI)
        mc.append(m); mc_err.append(e)
    axes[1, 1].errorbar(rb["bandpass_high_grid_hz"], mc, yerr=mc_err,
                        fmt="s-", color="C3", lw=1.2, ms=6, capsize=3)
    axes[1, 1].axhspan(28, 35, color="0.85", alpha=0.5,
                       label=r"PRL/GWTC-1 $\mathcal{M}_c^{\rm det}\sim31\,M_\odot$")
    axes[1, 1].axvline(F_FIT_HI, color="0.4", ls=":", lw=0.8,
                       label=r"Hilbert fit-anchor high edge (100 Hz, fixed)")
    axes[1, 1].set_xlabel("conditioning bandpass high edge (Hz)")
    axes[1, 1].set_ylabel(r"$\mathcal{M}_c$ (M$_\odot$)")
    axes[1, 1].set_ylim(0, 80)
    axes[1, 1].set_title(r"Hilbert $\mathcal{M}_c$ vs conditioning high edge")
    axes[1, 1].legend(fontsize=8, loc="upper right")

    fig.suptitle("Robustness sweep — windowing, PSD, conditioning bandpass.\n"
                 "Bottom row: dotted vertical lines mark the FIXED "
                 "Hilbert fit-anchor band [35, 100] Hz, NOT the swept "
                 "conditioning band.",
                 y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save_figure(fig, paths.fig_appendix, "figure_appendix_robustness", cfg)


if __name__ == "__main__":
    main()

