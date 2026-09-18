"""Appendix - projected ML waveform & residuals (gwastro 1811.04071).

Source of the projected-waveform / residual files
-------------------------------------------------
The four .txt files in ``data/reconstruction/`` are taken from
``gwastro/gw150914_investigation`` (Nitz, Capano et al., arXiv
1811.04071), which itself derives them from the LVC GW150914 strain
release plus the maximum-likelihood waveform parameters of Biwer et
al. (arXiv 1807.10312).  The repository convention (see ``res.py``)
is that each file's time column is given relative to the integer GPS
floor of the event, ``event_time = 1126259462``:

    absolute_GPS = file_time + 1126259462

So if our config sets ``gps_merger`` to the geocenter merger epoch,
then ``t_plot = file_time + 1126259462 - gps_merger`` puts the
reconstruction on the same merger-relative axis as our data, with no
ad-hoc cross-correlation alignment needed.

Conditioning convention
-----------------------
The released files contain the projected ML waveform and the residual
in PHYSICAL strain (units strain * 1e21), bandpassed and notched but
**not whitened**.  To make a coherent overlay we therefore apply only
the bandpass + notches to our raw strain, also in physical units, and
do not whiten.  This makes the top and bottom rows internally
consistent: identical processing on the same physical observable.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import setup, load_event_strain
from src.preprocess import (tukey_with_transitions, apply_bandpass,
                            apply_notches)
from src.style import detector_color, save_figure


RECON_DIR_NAME = "reconstruction"
GWASTRO_EVENT_TIME = 1126259462  # integer GPS floor used by gwastro repo


def _load_reconstruction(paths, kind, ifo):
    fname = f"fig1-{kind}-{ifo[0]}.txt"
    fpath = paths.raw.parent / RECON_DIR_NAME / fname
    arr = np.loadtxt(fpath, comments="#")
    return arr[:, 0], arr[:, 1]  # (t in file convention, strain*1e21)


def _bandpassed_strain_1e21(cfg, paths, ifo, t_around_merger=(-0.30, 0.30)):
    """Tukey + bandpass [35,350] Hz + notches on raw strain, returned
    in physical units of strain * 1e21 on a time axis where t = 0 is
    cfg.event.gps_merger (geocenter)."""
    pp = cfg["processing_pipeline"]
    ts = load_event_strain(cfg, paths, ifo)
    fs = float(ts.sample_rate.value)

    t_c_seg = pp["segment_center_gps"]
    half_seg = 0.5 * pp["segment_length_s"]
    seg = ts.crop(t_c_seg - half_seg, t_c_seg + half_seg).value
    n = len(seg)
    win = tukey_with_transitions(n, fs, pp["tukey_transition_s"])
    sb = apply_bandpass(seg * win, fs,
                        cfg["viz"]["bandpass_low_hz"],
                        cfg["viz"]["bandpass_high_hz"],
                        cfg["viz"]["butter_order"])
    sb = apply_notches(sb, fs, cfg["viz"]["notches_hz"])

    t_rel = (np.arange(n) / fs - half_seg)
    t_rel += (t_c_seg - cfg["event"]["gps_merger"])
    keep = (t_rel >= t_around_merger[0]) & (t_rel <= t_around_merger[1])
    return t_rel[keep], sb[keep] * 1e21, fs


def main():
    cfg, paths = setup()
    win = (-0.08, 0.05)
    coh_win = (-0.03, 0.03)  # signal-dominated comparison window

    # File-time -> "time from merger" using the GPS convention from
    # gwastro/res.py: absolute_GPS = file_time + GWASTRO_EVENT_TIME.
    t_offset = GWASTRO_EVENT_TIME - float(cfg["event"]["gps_merger"])

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1.0],
                                          "hspace": 0.30,
                                          "wspace": 0.18,
                                          "left": 0.075,
                                          "right": 0.985,
                                          "top": 0.875,
                                          "bottom": 0.092})

    summary = {}
    cache = {}
    y_top_max = 0.0
    y_bot_max = 0.0
    for ifo in ("H1", "L1"):
        t_d, x_d_1e21, fs = _bandpassed_strain_1e21(cfg, paths, ifo)

        t_w_file, w_1e21 = _load_reconstruction(paths, "waveform", ifo)
        t_r_file, r_1e21 = _load_reconstruction(paths, "residual", ifo)
        t_w = t_w_file + t_offset
        t_r = t_r_file + t_offset

        keep_w = (t_w >= win[0] - 0.005) & (t_w <= win[1] + 0.005)
        keep_r = (t_r >= win[0] - 0.005) & (t_r <= win[1] + 0.005)
        x_on_w = np.interp(t_w, t_d, x_d_1e21, left=0.0, right=0.0)
        rms_d = float(np.sqrt(np.mean(x_on_w[keep_w] ** 2)))
        rms_r_inwin = float(np.sqrt(np.mean(r_1e21[keep_r] ** 2)))
        var_explained = (max(0.0, 1.0 - (rms_r_inwin / rms_d) ** 2)
                         if rms_d > 0 else 0.0)
        summary[ifo] = (rms_d, rms_r_inwin, var_explained)

        cache[ifo] = (t_d, x_d_1e21, t_w, w_1e21, t_r, r_1e21,
                      keep_w, keep_r, x_on_w)

        y_top_max = max(y_top_max,
                        float(np.max(np.abs(w_1e21[keep_w]))),
                        float(np.max(np.abs(x_on_w[keep_w]))))
        y_bot_max = max(y_bot_max,
                        float(np.max(np.abs(r_1e21[keep_r]))))

    y_top = 1.25 * y_top_max
    y_bot = 1.25 * y_bot_max

    panel_letters = {("top", "H1"): "(a)", ("top", "L1"): "(b)",
                     ("bot", "H1"): "(c)", ("bot", "L1"): "(d)"}

    for col, ifo in enumerate(("H1", "L1")):
        (t_d, x_d_1e21, t_w, w_1e21, t_r, r_1e21,
         keep_w, keep_r, x_on_w) = cache[ifo]

        ax_top = axes[0, col]
        ax_top.axvspan(coh_win[0], coh_win[1], color="0.93", alpha=0.7,
                       zorder=0)
        ax_top.plot(t_d, x_d_1e21, color=detector_color(cfg, ifo), lw=0.9,
                    alpha=0.42, label=f"{ifo} bandpassed detector data")
        ax_top.plot(t_w, w_1e21, color="0.10", lw=1.9,
                    label="released ML waveform")
        ax_top.axhline(0, color="0.85", lw=0.4)
        ax_top.axvline(0, color="0.45", ls="--", lw=0.7, alpha=0.85)
        ax_top.set_ylabel(r"strain ($10^{-21}$)", fontsize=9)
        ax_top.set_title(f"{panel_letters[('top', ifo)]} {ifo} overlay",
                         fontsize=10, fontweight="normal", loc="left")
        ax_top.legend(loc="upper right", fontsize=7.5, frameon=False,
                      handlelength=1.4, handletextpad=0.5,
                      labelspacing=0.3)
        ax_top.tick_params(labelsize=8)
        ax_top.set_xlim(*win)
        ax_top.set_ylim(-y_top, y_top)

        ax_bot = axes[1, col]
        ax_bot.axvspan(coh_win[0], coh_win[1], color="0.93", alpha=0.7,
                       zorder=0)
        ax_bot.plot(t_r, r_1e21, color=detector_color(cfg, ifo), lw=1.0)
        ax_bot.axhline(0, color="0.55", lw=0.7)
        ax_bot.axvline(0, color="0.45", ls="--", lw=0.7, alpha=0.85)
        ax_bot.set_xlabel("time from geocenter merger (s)", fontsize=9)
        ax_bot.set_ylabel(r"residual ($10^{-21}$)", fontsize=9)
        ax_bot.set_title(
            f"{panel_letters[('bot', ifo)]} {ifo} — released residual",
            fontsize=9.5, fontweight="normal", loc="left")
        ax_bot.tick_params(labelsize=8)
        ax_bot.set_xlim(*win)
        ax_bot.set_ylim(-y_bot, y_bot)

    fig.suptitle(
        "GW150914 detector strain compared with the released "
        "maximum-likelihood waveform and residuals",
        fontsize=10.5, y=0.965,
    )
    save_figure(fig, paths.fig_main,
                "figure_06_projected_waveform", cfg)

    for ifo, (rd, rr, ve) in summary.items():
        print(f"  {ifo}: rms_data={rd:.3f}, rms_residual={rr:.3f}, "
              f"var.explained={100*ve:.1f}%")


if __name__ == "__main__":
    main()
