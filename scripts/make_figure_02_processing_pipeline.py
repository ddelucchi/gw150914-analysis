"""Figure 2 — single-detector (H1) conditioning sequence:
raw -> Tukey-windowed -> whitened -> whitened+bandpass-filtered.

Pure method figure: a 4 s segment centered on the geocenter merger
time, no inset, no event-zoom. Edge-affected regions are shaded in
panels (b)-(d).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import setup, load_event_strain, long_welch_asd
from src.preprocess import (tukey_with_transitions, whiten_explicit,
                            apply_bandpass)
from src.style import detector_color, save_figure


IFO = "H1"


def main() -> None:
    cfg, paths = setup()
    pp = cfg["processing_pipeline"]
    ts = load_event_strain(cfg, paths, IFO)
    fs = float(ts.sample_rate.value)

    # Centre the displayed segment ON the geocenter merger time so that
    # t = 0 sits in the middle of every panel.  The Welch ASD used for
    # whitening is computed from the full long stretch as elsewhere in
    # the paper.
    t_merger = float(cfg["event"]["gps_merger"])
    seg_len = float(pp["segment_length_s"])
    half = 0.5 * seg_len
    seg = ts.crop(t_merger - half, t_merger + half).value
    n = len(seg)
    t = np.arange(n) / fs - half  # symmetric about t = 0

    win = tukey_with_transitions(n, fs, pp["tukey_transition_s"])
    seg_win = seg * win

    f_asd, asd = long_welch_asd(ts, cfg)
    whit = whiten_explicit(seg_win, fs, f_asd, asd)
    band = apply_bandpass(whit, fs,
                          cfg["viz"]["bandpass_low_hz"],
                          cfg["viz"]["bandpass_high_hz"],
                          cfg["viz"]["butter_order"])

    fig, axes = plt.subplots(4, 1, figsize=(7.4, 8.6), sharex=True,
                             gridspec_kw={"hspace": 0.32,
                                          "left": 0.115,
                                          "right": 0.975,
                                          "top": 0.985,
                                          "bottom": 0.075})
    color = detector_color(cfg, IFO)

    # ---- panel (a): raw ------------------------------------------------
    axes[0].plot(t, seg, color=color, lw=0.6)
    axes[0].set_ylabel("raw strain")
    axes[0].set_title(f"(a) Raw {IFO} strain", loc="left", fontsize=9.5)

    # ---- panel (b): Tukey-windowed ------------------------------------
    axes[1].plot(t, seg_win, color=color, lw=0.6)
    axes[1].set_ylabel("windowed strain")
    axes[1].set_title("(b) Tukey-windowed strain",
                      loc="left", fontsize=9.5)

    # ---- panel (c): whitened ------------------------------------------
    axes[2].plot(t, whit, color=color, lw=0.6)
    axes[2].set_ylabel("whitened strain")
    axes[2].set_title("(c) Whitened strain", loc="left", fontsize=9.5)

    # ---- panel (d): whitened + bandpass -------------------------------
    axes[3].plot(t, band, color=color, lw=0.8)
    axes[3].set_ylabel("whitened, bandpassed\nstrain")
    axes[3].set_title(
        "(d) Whitened, bandpass-filtered strain (35\u2013350 Hz)",
        loc="left", fontsize=9.5)
    axes[3].set_xlabel("time from geocenter merger (s)")
    yfull = 1.10 * float(np.max(np.abs(band)))
    axes[3].set_ylim(-yfull, yfull)

    # ---- shade edge-affected regions in (b), (c), (d) ----------------
    tt = float(pp["tukey_transition_s"])
    for ax in axes[1:]:
        ax.axvspan(t[0], t[0] + tt, color="0.95", alpha=0.65, zorder=0)
        ax.axvspan(t[-1] - tt, t[-1], color="0.95", alpha=0.65, zorder=0)

    # ---- t = 0 reference epoch in every panel ------------------------
    for ax in axes:
        ax.axvline(0.0, color="k", ls="--", lw=0.8, alpha=0.55)
    y0_top = axes[0].get_ylim()[1]
    axes[0].text(0.0, 0.92 * y0_top,
                 r"  $t=0$: geocenter merger epoch (H1 in detector time)",
                 ha="left", va="top", fontsize=7.2, color="0.40")

    # symmetric x-limits (already symmetric, but enforce explicitly)
    axes[3].set_xlim(-half, +half)

    # No global super-title: the caption carries the explanatory load.
    save_figure(fig, paths.fig_main, "figure_02_processing_pipeline", cfg)


if __name__ == "__main__":
    main()

