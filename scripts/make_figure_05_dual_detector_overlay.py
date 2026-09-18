"""Figure 5 — H1/L1 whitened-strain overlay after inter-site delay correction
and polarity alignment.

Two-panel design:
  (a) wide context window, with the merger comparison interval shaded;
  (b) zoomed merger window with a quantitative normalized
      cross-correlation peak annotated in-panel.

Both panels share identical preprocessing: 4 s Tukey-windowed segments,
whitened by the 1024 s Welch ASD, bandpassed [35, 350] Hz, with narrow
instrumental lines notched. H1 is shifted by -6.9 ms and sign-flipped to
account for the expected inter-detector delay and antenna-response polarity.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from _common import setup, load_event_strain, long_welch_asd
from src.preprocess import viz_condition, extract_segment
from src.style import detector_color, save_figure


# Display windows (s relative to merger).
T_LO_WIDE, T_HI_WIDE = -0.10, +0.05
T_LO_ZOOM, T_HI_ZOOM = -0.020, +0.040
# Comparison window for the cross-correlation metric and the shaded band.
T_LO_CMP, T_HI_CMP = T_LO_ZOOM, T_HI_ZOOM


def _norm_xcorr(x: np.ndarray, y: np.ndarray, fs: float,
                max_lag_s: float = 0.010):
    """Return (lags_s, normalized cross-correlation) of x vs y.

    Inputs are first standardized to zero mean and unit variance over the
    supplied window so the peak value lies in [-1, +1].
    """
    x = (x - x.mean()) / (x.std() + 1e-30)
    y = (y - y.mean()) / (y.std() + 1e-30)
    n = len(x)
    full = np.correlate(x, y, mode="full") / n
    lags = (np.arange(full.size) - (n - 1)) / fs
    sel = np.abs(lags) <= max_lag_s
    return lags[sel], full[sel]


def main() -> None:
    cfg, paths = setup()
    t_c = cfg["event"]["gps_merger"]
    delay = cfg["event"]["hl_delay_ms"] * 1e-3
    half = 0.5

    panels = {}
    for ifo in ("H1", "L1"):
        ts = load_event_strain(cfg, paths, ifo)
        fs = float(ts.sample_rate.value)
        f_asd, asd = long_welch_asd(ts, cfg)
        seg = extract_segment(ts, t_c, half).value
        t = np.arange(len(seg)) / fs - half
        panels[ifo] = (t, viz_condition(seg, fs, cfg, f_asd, asd), fs)

    t, l1, fs = panels["L1"]
    th, h1, _ = panels["H1"]

    # Normalize each whitened series by the off-source RMS so the two
    # detectors are visually comparable (whitening prefactor is arbitrary).
    off = (t >= -0.45) & (t <= -0.10)
    l1 = l1 / np.std(l1[off])
    h1 = h1 / np.std(h1[off])
    h_shift = -np.interp(t, th - delay, h1)

    # Quantitative metric: normalized cross-correlation in the comparison
    # window, peak value and peak lag relative to the applied alignment.
    cmp_mask = (t >= T_LO_CMP) & (t <= T_HI_CMP)
    lags, xc = _norm_xcorr(l1[cmp_mask], h_shift[cmp_mask], fs,
                           max_lag_s=0.008)
    k_peak = int(np.argmax(xc))
    rho_peak = float(xc[k_peak])
    lag_peak_ms = float(lags[k_peak] * 1e3)

    cL = detector_color(cfg, "L1")
    cH = detector_color(cfg, "H1")

    fig = plt.figure(figsize=(8.4, 5.9))
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.0], hspace=0.42,
                  left=0.085, right=0.975, top=0.880, bottom=0.085)
    ax_w = fig.add_subplot(gs[0, 0])
    ax_z = fig.add_subplot(gs[1, 0])

    # --- (a) wide context overlay ---
    # Vertical delimiters for the correlation window (lighter than a filled
    # block; reads cleaner behind the data).
    ax_w.axvspan(T_LO_CMP, T_HI_CMP, color="0.96", alpha=0.7, lw=0,
                 zorder=0)
    for x_b in (T_LO_CMP, T_HI_CMP):
        ax_w.axvline(x_b, color="0.55", ls="-", lw=0.7, alpha=0.8, zorder=1)
    ax_w.axvline(0.0, color="0.35", ls="--", lw=0.8, alpha=0.9, zorder=1)
    ax_w.plot(t, l1, color=cL, lw=1.0, label="L1 (reference)", zorder=2)
    ax_w.plot(t, h_shift, color=cH, lw=1.0, alpha=0.85,
              label=r"H1, corrected for $-6.9$ ms inter-site delay and "
                    r"detector sign",
              zorder=3)
    ax_w.set_xlim(T_LO_WIDE, T_HI_WIDE)
    ax_w.set_ylim(-7.5, 7.5)
    ax_w.set_xlabel("time from merger (s)", fontsize=8.5)
    ax_w.set_ylabel("normalized\nwhitened strain", fontsize=8.5)
    ax_w.set_title("(a) Wide context with correlation window",
                   fontsize=8.5, fontweight="normal", loc="left")
    ax_w.tick_params(labelsize=7.5)
    ax_w.legend(loc="lower left", fontsize=7.5, frameon=False, ncol=1,
                handlelength=1.4, handletextpad=0.5,
                labelspacing=0.35)
    for line in ax_w.get_legend().get_lines():
        line.set_linewidth(1.6)
    # Label the shaded correlation window inside the axes, just above the
    # band, in the white space at the top of panel (a).
    ax_w.text(0.5 * (T_LO_CMP + T_HI_CMP), 6.9,
              r"correlation window $[-20,\,+40]$ ms",
              ha="center", va="top", fontsize=7.5, color="0.30")

    # --- (b) merger-region detail ---
    ax_z.axvline(0.0, color="0.35", ls="--", lw=0.8, alpha=0.9)
    ax_z.plot(t, l1, color=cL, lw=1.3, label="L1")
    ax_z.plot(t, h_shift, color=cH, lw=1.3, alpha=0.9, label="H1")
    ax_z.set_xlim(T_LO_ZOOM, T_HI_ZOOM)
    win = (t >= T_LO_ZOOM) & (t <= T_HI_ZOOM)
    ymax = max(np.max(np.abs(l1[win])), np.max(np.abs(h_shift[win])))
    ax_z.set_ylim(-1.10 * ymax, 1.10 * ymax)
    ax_z.set_xlabel("time from merger (s)", fontsize=8.5)
    ax_z.set_ylabel("normalized\nwhitened strain", fontsize=8.5)
    ax_z.set_title("(b) Merger and post-merger oscillations", fontsize=8.5,
                   fontweight="normal", loc="left")
    ax_z.tick_params(labelsize=7.5)

    # Compact metric line above panel (b), right-aligned. Displayed traces
    # carry only the fixed -6.9 ms / sign-flip; rho_max and Delta t are the
    # cross-correlation peak over the same window shown in (b), now equal
    # to the shaded correlation window in (a).
    metric_txt = (r"$\rho_{\max}=%.2f$,  $\Delta t_{\rm peak}=%+0.2f$ ms"
                  r"  (over the correlation window shown in (a) and (b))"
                  % (rho_peak, lag_peak_ms))
    ax_z.text(1.0, 1.02, metric_txt,
              transform=ax_z.transAxes, ha="right", va="bottom",
              fontsize=7.5, color="0.15")

    fig.suptitle(
        "H1/L1 whitened-strain overlay after expected delay "
        "and sign correction",
        fontsize=10.0, y=0.965)

    save_figure(fig, paths.fig_main, "figure_05_dual_detector_overlay", cfg)


if __name__ == "__main__":
    main()

