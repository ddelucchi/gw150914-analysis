"""Figure 3 — PSD method figure (2x2 layout, log-binned display).

Layout (per critic redesign):

  +-------------------+-------------------+
  |  (a) tapering     |  (b) Welch ref.   |     two square upper panels
  +-------------------+-------------------+
  |  (c)  log-ratio distortion (full-w)   |     tall, full-width panel
  +---------------------------------------+

Display strategy:
  - Raw bin-level ASDs are kept faintly in the background for honesty.
  - Foreground curves are log-frequency-binned medians (~30 bins/decade)
    so the short-segment estimate stops reading as a "blue carpet"
    glued to the bottom of the axis.
  - Panel (c) plots log10(ASD_short / ASD_Welch) on a LINEAR y-axis
    centered on zero.  Then "agreement with Welch" is at y=0 in the
    middle of the frame, and upward/downward distortions are visually
    symmetric."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from _common import setup, load_event_strain, long_welch_asd
from src.spectral import psd_no_window, psd_tukey
from src.style import save_figure


# Frequency bands.
F_LO_AB, F_HI_AB = 8.0, 2048.0
F_LO_C, F_HI_C = 30.0, 2048.0

# ASD y-range for panels (a)/(b).
ASD_LO, ASD_HI = 2e-24, 7e-20

# Panel (c): symmetric log10-ratio range centered on zero.
LOG_RATIO_LO, LOG_RATIO_HI = -0.8, +1.6

C_NOWIN = "0.32"   # darker grey: clearly secondary, fully legible in print
C_TUKEY = "C0"
C_WELCH = "C3"
C_GUIDE = "0.35"

BINS_PER_DECADE = 30
BINS_PER_DECADE_RATIO = 18  # coarser binning for (c) stabilises the trace
RAW_ALPHA = 0.07   # very faint raw spectral background in (a)/(b)


def _log_bin(f: np.ndarray, y: np.ndarray,
             f_lo: float, f_hi: float,
             bins_per_decade: int = BINS_PER_DECADE
             ) -> tuple[np.ndarray, np.ndarray]:
    """Median of `y` in log-frequency bins."""
    pos = (f > 0) & np.isfinite(y)
    f = f[pos]; y = y[pos]
    n_dec = np.log10(f_hi) - np.log10(f_lo)
    n_bin = max(int(round(n_dec * bins_per_decade)), 8)
    edges = np.logspace(np.log10(f_lo), np.log10(f_hi), n_bin + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    medians = np.full(n_bin, np.nan)
    idx = np.searchsorted(edges, f) - 1
    for k in range(n_bin):
        sel = idx == k
        if sel.any():
            medians[k] = np.median(y[sel])
    keep = np.isfinite(medians)
    return centers[keep], medians[keep]


def _ratio(f_short: np.ndarray, asd_short: np.ndarray,
           f_ref: np.ndarray, asd_ref: np.ndarray
           ) -> tuple[np.ndarray, np.ndarray]:
    """Resample reference ASD onto short-segment grid and return their ratio."""
    pos = f_short > 0
    f = f_short[pos]
    asd_ref_on_f = np.interp(f, f_ref, asd_ref, left=np.nan, right=np.nan)
    return f, asd_short[pos] / asd_ref_on_f


def main() -> None:
    cfg, paths = setup()
    pp = cfg["processing_pipeline"]
    ts = load_event_strain(cfg, paths, "H1")
    fs = float(ts.sample_rate.value)
    t_c = pp["segment_center_gps"]
    seg = ts.crop(t_c - pp["segment_length_s"] / 2,
                  t_c + pp["segment_length_s"] / 2).value

    f_nw, p_nw = psd_no_window(seg, fs)
    f_tk, p_tk = psd_tukey(seg, fs, alpha=cfg["viz"]["tukey_alpha"])
    f_w, asd_w = long_welch_asd(ts, cfg)
    asd_nw = np.sqrt(p_nw)
    asd_tk = np.sqrt(p_tk)

    # Log-binned foreground curves for (a)/(b).
    f_nw_b, asd_nw_b = _log_bin(f_nw, asd_nw, F_LO_AB, F_HI_AB)
    f_tk_b, asd_tk_b = _log_bin(f_tk, asd_tk, F_LO_AB, F_HI_AB)
    f_w_b,  asd_w_b  = _log_bin(f_w,  asd_w,  F_LO_AB, F_HI_AB)

    # 1/f^2 leakage guide anchored on the binned Welch curve at 50-100 Hz.
    band = (f_w_b > 50) & (f_w_b < 100)
    f_anchor = f_w_b[band][0]
    a_anchor = asd_w_b[band][0]
    guide = a_anchor * (f_anchor / f_w_b) ** 2

    # Ratios for (c).  Coarser log-binning here stabilises the displayed
    # ratio against the bin-level fluctuations of the short-segment ASD.
    f_r_nw, r_nw = _ratio(f_nw, asd_nw, f_w, asd_w)
    f_r_tk, r_tk = _ratio(f_tk, asd_tk, f_w, asd_w)
    f_r_nw_b, r_nw_b = _log_bin(f_r_nw, r_nw, F_LO_C, F_HI_C,
                                bins_per_decade=BINS_PER_DECADE_RATIO)
    f_r_tk_b, r_tk_b = _log_bin(f_r_tk, r_tk, F_LO_C, F_HI_C,
                                bins_per_decade=BINS_PER_DECADE_RATIO)

    # ----------------------------- layout --------------------------------
    fig = plt.figure(figsize=(8.6, 7.2))
    gs = GridSpec(2, 2, figure=fig,
                  height_ratios=[1.0, 0.95],
                  hspace=0.34, wspace=0.06,
                  left=0.085, right=0.975, top=0.955, bottom=0.075)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[1, :])

    # ----------------------------- (a) -----------------------------------
    ax_a.loglog(f_nw, asd_nw, color=C_NOWIN, lw=0.35, alpha=RAW_ALPHA)
    ax_a.loglog(f_tk, asd_tk, color=C_TUKEY, lw=0.35, alpha=RAW_ALPHA)
    ax_a.loglog(f_nw_b, asd_nw_b, color=C_NOWIN, lw=1.2, alpha=0.95,
                label="4 s, no window")
    ax_a.loglog(f_tk_b, asd_tk_b, color=C_TUKEY, lw=1.4,
                label=fr"4 s, Tukey ($\alpha={cfg['viz']['tukey_alpha']}$)")
    ax_a.loglog(f_w_b, guide, color=C_GUIDE, ls=":", lw=1.0, alpha=1.0)
    # f^-2 label tucked tightly under the dotted guide.
    g_idx = np.searchsorted(f_w_b, 11.0)
    ax_a.text(f_w_b[g_idx], guide[g_idx] * 0.55,
              r"$f^{-2}$",
              fontsize=8, color="0.2", ha="left", va="top")
    ax_a.set_title("(a) Leakage suppression by tapering",
                   fontsize=8.5, fontweight="normal")
    ax_a.set_xlabel("frequency (Hz)", fontsize=8.5)
    ax_a.set_ylabel(r"ASD (strain $/\sqrt{\rm Hz}$)", fontsize=8.5)
    ax_a.legend(loc="upper right", fontsize=7,
                frameon=False, borderpad=0.2, handlelength=1.4,
                labelspacing=0.3)

    # ----------------------------- (b) -----------------------------------
    ax_b.loglog(f_tk, asd_tk, color=C_TUKEY, lw=0.35, alpha=RAW_ALPHA)
    ax_b.loglog(f_w,  asd_w,  color=C_WELCH, lw=0.35, alpha=RAW_ALPHA + 0.05)
    ax_b.loglog(f_tk_b, asd_tk_b, color=C_TUKEY, lw=1.2, alpha=0.9,
                label=fr"4 s, Tukey ($\alpha={cfg['viz']['tukey_alpha']}$)")
    welch_total = int(cfg["processing_pipeline"]["welch_total_s"])
    ax_b.loglog(f_w_b, asd_w_b, color=C_WELCH, lw=1.7,
                label=f"{welch_total} s Welch ASD")
    ax_b.set_title("(b) Short-segment ASD vs. Welch reference",
                   fontsize=8.5, fontweight="normal")
    ax_b.set_xlabel("frequency (Hz)", fontsize=8.5)
    ax_b.legend(loc="upper right", fontsize=7,
                frameon=False, borderpad=0.2, handlelength=1.4,
                labelspacing=0.3)

    for ax in (ax_a, ax_b):
        ax.set_xlim(F_LO_AB, F_HI_AB)
        ax.set_ylim(ASD_LO, ASD_HI)
        ax.grid(True, which="both", alpha=0.22, lw=0.4)
        ax.tick_params(axis="both", which="major", labelsize=7.5)
    ax_b.tick_params(axis="y", labelleft=False)

    # ----------------------------- (c) -----------------------------------
    # Bold log-binned curves only — no raw chatter on the lower edge.
    log_r_nw_b = np.log10(r_nw_b)
    log_r_tk_b = np.log10(r_tk_b)

    ax_c.axhline(0.0, color=C_GUIDE, ls="-", lw=0.9, alpha=0.85)
    # Faint agreement band: ± factor of 1.25 around unity.
    ax_c.axhspan(np.log10(1 / 1.25), np.log10(1.25),
                 color=C_TUKEY, alpha=0.06, lw=0)
    # Faint factor-of-2 reference lines, no floating annotations.
    for y_val in (+0.301, -0.301):
        ax_c.axhline(y_val, color=C_GUIDE, ls=":", lw=0.5, alpha=0.4)

    ax_c.semilogx(f_r_nw_b, log_r_nw_b, color=C_NOWIN, lw=1.5, alpha=0.95,
                  label="no-window / Welch")
    ax_c.semilogx(f_r_tk_b, log_r_tk_b, color=C_TUKEY, lw=1.8,
                  label="Tukey / Welch")

    ax_c.set_xlim(F_LO_C, F_HI_C)
    ax_c.set_ylim(LOG_RATIO_LO, LOG_RATIO_HI)
    # Custom y-ticks: numeric factors, axis label carries the meaning.
    yticks = [-0.301, 0.0, 0.301, 1.0]
    yticklabels = ["0.5", "1", "2", "10"]
    ax_c.set_yticks(yticks)
    ax_c.set_yticklabels(yticklabels)
    ax_c.set_title("(c) Ratio to Welch reference",
                   fontsize=8.5, fontweight="normal")
    ax_c.set_xlabel("frequency (Hz)", fontsize=8.5)
    ax_c.set_ylabel(r"ASD$_{4\,\rm s}\ /\ $ASD$_{\rm Welch}$", fontsize=8.5)
    ax_c.legend(loc="upper right", fontsize=7,
                frameon=False, borderpad=0.2, handlelength=1.4,
                labelspacing=0.3)
    ax_c.grid(True, which="both", axis="x", alpha=0.22, lw=0.4)
    ax_c.tick_params(axis="both", which="major", labelsize=7.5)

    save_figure(fig, paths.fig_main, "figure_03_psd_windowing", cfg)


if __name__ == "__main__":
    main()

