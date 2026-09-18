"""Figure 9 — H1 ↔ L1 lag correlation.

Three panels:

* **Left** — *raw* whitened H1 vs L1 (no alignment, no sign flip).
  The cross-correlation peaks near :math:`+\Delta t_{\rm HL}\approx +7`
  ms — the inter-site light-travel time — and is *anti-correlated*
  because the detectors have opposite arm orientations for this sky
  position.  Reference taken from the LIGO detector-noise guide.
* **Middle** — *aligned* (H1 shifted by :math:`-\Delta t_{\rm HL}` and
  sign-flipped).  The peak now sits at :math:`0` ms with positive sign,
  confirming a common signal in both detectors.
* **Right** — same as middle, but on residuals after subtracting the
  GWTC-1 best-fit IMR sketch.  A correctly-removed signal leaves *no*
  notable peak at zero lag.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import setup, load_event_strain, long_welch_asd
from src.preprocess import viz_condition, extract_segment, apply_bandpass
from src.correlation import lag_correlation
from src.waveform_overlay import build_waveform_product
from src.style import save_figure


def _conditioned(cfg, paths, ifo, half_s):
    t_c = cfg["event"]["gps_merger"]
    ts = load_event_strain(cfg, paths, ifo)
    fs = float(ts.sample_rate.value)
    f_asd, asd = long_welch_asd(ts, cfg)
    seg = extract_segment(ts, t_c, half_s).value
    t = np.arange(len(seg)) / fs - half_s
    cond = viz_condition(seg, fs, cfg, f_asd, asd)
    return t, cond, fs


def _align_h_to_l(t_l, t_h, h, delay_s):
    return -np.interp(t_l, t_h - delay_s, h)


def main() -> None:
    cfg, paths = setup()
    delay_ms = float(cfg["event"]["hl_delay_ms"])
    delay_s = delay_ms * 1e-3
    windows = [(-0.20, +0.05), (-0.10, +0.05), (-0.05, +0.02)]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    sample_path = paths.posterior / cfg["posterior"]["pesummary_inner_dat"]

    # Build conditioned strands once
    t_h, h, fs = _conditioned(cfg, paths, "H1", half_s=0.5)
    t_l, l, _  = _conditioned(cfg, paths, "L1", half_s=0.5)
    h_aligned = _align_h_to_l(t_l, t_h, h, delay_s)

    # ---------- Pane A: raw, no alignment ----------
    h_raw_on_l = np.interp(t_l, t_h, h)
    for w in windows:
        m = (t_l >= w[0]) & (t_l <= w[1])
        lags, c = lag_correlation(h_raw_on_l[m], l[m], fs)
        axes[0].plot(lags * 1e3, c, lw=1.0, label=f"window {w}")
    axes[0].axvline(+delay_ms, color="k", ls="--", lw=0.7,
                    label=fr"+{delay_ms:.1f} ms (expected $\Delta t_{{HL}}$)")
    axes[0].axvline(0, color="0.6", ls=":", lw=0.6)
    axes[0].set_xlim(-25, 25)
    axes[0].set_xlabel("lag (ms)")
    axes[0].set_ylabel("normalized cross-correlation")
    axes[0].set_title("Raw — H1 vs L1\n(no alignment, no sign flip)")
    axes[0].legend(fontsize=8, loc="lower left")

    # ---------- Pane B: aligned data ----------
    for w in windows:
        m = (t_l >= w[0]) & (t_l <= w[1])
        lags, c = lag_correlation(h_aligned[m], l[m], fs)
        axes[1].plot(lags * 1e3, c, lw=1.0, label=f"window {w}")
    axes[1].axvline(0, color="k", ls="--", lw=0.7,
                    label="0 ms (expected after alignment)")
    axes[1].set_xlim(-25, 25)
    axes[1].set_xlabel("lag (ms)")
    axes[1].set_title("Aligned — H1 shifted by $-\\Delta t_{HL}$ and sign-flipped\n"
                      "(common signal → peak at 0 ms)")
    axes[1].legend(fontsize=8, loc="lower left")

    # ---------- Pane C: aligned residuals ----------
    try:
        t_c = cfg["event"]["gps_merger"]
        tw, hw = maximum_likelihood_waveform(sample_path, cfg, t_c, ifo="L1")
        hw_b = apply_bandpass(hw, fs,
                              cfg["viz"]["bandpass_low_hz"],
                              cfg["viz"]["bandpass_high_hz"],
                              cfg["viz"]["butter_order"])
        model_on_l = np.interp(t_l, tw - t_c, hw_b)
        # amplitude-match in inspiral window
        mw = (t_l >= -0.18) & (t_l <= 0.02)
        if np.any(mw) and np.std(model_on_l[mw]) > 0:
            model_on_l = model_on_l * (np.std(l[mw]) / np.std(model_on_l[mw]))
        res_l = l - model_on_l
        res_h = h_aligned - model_on_l
        for w in windows:
            m = (t_l >= w[0]) & (t_l <= w[1])
            lags, c = lag_correlation(res_h[m], res_l[m], fs)
            axes[2].plot(lags * 1e3, c, lw=1.0, label=f"window {w}")
        axes[2].axvline(0, color="k", ls="--", lw=0.7,
                        label="0 ms")
        axes[2].set_xlim(-25, 25)
        axes[2].set_title("Aligned residuals — best-fit IMR subtracted\n"
                          "(residual peak at 0 ms ⇒ sketch ≠ true max-L)")
        axes[2].set_xlabel("lag (ms)")
        axes[2].legend(fontsize=8, loc="lower left")
    except Exception as exc:
        axes[2].text(0.5, 0.5, f"residual correlation unavailable:\n{exc}",
                     ha="center", va="center", transform=axes[2].transAxes)

    for ax in axes:
        ax.axhline(0, color="0.85", lw=0.5)
        ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    save_figure(fig, paths.fig_appendix, "figure_appendix_correlation_residuals", cfg)


if __name__ == "__main__":
    main()

