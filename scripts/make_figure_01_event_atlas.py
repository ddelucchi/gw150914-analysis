"""Figure 1 — GW150914 event atlas (PRL-inspired 4 x 2 panel).

Rows:
    0. Whitened + bandpassed strain, RMS-normalised (so H1/L1 are comparable).
    1. Detector-aligned overlay (H1 shifted by -6.9 ms, sign-flipped) on L1.
    2. Whitened strain with a maximum-likelihood IMR template overlaid in
       green.  Shows the inspiral-merger-ringdown morphology that GR predicts.
    3. Q-transform spectrogram with the famous chirp arc 35 -> 250 Hz.

Caveat (per LIGO data guide): this presentation is for *visual comparison
with GR*, not the LVC statistical detection pipeline.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import setup, load_event_strain, long_welch_asd
from src.preprocess import viz_condition, extract_segment, apply_bandpass
from src.spectral import qtransform
from src.style import detector_color, save_figure
from src.waveform_overlay import maximum_likelihood_waveform


def _rms_in_band(t: np.ndarray, x: np.ndarray, lo: float, hi: float) -> float:
    m = (t >= lo) & (t <= hi)
    return float(np.std(x[m])) if np.any(m) else float(np.std(x))


def main() -> None:
    cfg, paths = setup()
    t_c = cfg["event"]["gps_merger"]
    half = 1.0
    delay = cfg["event"]["hl_delay_ms"] * 1e-3
    sample_path = paths.posterior / cfg["posterior"]["pesummary_inner_dat"]

    fig, axes = plt.subplots(4, 2, figsize=(11.5, 12.5))

    panels: dict[str, np.ndarray] = {}
    times: dict[str, np.ndarray] = {}
    fs_global = 16384.0

    # ---------- Row 0: whitened, RMS-normalised so H1 vs L1 are visually comparable
    for j, ifo in enumerate(("H1", "L1")):
        ts = load_event_strain(cfg, paths, ifo)
        fs = float(ts.sample_rate.value)
        fs_global = fs
        f_asd, asd = long_welch_asd(ts, cfg)
        seg = extract_segment(ts, t_c, half).value
        t = np.arange(len(seg)) / fs - half
        cond = viz_condition(seg, fs, cfg, f_asd, asd)
        rms = _rms_in_band(t, cond, -0.5, 0.2)
        cond = cond / rms
        panels[ifo] = cond
        times[ifo] = t

        axes[0, j].plot(t, cond, color=detector_color(cfg, ifo), lw=0.9)
        axes[0, j].set_xlim(*cfg["viz"]["zoom_window_s"])
        axes[0, j].set_ylim(-5, 5)
        axes[0, j].axvline(0, color="0.4", ls=":", lw=0.7)
        axes[0, j].set_title(f"{ifo} whitened + [35,350] Hz (RMS-norm.)")
        axes[0, j].set_ylabel(r"strain ($\sigma$ units)")

    # ---------- Row 1: detector-aligned overlay
    h_shift = -np.interp(times["L1"], times["H1"] - delay, panels["H1"])
    for j, ifo_focus in enumerate(("L1", "H1")):
        ax = axes[1, j]
        if ifo_focus == "L1":
            ax.plot(times["L1"], panels["L1"], color=detector_color(cfg, "L1"),
                    lw=1.1, label="L1", zorder=3)
            ax.plot(times["L1"], h_shift, color=detector_color(cfg, "H1"),
                    lw=0.9, alpha=0.55,
                    label=f"H1 (shifted -{delay*1e3:.1f} ms, sign-flipped)",
                    zorder=2)
        else:
            ax.plot(times["L1"], h_shift, color=detector_color(cfg, "H1"),
                    lw=1.1,
                    label=f"H1 (shifted -{delay*1e3:.1f} ms, sign-flipped)",
                    zorder=3)
            ax.plot(times["L1"], panels["L1"], color=detector_color(cfg, "L1"),
                    lw=0.9, alpha=0.55, label="L1", zorder=2)
        ax.set_xlim(*cfg["viz"]["zoom_window_s"])
        ax.set_ylim(-5, 5)
        ax.axvline(0, color="0.4", ls=":", lw=0.7)
        ax.set_title("Detector-aligned overlay")
        ax.set_ylabel(r"strain ($\sigma$ units)")
        ax.legend(loc="upper left", fontsize=8)

    # ---------- Row 2: whitened strain with best-fit IMR overlay
    try:
        for j, ifo in enumerate(("H1", "L1")):
            tw, hw = maximum_likelihood_waveform(sample_path, cfg, t_c, ifo=ifo)
            tw_rel = tw - t_c
            hw_b = apply_bandpass(hw, fs_global,
                                  cfg["viz"]["bandpass_low_hz"],
                                  cfg["viz"]["bandpass_high_hz"],
                                  cfg["viz"]["butter_order"])
            data = panels[ifo]
            tdata = times[ifo]
            # Peak-match in a tight merger window
            mw = (tw_rel >= -0.05) & (tw_rel <= 0.02)
            md = (tdata >= -0.05) & (tdata <= 0.02)
            if np.any(mw) and np.any(md):
                p_tmpl = float(np.max(np.abs(hw_b[mw])))
                p_data = float(np.max(np.abs(data[md])))
                if p_tmpl > 0:
                    hw_b = hw_b * (p_data / p_tmpl)
            ax = axes[2, j]
            sign = -1 if ifo == "H1" else 1
            ax.plot(tdata, sign * data, color=detector_color(cfg, ifo),
                    lw=0.9, alpha=0.85,
                    label=f"{ifo} data" + (" (sign-flipped)" if sign == -1 else ""))
            ax.plot(tw_rel, hw_b,
                    color=cfg["style"]["template_color"], lw=1.4,
                    label="best-fit IMR sketch")
            ax.set_xlim(*cfg["viz"]["zoom_window_s"])
            ax.set_ylim(-5, 5)
            ax.axvline(0, color="0.4", ls=":", lw=0.7)
            ax.set_title(f"{ifo} data vs best-fit IMR (PN+RD sketch)")
            ax.set_ylabel(r"strain ($\sigma$ units)")
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    except Exception as exc:
        for j in range(2):
            axes[2, j].text(0.5, 0.5, f"best-fit overlay unavailable:\n{exc}",
                            ha="center", va="center")

    # ---------- Row 3: Q-transform per detector
    for j, ifo in enumerate(("H1", "L1")):
        ts = load_event_strain(cfg, paths, ifo)
        try:
            q = qtransform(ts, cfg)
            ax = axes[3, j]
            im = ax.imshow(
                q.value.T, aspect="auto", origin="lower",
                extent=[q.xspan[0] - t_c, q.xspan[1] - t_c,
                        q.yspan[0], q.yspan[1]],
                cmap=cfg["style"]["cmap_tf"],
                vmin=cfg["qtransform"]["vmin"],
                vmax=cfg["qtransform"]["vmax"],
            )
            ax.set_yscale("log")
            ax.set_ylim(*cfg["qtransform"]["frange_hz"])
            ax.set_xlim(*cfg["viz"]["zoom_window_s"])
            ax.axvline(0, color="w", ls=":", lw=0.7, alpha=0.7)
            ax.set_title(f"{ifo} Q-transform")
            ax.set_xlabel("time from merger (s)")
            ax.set_ylabel("frequency (Hz)")
            plt.colorbar(im, ax=ax, label="normalized energy",
                         shrink=0.85, pad=0.02)
        except Exception as exc:
            axes[3, j].text(0.5, 0.5, f"Q-transform failed:\n{exc}",
                            ha="center", va="center")

    fig.suptitle("GW150914 — event atlas (visualization mode)", y=0.995,
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save_figure(fig, paths.fig_appendix, "figure_appendix_event_atlas", cfg)


if __name__ == "__main__":
    main()

