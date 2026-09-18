"""Figure 7 — Best-fit IMR overlay.

Two stacked panels (H1 sign-flipped, L1 native), both whitened
identically and shown in σ-units in the discovery-paper window.  The
overlaid green curve is the **GWTC-1 best-fit IMR sketch** built from
the maximum-likelihood detector-frame masses :math:`(m_1, m_2)` read
out of ``GW150914_GWTC-1.hdf5`` and combined as

* Newtonian (leading-order PN) inspiral up to :math:`f_{\rm ISCO}`,
* a half-Hann ramp at merger,
* a damped sinusoid ringdown with :math:`f_{\rm RD}\!\approx\!250` Hz,
  :math:`\tau\!\approx\!4` ms,

and bandpass-filtered with the same ``[35, 350]`` Hz Butterworth as
the data.  This is **not** the LVC IMRPhenomPv2 maximum-likelihood
waveform — that requires the PESummary release which is not available
on this build — but it carries the same dominant features (chirp
slope, peak amplitude, ringdown timescale) and is amplitude-matched
to the data over the merger window.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import setup, load_event_strain, long_welch_asd
from src.preprocess import viz_condition, extract_segment, apply_bandpass
from src.style import detector_color, save_figure
from src.waveform_overlay import build_waveform_product


def main() -> None:
    cfg, paths = setup()
    t_c = cfg["event"]["gps_merger"]
    sample_path = paths.posterior / cfg["posterior"]["pesummary_inner_dat"]
    have_pesummary_dat = sample_path.exists()
    if not have_pesummary_dat:
        print("[info] PESummary .dat absent; using GWTC-1 best-fit + "
              "Newtonian-inspiral + ringdown sketch.")

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True, sharey=True)
    for ax, ifo in zip(axes, ("H1", "L1")):
        ts = load_event_strain(cfg, paths, ifo)
        fs = float(ts.sample_rate.value)
        f_asd, asd = long_welch_asd(ts, cfg)
        seg = extract_segment(ts, t_c, 0.3).value
        t = np.arange(len(seg)) / fs - 0.3
        cond = viz_condition(seg, fs, cfg, f_asd, asd)
        rms = float(np.std(cond[(t >= -0.5) & (t <= 0.2)]))
        if rms > 0:
            cond = cond / rms

        try:
            tw, hw = maximum_likelihood_waveform(sample_path, cfg, t_c, ifo=ifo)
            tw_rel = tw - t_c
        except Exception as exc:
            tw, hw, tw_rel = None, None, None
            print(f"[warn] best-fit waveform unavailable for {ifo}: {exc}")

        if hw is not None and tw_rel is not None:
            hw_b = apply_bandpass(hw, fs,
                                  cfg["maxl"]["bandpass_hz"][0],
                                  cfg["maxl"]["bandpass_hz"][1],
                                  cfg["viz"]["butter_order"])
            sign = -1 if ifo == "H1" else 1
            # Peak-match around the merger
            mw = (tw_rel >= -0.05) & (tw_rel <= 0.02)
            md = (t >= -0.05) & (t <= 0.02)
            if np.any(mw) and np.any(md):
                p_tmpl = float(np.max(np.abs(hw_b[mw])))
                p_data = float(np.max(np.abs(cond[md])))
                if p_tmpl > 0:
                    hw_b = hw_b * (p_data / p_tmpl)
            ax.plot(t, sign * cond, color=detector_color(cfg, ifo), lw=0.9,
                    label=f"{ifo} data" + (" (sign-flipped)" if sign == -1 else ""))
            ax.plot(tw_rel, hw_b,
                    color=cfg["style"]["template_color"], lw=1.5,
                    label=("max-L IMRPhenomPv2 (PESummary)"
                           if have_pesummary_dat
                           else "GWTC-1 best-fit IMR sketch (PN + RD)"))
        else:
            ax.plot(t, cond, color=detector_color(cfg, ifo), lw=0.9,
                    label=f"{ifo} data")
        ax.axvline(0, color="0.4", ls=":", lw=0.7)
        ax.set_xlim(*cfg["maxl"]["window_s"])
        ax.set_ylabel(r"whitened strain ($\sigma$)")
        ax.set_title(ifo)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    axes[0].set_ylim(-3.5, 3.5)
    axes[-1].set_xlabel("time from merger (s)")
    title = ("Max-likelihood IMRPhenomPv2 overlay (PESummary)"
             if have_pesummary_dat
             else "PN + ringdown cartoon overlay — GWTC-1 max-L masses\n"
                  "Newtonian inspiral + Hann ramp + 250 Hz / 4 ms damped ringdown, "
                  "amplitude-matched at merger.  "
                  "Pedagogical timescale guide only — *not* the LVC IMRPhenomPv2 max-L waveform.")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save_figure(fig, paths.fig_appendix, "figure_appendix_pn_rd_cartoon", cfg)


if __name__ == "__main__":
    main()

