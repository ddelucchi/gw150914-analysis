"""Figure 4 — Q-transform spectrograms (H1, L1)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import setup, load_event_strain
from src.spectral import qtransform
from src.style import save_figure


def main() -> None:
    cfg, paths = setup()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    t_c = cfg["event"]["gps_merger"]
    # Per-detector percentile ceiling: caps the very brightest pixels
    # (which would otherwise saturate the H1 chirp into a yellow plateau)
    # without darkening the rest of the field.  99.5th-percentile by
    # default; matches the cleaner visual of the L1 panel.
    per_pct = 99.5
    for ax, ifo in zip(axes, ("H1", "L1")):
        ts = load_event_strain(cfg, paths, ifo)
        q = qtransform(ts, cfg)
        z = np.asarray(q.value)
        vmax = float(np.percentile(z, per_pct))
        im = ax.imshow(
            z.T, aspect="auto", origin="lower",
            extent=[q.xspan[0] - t_c, q.xspan[1] - t_c,
                    q.yspan[0], q.yspan[1]],
            cmap=cfg["style"]["cmap_tf"],
            vmin=cfg["qtransform"]["vmin"],
            vmax=vmax,
        )
        ax.set_yscale("log")
        ax.set_ylim(*cfg["qtransform"]["frange_hz"])
        ax.set_xlabel("time from merger (s)")
        ax.set_title(f"{ifo}")
    axes[0].set_ylabel("frequency (Hz)")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("normalized energy")
    fig.suptitle("GW150914 Q-transform (chirp 35 → 250 Hz)")
    save_figure(fig, paths.fig_main, "figure_04_qtransform", cfg)


if __name__ == "__main__":
    main()

