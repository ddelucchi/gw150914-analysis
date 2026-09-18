# GW150914 reproducible figure pipeline

A from-scratch reproduction of the publication-grade figure set for the
first LIGO binary-black-hole detection (GW150914): a chirp signal sweeping
**~35 → 250 Hz**, peak strain **~10⁻²¹**, network matched-filter **SNR ≈ 24**,
**6.9 ms** Hanford/Livingston delay, source-frame component masses
**~36 / 29 M☉**, final mass **~62 M☉**, **~3 M☉c²** radiated.

The pipeline strictly separates **visualization-only conditioning** from
**analysis-grade conditioning** — the narrow [35, 350] Hz bandpass is for
display and is *not* the LVC statistical pipeline.

## Quick start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate gw150914

# 2. (Or) install via pyproject
pip install -e .

# 3. Build everything end-to-end (downloads ~few hundred MB of data once)
python scripts/make_all_figures.py            # paper mode (default)
python scripts/make_all_figures.py --mode robustness
python scripts/make_all_figures.py --mode all
```

A first run downloads:
- H1/L1 strain (16384 Hz preferred, fallback 4096 Hz) over GPS 1126259446–1126259478, via `gwpy.timeseries.TimeSeries.fetch_open_data`
- The official **GWTC-1 PE samples** `GW150914_GWTC-1.hdf5` (supersedes older O1 samples)
- The **PESummary public bundle** containing `GW150914_bilby_pesummary.dat`

Provenance (URLs, byte counts, SHA-256 hashes, software versions) is written to
`data/manifests/`.

## Repository layout

```
configs/gw150914.yaml         # all numerics live here — no script hard-codes values
data/
  raw/                        # cached strain HDF5 from GWOSC
  intermediate/
  posterior/                  # GWTC-1 + PESummary
  manifests/                  # data_manifest.json, software_versions.json
src/
  config.py                   # config + path + version helpers
  download_data.py            # fetch strain & posterior products
  preprocess.py               # viz-mode + analysis-mode conditioning, explicit whitening
  spectral.py                 # PSDs, Q-transform
  waveform_overlay.py         # PESummary max-L IMRPhenomPv2 waveform
  matched_filter.py           # PyCBC matched filter (GWpy SNR-tutorial style)
  ridge_fit.py                # zero-crossing chirp-mass fit
  correlation.py              # H1/L1 lag correlation
  ringdown.py                 # damped-sinusoid guide curve
  posterior_plots.py          # GWTC-1 HDF5 loader + corner helpers
  style.py                    # consistent typography / colors / save helpers
scripts/
  _common.py                  # shared boilerplate
  make_all_figures.py         # end-to-end build
  make_figure_01..11_*.py     # one polished figure per script
  make_figure_appendix_robustness.py
figures/main/                 # final paper figures (.pdf + .png)
figures/appendix/             # robustness sweep
```

## Two distinct conditioning modes

### Visualization mode  (`src.preprocess.viz_condition`)
Tukey window → explicit whitening (FFT / ASD / iFFT, unit-variance) →
zero-phase 8th-order Butterworth bandpass **[35, 350] Hz** → narrow notches
(60/120/180/300 Hz). Reproduces the discovery-paper Figure 1 / DCC noise-guide
Figure 2 conditioning sequence. **For display only.**

### Analysis mode  (matched-filter family)
Highpass at **15 Hz**, Welch PSD on 4 s / 2 s overlap, low-frequency cutoff
**15 Hz**, IMRPhenomD template — the GWpy SNR tutorial recipe.

Both modes share the same `configs/gw150914.yaml`; nothing is hard-coded.

## Figure index

| # | Script | Scientific content |
|---|--------|-------------------|
| 1 | `make_figure_01_event_atlas.py` | 4×2 PRL-style atlas (viz strain, aligned overlay, residuals, Q-transform) |
| 2 | `make_figure_02_processing_pipeline.py` | DCC sequence: raw → Tukey → whitened → +bandpass |
| 3 | `make_figure_03_psd_windowing.py` | Spectral leakage demo + Welch reference |
| 4 | `make_figure_04_qtransform.py` | H1/L1 Q-transform spectrograms |
| 5 | `make_figure_05_dual_detector_overlay.py` | 6.9 ms shift + sign-flip overlay with zoom inset |
| 6 | `make_figure_06_chirp_mass_fit.py` | Zero-crossing pedagogical chirp-mass fit |
| 7 | `make_figure_07_maxl_overlay.py` | PESummary max-L IMRPhenomPv2 overlay |
| 8 | `make_figure_08_snr_timeseries.py` | Single-detector PyCBC matched filter (tutorial + 36/29 M☉) |
| 9 | `make_figure_09_correlation_residuals.py` | H1↔L1 lag correlation, data vs max-L residual |
| 10 | `make_figure_10_ringdown_inset.py` | Late-time inset, ~250 Hz / 4 ms damped guide |
| 11 | `make_figure_11_posterior_corner.py` | Corner plot from `GW150914_GWTC-1.hdf5` |
| A | `make_figure_appendix_robustness.py` | Window / PSD / bandpass parameter sweep |

## Validation checklist (the build is "done" only when these hold)

1. Q-transform shows a chirp from ~35 Hz to >200 Hz over ~0.15–0.2 s.
2. Dual-detector overlay agrees after 6.9 ms shift + sign flip.
3. Direct-data chirp-mass falls within ~30–40 M☉ (caption distinguishes from posterior).
4. Tutorial matched-filter SNR exceeds ~17 at the event time.
5. Max-L residual correlation has no compelling peak at ±7 ms.
6. Ringdown inset visually consistent with ~250 Hz, few-ms damping.

## Scientific cautions hard-coded into captions

- Visualization-mode plots are **not** the collaboration's statistical analysis.
- Residuals against an **illustrative** waveform are **not** residuals against the **max-L** waveform.
- The zero-crossing chirp-mass is a **direct-data pedagogical estimate**, not the LVC posterior.
- The single-detector matched-filter SNR is **not** the LVC network SNR 24.
- Use **GWTC-1** posterior samples; do not use older superseded O1 samples.

## Reproducibility

Random-seeded NumPy throughout (`configs/gw150914.yaml :: seeds`). Software
versions auto-recorded to `data/manifests/software_versions.json`. Every
downloaded file is hashed (SHA-256) into `data/manifests/data_manifest.json`.

## Modes

- `paper` — final polished figures only (default)
- `robustness` — appendix parameter-sweep diagnostics
- `all` — both

