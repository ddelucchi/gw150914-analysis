# GW150914 Reproducible Analysis & Figure Pipeline

This repository is a reproducible scientific-computing pipeline for exploring and visualizing GW150914 from public gravitational-wave data. It separates display-oriented conditioning from analysis-oriented processing, records data provenance, and provides one script per main or appendix figure.

The project is designed for inspection and reproducibility. It is not a replacement for the LIGO/Virgo Collaboration's parameter-estimation pipelines and does not claim that simplified fallback waveforms reproduce the full inference stack.

## Implemented surface

- GWOSC strain acquisition through GWpy
- public posterior-product download hooks
- SHA-256 manifests for downloaded data
- software-version manifests
- explicit visualization and analysis conditioning paths
- Welch ASD/PSD utilities
- explicit whitening
- high-pass, band-pass, and notch filtering
- Q-transform orchestration
- H1/L1 alignment and correlation utilities
- matched-filter SNR paths
- TaylorF2/SciPy matched-filter fallback plus provenance-labelled synthetic waveform overlays
- chirp-ridge fitting
- ringdown-scale analysis
- posterior plotting
- robustness sweeps
- centralized configuration for event, detector, conditioning, waveform, plotting, and seed parameters

## Scientific boundary

Two processing modes are intentionally separated.

**Visualization mode** is used for display-oriented whitening, filtering, and figure construction. It is not presented as the LVC statistical inference pipeline.

**Analysis mode** follows matched-filter conventions more closely and can use GWpy/PyCBC when those packages are available.

The lightweight fallback paths are diagnostic implementations. The matched-filter fallback uses a truncated TaylorF2-style frequency-domain inspiral model; visual overlays use an explicitly provenance-labelled synthetic inspiral + ringdown sketch. Neither is equivalent to full IMR parameter estimation.

One historical SNR figure is deliberately excluded from the default paper build because its detector-timing behavior did not agree with the intended reference path. Keeping a known-bad diagnostic out of the release build is part of the reproducibility contract.

## Repository layout

```text
configs/
  gw150914.yaml       central numerical/configuration contract
src/
  config.py           provenance and path management
  download_data.py    public-data acquisition and hashing
  preprocess.py       conditioning and spectral preparation
  spectral.py         PSD and Q-transform utilities
  matched_filter.py   PyCBC path plus documented SciPy fallback
  synth_waveform.py   lightweight waveform utilities
  ...                 fitting, correlation, plotting, and style modules
scripts/
  make_all_figures.py
  make_figure_*.py
tests/
  test_core_numerics.py
```

## Lightweight verification

The repository has a network-free core verification path. It does not download LIGO data and does not require PyCBC, GWpy, or PESummary.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
python -m compileall -q src scripts
```

On Windows, activate with `.venv\Scripts\activate`.

The core tests exercise configuration invariants, Welch ASD behavior, explicit whitening, fallback waveform support, leading-order chirp-mass scaling, matched-filter normalization, causal ringdown behavior, requested sample-rate preservation, and atomic-download cleanup semantics.

## Full analysis environment

Install the heavier gravitational-wave stack:

```bash
pip install -e ".[analysis,dev]"
```

Then build the configured figures:

```bash
python scripts/make_all_figures.py --mode paper
```

The script downloads public data unless `--skip-download` is supplied. Downloaded products are hashed into a manifest.

Robustness figures:

```bash
python scripts/make_all_figures.py --mode robustness
```

## Reproducibility

Numerical choices are centralized in [configs/gw150914.yaml](configs/gw150914.yaml), including:

- detector list and merger epoch
- sample rates
- visualization and analysis filters
- PSD segmentation
- Q-transform ranges
- waveform approximants and masses
- ringdown guide values
- plotting settings
- random/bootstrap seeds
- robustness grids

Downloaded data, intermediate products, and generated figures are excluded from Git. The pipeline records software versions and source-file hashes for fetched products instead.

## Validation status

The initial curated source release was checked for Python compilation and CLI construction. The public source has since received additional numerical and provenance hardening, including corrected leading-order chirp-mass scaling, complex matched-filter normalization, sample-rate-truthful strain caching, atomic downloads, and runtime waveform provenance labels.

The current network-free test suite is the executable contract for those local invariants. GitHub Actions is configured for it, but the account currently reports workflow startup failures before job creation; until runner execution is restored, use the clone-local commands above rather than inferring a green hosted-CI state.

Passing the core suite does not independently reproduce the LVC discovery or parameter-estimation analysis.

## Release audit

See [CODEX_AUDIT.md](CODEX_AUDIT.md) for the curated source boundary and observed verification state.

## License

Source is publicly viewable for portfolio, technical evaluation, and reproducibility review. See [LICENSE](LICENSE).
