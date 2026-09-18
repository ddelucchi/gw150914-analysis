# GW150914 Reproducible Figure Pipeline

A private archive of a from-scratch, publication-grade figure pipeline for GW150914. It explicitly separates visualization conditioning from analysis-grade conditioning and orchestrates strain acquisition, whitening, PSD/Q-transform analysis, detector alignment, matched filtering, chirp-mass/ringdown fits, posterior overlays, and robustness sweeps.

## Architecture

- `src/`: configuration/provenance, strain preprocessing, spectral/Q-transform tools, waveform overlays, matched filtering, ridge fitting, correlation, ringdown, posterior plotting, and consistent figure styling.
- `scripts/`: master build plus one reproducible script per main/appendix figure.
- `configs/gw150914.yaml`: centralized event, detector, conditioning, waveform, posterior, aesthetic, and robustness parameters.
- `pyproject.toml`, `environment.yml`: reproducible Python environments.

The archive excludes downloaded GWOSC/posterior data and generated PDF/PNG figures. The original README is retained as `LOCAL_README.md`; see `CODEX_AUDIT.md` for the curation boundary and validation record.

## Local validation

All retained source and figure scripts passed Python compilation, and the master CLI help path completed successfully. End-to-end figure generation requires the excluded downloaded data and heavyweight scientific dependencies.