# Curation audit

Source: C:\Users\deluc\Documents\GW150914.

## Included

- configs/gw150914.yaml, environment.yml, pyproject.toml, and the original README as LOCAL_README.md.
- src/: configuration/provenance, download hooks, preprocessing, PSD/spectral tools, waveform overlays, matched filtering, ridge fitting, detector correlation, ringdown, posterior plotting, synthetic waveform, and styling.
- scripts/: master build and all main/appendix figure-generation scripts.

## Validation

- All retained Python source and figure scripts passed Python compilation during curation.
- scripts/make_all_figures.py --help completed successfully.
- Full figure generation was not claimed because it downloads/uses the excluded GWOSC strain and posterior data and heavyweight scientific dependencies.

## Deliberate exclusions

- data/ downloaded strain, posterior samples, manifests, and intermediate data.
- figures/ generated PDF/PNG publication figures.
- scripts/_backup/, bytecode, virtual environments, caches, and transient output.
- No credentials or private keys were detected in the retained source/configuration surface.

This is a private, reproducible-code archive of the GW150914 analysis/visualization pipeline without publishing downloaded scientific data or generated figures.
