# Seleno-Gravitational Rhythm (SGR)

**Cross-city meta-analysis and open Python pipeline for studying impulsive lunar tidal waves as modulators of human consciousness, behavior, and somatic health.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10.11](https://img.shields.io/badge/python-3.10.11-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20518661.svg)](https://doi.org/10.5281/zenodo.20518661)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--1974--7894-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0006-1974-7894)

---

## What this repository is

This repository contains the **full reproducible pipeline** behind the SGR study —
the first cross-city meta-analysis of impulsive lunar tidal waves (second
derivative of the tidal potential, *d²F/dt²*) as a population-level modulator
of crime, emergency calls, and acute medical events across five US
metropolitan areas (Chicago, Los Angeles, New York, Philadelphia, San
Francisco) over 2001–2025, covering **~106 million records** from seven
official open databases.

The work introduces:

- a novel **ddF detector** of impulsive lunar tidal jerks;
- an **orthogonal decomposition** (Branch A / Branch B) of the phasic vs.
  wave-driven components of the lunar signal;
- a **HKSJ random-effects meta-analysis** with TIER classification;
- an internal **robustness battery** (block-permutation, circular-shift,
  bootstrap, slice stability, placebo) for unique single-city databases.

Key empirical findings — a TIER1 cross-city replication for property crime
in the pre-quadrature wave window, and a structural sizygy/quadrature
dichotomy across cardiovascular, obstetric, and psychiatric outcomes — are
detailed in the accompanying article (see *How to cite*).

---

## How to cite

If you use this code, data, or findings, please cite both:

> Tysiatskii, A. (2026). *Seleno-Gravitational Rhythm (SGR): Impulsive lunar
> tidal waves as a modulator of human consciousness, behavior, and somatic
> health. Cross-city meta-analysis of US metropolitan crime and medical
> statistics (2001–2025).* Zenodo. https://doi.org/10.5281/zenodo.20518661

BibTeX and a machine-readable citation are provided in [`CITATION.cff`](CITATION.cff).

---

## Repository structure

```
sgr/
├── README.md                ← this file
├── LICENSE                  ← MIT
├── CITATION.cff             ← machine-readable citation
├── requirements.txt         ← exact library versions
├── src/
│   ├── base.py              ← raw → unified daily CSV
│   ├── gen.py               ← astronomy engine, ddF detector, wave/phase triggers
│   ├── correlator.py        ← Branch A/B detrending, Hedges' g, BH-FDR (multi-city)
│   ├── verdict.py           ← random-effects + HKSJ meta-analysis, TIER ranking
│   ├── single_correlator.py ← single-DB pipeline with full robustness battery
│   └── single_verdict.py    ← TIER ranking for single-DB outputs
├── data/
│   ├── raw/                 ← (user downloads city CSVs here — see data/raw/README.md)
│   ├── interim/             ← unified daily CSVs (chicago_daily.csv, etc.)
│   ├── waves/               ← computed lunar wave files (waves_<city>_<years>.csv)
│   └── external/            ← JPL DE421 ephemeris (de421.bsp)
├── outputs/
│   ├── correlations/        ← per-city correlation tables
│   └── verdicts/            ← meta-analysis and TIER tables
├── figures/                 ← forest plot, ddF schema, cycle map
└── docs/
    ├── semantic_map.md                     ← full SEMANTIC_MAP of crime codes
    ├── semantic_map_single.tsv             ← single-DB signals dictionary (EN/RU)
    ├── Seleno_Gravitational_Rhythm_RU.pdf  ← article (Russian)
    └── Seleno_Gravitational_Rhythm_EN.pdf  ← article (English)
```

---

## Reproducing the analysis

### 1. Environment

```bash
python --version          # must be 3.10.11
pip install -r requirements.txt
```

Pinned versions: NumPy 2.2.4, pandas 2.2.3, SciPy 1.15.2,
statsmodels 0.14.6, Skyfield 1.53.

### 2. Data

Raw municipal CSVs are **not redistributed** in this repo (size constraints
and source-of-truth policy). Download them yourself following the
instructions in [`data/raw/README.md`](data/raw/README.md). The JPL DE421
ephemeris file (`de421.bsp`) is already shipped in `data/external/`.

### 3. Pipeline

Run the scripts sequentially:

```bash
# 1. Unify raw city CSVs into daily tables
python src/base.py

# 2. Compute lunar tidal scalar, derivatives, and wave/phase triggers
python src/gen.py

# 3. Per-city correlations (Branch A / Branch B, Hedges' g, BH-FDR)
python src/correlator.py

# 4. Cross-city meta-analysis (random-effects + HKSJ, TIER classification)
python src/verdict.py

# 5. Single-database pipeline (NYC 911, NYC EMS) + robustness battery
python src/single_correlator.py
python src/single_verdict.py
```

Total runtime on a modern laptop: ~30–60 min depending on raw-data size.

---

## Key outputs

- `outputs/verdicts/sgr_meta_family.csv` — cross-city meta-analysis (family level)
- `outputs/verdicts/sgr_meta_crimetype.csv` — per-crime-type meta results
- `outputs/verdicts/sgr_forest_data.csv` — data for the headline forest plot
- `outputs/verdicts/sgr_single_verdict_signals.csv` — TIER1/2/3 single-DB signals
- `figures/figure_1_ddF_schema.png` — ddF detector schema
- `figures/figure_2_forest.png` — headline forest plot (PROPERTY_CRIME × WAVE_before_quadrature)
- `figures/figure_3_cycle_map.png` — lunar-cycle signal map

---

## Methodology in one paragraph

For every city, the lunar+solar tidal scalar *F(t)*, its first derivative
*dF/dt* and second derivative *d²F/dt²* are computed at local solar noon
from JPL DE421 ephemerides. A formal threshold-based detector flags
short-lived *d²F/dt²* jerks (`ddF events`) and classifies them by their
nearest lunar phase (sizygy/quadrature, before/after). Daily event counts
are log-transformed, detrended against polynomial time, annual Fourier
harmonics, weekdays and US federal holidays, and winsorized. Two parallel
residual branches are produced: **Branch A** (no lunar covariates — for
static phase contrasts) and **Branch B** (with 12 static-phase dummies
partialled out — for wave-only effects). Effect sizes (Hedges' *g* with
small-sample correction) and a non-parametric Mann–Whitney backstop are
computed per (city × epoch × crime × trigger × lag), with BH-FDR within
strata. Cross-city aggregation uses **random-effects meta-analysis with
HKSJ correction** (the modern standard for small *k*). Single-database
signals additionally pass a robustness battery (block-permutation,
circular-shift, bootstrap, slice stability, placebo).

---

## License

- **Code:** MIT License (see [`LICENSE`](LICENSE)).
- **Derived data tables and figures:** Creative Commons Attribution 4.0
  International (CC-BY 4.0).
- **Raw municipal data:** licensed by the respective city open-data
  portals; please consult each portal's terms.

---

## Author

**Artem Tysiatskii** — Independent researcher and software developer.
ORCID: [0009-0006-1974-7894](https://orcid.org/0009-0006-1974-7894)
Email: artem.tysiatskii@gmail.com

The author thanks the open-data programs of NYC, LA, Chicago,
Philadelphia, and San Francisco for making this analysis possible.

---

## Contact and contributions

Issues, replication attempts on other cities/countries, and methodological
critique are welcome via GitHub Issues. For dataset-specific questions,
please consult the source city open-data portal first.
