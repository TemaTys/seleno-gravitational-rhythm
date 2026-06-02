# -*- coding: utf-8 -*-
"""
File: correlator.py
Author: Artem Tysiatskii
Python 3.10.11

================================================================================
SGR Correlator — Lunar Tidal-Wave Effect Estimator (per-city engine)
================================================================================

Purpose
-------
Quantifies the impact of impulsive lunar tidal-wave events (syzygy / quadrature,
pre-detected on d²F/dt² of the local-noon tidal scalar) on city-level daily
crime counts. Outputs Hedges' g + Var(g) per (city, epoch, crime, trigger, lag),
to be aggregated by verdict.py via a random-effects meta-analysis.

Methodology
-----------
1) Detrending / deseasoning (OLS on log1p(counts), per city × epoch):
       • Polynomial trend (degree 4)
       • Annual Fourier basis (3 harmonics)
       • Day-of-week dummies (6)
       • US federal holidays with ±1-day lags

2) Orthogonal Branch A vs Branch B residualization:
       Branch A — no lunar dummies in OLS. Residuals retain ALL lunar variance.
                  Used for PHASE_* triggers (clean phase-vs-non-phase contrast).
       Branch B — full set of static phase dummies (Full/New/First/Last × {-1,0,+1}).
                  Residuals have static phase means partialled out. Used for
                  WAVE_* triggers: tests whether the impulsive ddF event adds
                  signal BEYOND the static phase baseline.

       NOTE — conservative design: because wave events fall within ±2 d of
       phase, the phase dummies in Branch B absorb part of the wave variance.
       Therefore reported WAVE_* effects are LOWER BOUNDS on the true signal.

3) Symmetric winsorization (1% / 1%) on residuals. Robust against riots and
   mass-event spikes; does not bias the residual mean.

4) Effect-size estimation:
       • Hedges' g with pooled variance + J small-sample correction
       • Var(g) for downstream random-effects meta-analysis
       • Two-sided Mann-Whitney U as non-parametric backstop
       • Δ% on original counts via exp(Δresid)-1

5) Lag scheme (biological response may jitter ±1 d around the trigger):
       lag = -1   : day before event
       lag =  0   : day of event
       lag = +1   : day after event
       lag = WIN3 : 3-day windowed EXPOSURE  =  union of {-1, 0, +1}
                    — timing-robust primary estimate. Uses a fixed window
                    (not per-event argmax) to avoid selection-bias /
                    double-dipping.

6) Multiple-testing control: Benjamini-Hochberg FDR within each
   (epoch, trigger_family, lag) stratum across crime types of one city.

7) Strict gatekeeping:
       • mean(raw counts) ≥ 8/day
       • n_target ≥ 15 exposed days, n_ctrl ≥ 60 control days

Epochs (independent per city):
       PRE2014  — pre-2015 (replication cohort, pre-decrim era in several US cities)
       PRE      — full pre-COVID timeline
       POST     — post-COVID timeline
       COVID    — optional, off by default (extreme regime shift)

Output: ../outputs/correlations/sgr_<city>_results.csv

References
----------
Hedges & Olkin (1985); Borenstein et al. (2009) Introduction to Meta-Analysis;
Benjamini & Hochberg (1995).
"""
from __future__ import annotations

import hashlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════
SEED = 42
RNG = np.random.default_rng(SEED)

DATA_DIR   = Path("../data/interim")
WAVES_DIR  = Path("../data/waves")
OUTPUT_DIR = Path("../outputs/correlations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = {
    "chicago": {
        "crime_file":  DATA_DIR  / "chicago_daily.csv",
        "waves_file":  WAVES_DIR / "waves_chicago_2001_2026.csv",
    },
    "la": {
        "crime_file":  DATA_DIR  / "la_daily.csv",
        "waves_file":  WAVES_DIR / "waves_la_2010_2022.csv",
    },
    "philly": {
        "crime_file":  DATA_DIR  / "philly_daily.csv",
        "waves_file":  WAVES_DIR / "waves_philly_2006_2026.csv",
    },
    "nyc": {
        "crime_file":  DATA_DIR  / "nyc_crime_daily.csv",
        "waves_file":  WAVES_DIR / "waves_nyc_2005_2025.csv",
    },
    "sf": {
        "crime_file":  DATA_DIR  / "sf_daily.csv",
        "waves_file":  WAVES_DIR / "waves_sf_2003_2017.csv",
    },
}

INCLUDE_COVID_EPOCH = 0
COVID_START = "2020-03-01"
COVID_END   = "2021-06-30"

# OLS detrending
POLY_DEGREE       = 4
FOURIER_HARMONICS = 3

# Gatekeeping
MIN_MEAN_RAW = 8.0
MIN_N_TARGET = 15
MIN_N_CTRL   = 60

# Lags: integer = single-day shift; 'WIN3' = 3-day windowed exposure (union of -1,0,+1)
LAGS = [-1, 0, 1, 'WIN3']

# Symmetric winsorization (lower / upper tail fractions)
WINSOR_LIMITS = (0.01, 0.01)


def _config_hash() -> str:
    h = hashlib.md5()
    keys = ["POLY_DEGREE", "FOURIER_HARMONICS", "MIN_MEAN_RAW", "MIN_N_TARGET",
            "MIN_N_CTRL", "LAGS", "WINSOR_LIMITS", "COVID_START", "COVID_END", "SEED"]
    for k in keys:
        h.update(f"{k}={globals()[k]};".encode())
    return h.hexdigest()[:10]


CONFIG_HASH = _config_hash()

# ════════════════════════════════════════════════════════════════════════
# TRIGGER REGISTRY
# Branch A: pure signal (no lunar dummies in OLS) — for PHASE_* triggers.
# Branch B: strict (lunar dummies in OLS)        — for WAVE_*  triggers.
# ════════════════════════════════════════════════════════════════════════
TRIGGERS = [
    # WAVE_* — impulsive ddF events. Branch B isolates wave dynamics beyond
    # the static phase baseline. Conservative — reported effects are lower bounds.
    ("WAVE_event",             "wave_event",              "B", "WAVE"),
    ("WAVE_syzygy",            "wave_syzygy",             "B", "WAVE"),
    ("WAVE_quadrature",        "wave_quadrature",         "B", "WAVE"),
    ("WAVE_before_syzygy",     "wave_before_syzygy",      "B", "WAVE"),
    ("WAVE_after_syzygy",      "wave_after_syzygy",       "B", "WAVE"),
    ("WAVE_before_quadrature", "wave_before_quadrature",  "B", "WAVE"),
    ("WAVE_after_quadrature",  "wave_after_quadrature",   "B", "WAVE"),

    # PHASE_* — static calendar phases evaluated against a clean non-phase baseline.
    ("PHASE_Syzygy",     "is_syzygy",     "A", "PHASE"),
    ("PHASE_Quadrature", "is_quadrature", "A", "PHASE"),
    ("PHASE_Full",       "is_full",       "A", "PHASE"),
    ("PHASE_New",        "is_new",        "A", "PHASE"),
    ("PHASE_FirstQ",     "is_first",      "A", "PHASE"),
    ("PHASE_LastQ",      "is_last",       "A", "PHASE"),
]

WAVES_COLS = {
    "dF_dt_micro", "d2F_dt2_micro", "lunar_declination", "perigee_proximity",
    "nodal_modulation", "apsidal_modulation", "lunar_phase",
    "wave_event", "wave_syzygy", "wave_quadrature",
    "wave_before_syzygy", "wave_after_syzygy",
    "wave_before_quadrature", "wave_after_quadrature",
}

# ════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════
def load_city(city: str) -> tuple[pd.DataFrame, list[str]]:
    cfg = CITIES[city]
    crime = pd.read_csv(cfg["crime_file"], parse_dates=["date"]).set_index("date")
    waves = pd.read_csv(cfg["waves_file"], parse_dates=["date"]).set_index("date")

    crime_cols = [c for c in crime.columns if c not in WAVES_COLS]
    df = crime.join(waves, how="inner").asfreq("D")
    assert df.index.is_monotonic_increasing and not df.index.has_duplicates

    for c in crime_cols:
        df[c] = df[c].fillna(0)

    event_cols = [
        "wave_event", "wave_syzygy", "wave_quadrature",
        "wave_before_syzygy", "wave_after_syzygy",
        "wave_before_quadrature", "wave_after_quadrature",
    ]
    for c in event_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    mp = df["lunar_phase"].fillna("").astype(str)
    df["is_full"]  = (mp == "Full Moon").astype(int)
    df["is_new"]   = (mp == "New Moon").astype(int)
    df["is_first"] = (mp == "First Quarter").astype(int)
    df["is_last"]  = (mp == "Last Quarter").astype(int)

    df["is_syzygy"]     = (df["is_full"]  | df["is_new"]).astype(int)
    df["is_quadrature"] = (df["is_first"] | df["is_last"]).astype(int)

    for phase in ["is_full", "is_new", "is_first", "is_last", "is_syzygy", "is_quadrature"]:
        df[f"{phase}_p1"] = df[phase].shift(1,  fill_value=0)
        df[f"{phase}_m1"] = df[phase].shift(-1, fill_value=0)

    cal = USFederalHolidayCalendar()
    hol = pd.DatetimeIndex(cal.holidays(start=df.index.min(), end=df.index.max()))
    df["is_holiday"]    = df.index.isin(hol).astype(int)
    df["is_holiday_p1"] = df.index.isin(hol + pd.Timedelta(days=1)).astype(int)
    df["is_holiday_m1"] = df.index.isin(hol - pd.Timedelta(days=1)).astype(int)

    return df, crime_cols


# ════════════════════════════════════════════════════════════════════════
# DESIGN MATRIX & RESIDUALIZATION
# ════════════════════════════════════════════════════════════════════════
def build_design(df_e: pd.DataFrame, with_lunar: bool) -> np.ndarray:
    n = len(df_e)
    cols = [np.ones(n)]

    t = np.arange(n) / max(n - 1, 1)
    for k in range(1, POLY_DEGREE + 1):
        cols.append(t ** k)

    doy = df_e.index.dayofyear.values
    for k in range(1, FOURIER_HARMONICS + 1):
        cols.append(np.sin(2 * np.pi * k * doy / 365.25))
        cols.append(np.cos(2 * np.pi * k * doy / 365.25))

    dow = df_e.index.dayofweek.values
    for d in range(1, 7):
        cols.append((dow == d).astype(float))

    for c in ["is_holiday", "is_holiday_p1", "is_holiday_m1"]:
        cols.append(df_e[c].values.astype(float))

    if with_lunar:
        for phase in ["is_full", "is_new", "is_first", "is_last"]:
            cols.append(df_e[phase].values.astype(float))
            cols.append(df_e[f"{phase}_p1"].values.astype(float))
            cols.append(df_e[f"{phase}_m1"].values.astype(float))

    return np.column_stack(cols)


def fit_resid(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    res = sm.OLS(y, X).fit()
    return res.resid, float(res.rsquared)


# ════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════
def winsorize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Symmetric winsorization on residuals: dampens extreme spikes
    (riots, mass events) without skewing the mean of the residual."""
    ql = np.quantile(x, lo)
    qh = np.quantile(x, 1.0 - hi)
    return np.clip(x, ql, qh)


def build_ctrl_mask(df_e: pd.DataFrame, family: str) -> np.ndarray:
    """
    Dynamic control mask by trigger family.

    PHASE: exclude any day within ±1 d of ANY moon phase from controls
           (Branch A residuals still carry phase variance — must hold out).
    WAVE : Branch B residuals already have the phase mean removed;
           the only requirement is that controls are not target days,
           which is enforced downstream via `~t_mask`.
    """
    if family == "PHASE":
        phase_any = np.zeros(len(df_e), dtype=bool)
        for phase in ["is_full", "is_new", "is_first", "is_last"]:
            phase_any |= (df_e[phase].values > 0)
            phase_any |= (df_e[f"{phase}_p1"].values > 0)
            phase_any |= (df_e[f"{phase}_m1"].values > 0)
        return ~phase_any
    return np.ones(len(df_e), dtype=bool)


def shifted_mask(mask: np.ndarray, lag: int) -> np.ndarray:
    n = len(mask)
    out = np.zeros(n, dtype=bool)
    if lag == 0:
        return mask.copy()
    if lag > 0:
        out[lag:] = mask[:n - lag]
    else:
        out[:n + lag] = mask[-lag:]
    return out


def win3_mask(mask: np.ndarray) -> np.ndarray:
    """3-day windowed exposure: union of {-1, 0, +1} day shifts of `mask`.
    Used as the timing-robust primary lag (no per-event argmax — avoids
    selection bias / double dipping)."""
    n = len(mask)
    out = mask.copy()
    out[1:]  |= mask[:n - 1]   # +1 day
    out[:n-1] |= mask[1:]      # -1 day
    return out


# ════════════════════════════════════════════════════════════════════════
# STATISTICS
# ════════════════════════════════════════════════════════════════════════
def hedges_g(t: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    """Hedges' g with pooled variance and J small-sample correction.
    Returns (g, Var(g)) ready for random-effects meta-analysis."""
    nt, nc = len(t), len(c)
    if nt < 2 or nc < 2:
        return np.nan, np.nan

    vt = t.var(ddof=1)
    vc = c.var(ddof=1)
    sp2 = ((nt - 1) * vt + (nc - 1) * vc) / (nt + nc - 2)
    if sp2 <= 0:
        return 0.0, np.nan

    d = (t.mean() - c.mean()) / np.sqrt(sp2)
    J = 1.0 - (3.0 / (4.0 * (nt + nc) - 9.0))
    g_val = J * d
    var_g_val = (nt + nc) / (nt * nc) + (g_val ** 2) / (2.0 * (nt + nc - 2))
    return float(g_val), float(var_g_val)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg step-up FDR adjustment."""
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]; m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    res = np.empty(m); res[order] = adj
    out[valid] = res
    return out


# ════════════════════════════════════════════════════════════════════════
# CORE ENGINE
# ════════════════════════════════════════════════════════════════════════
def process_city(city: str) -> pd.DataFrame:
    t0 = time.time()
    print(f"[{city}] Processing...", flush=True)
    df, crime_cols = load_city(city)
    rows = []

    epochs_to_run = ["PRE2014", "PRE", "COVID", "POST"] if INCLUDE_COVID_EPOCH else ["PRE2014", "PRE", "POST"]

    for epoch in epochs_to_run:
        if epoch == "PRE2014":
            df_e = df[df.index.year <= 2014].copy()
        elif epoch == "PRE":
            df_e = df[df.index < COVID_START].copy()
        elif epoch == "COVID":
            df_e = df[(df.index >= COVID_START) & (df.index <= COVID_END)].copy()
        elif epoch == "POST":
            df_e = df[df.index > COVID_END].copy()
        else:
            continue

        n_e = len(df_e)
        if n_e < 365:
            continue

        XA = build_design(df_e, with_lunar=False)
        XB = build_design(df_e, with_lunar=True)

        trig_src = {
            name: ((df_e[col].values > 0) if col in df_e.columns else np.zeros(n_e, dtype=bool))
            for name, col, _, _ in TRIGGERS
        }

        for crime in crime_cols:
            mean_raw = float(df_e[crime].mean())
            if mean_raw < MIN_MEAN_RAW:
                continue

            y = np.log1p(df_e[crime].values.astype(float))
            try:
                rA, R2A = fit_resid(y, XA)
                rB, R2B = fit_resid(y, XB)
            except Exception:
                continue

            rA_w = winsorize(rA, *WINSOR_LIMITS)
            rB_w = winsorize(rB, *WINSOR_LIMITS)

            for name, _, branch, fam in TRIGGERS:
                resid_v   = rA_w if branch == "A" else rB_w
                t_src     = trig_src[name]
                ctrl_mask = build_ctrl_mask(df_e, fam)

                for lag in LAGS:
                    if lag == 'WIN3':
                        # Windowed exposure (union ±1 day), NOT per-event argmax.
                        # Honest timing-robust estimator: accommodates ±1 d
                        # biological jitter without selection bias.
                        t_mask = win3_mask(t_src)
                    else:
                        t_mask = shifted_mask(t_src, int(lag))

                    c_eff_mask = ctrl_mask & ~t_mask
                    nt = int(t_mask.sum())
                    nc = int(c_eff_mask.sum())

                    base = {
                        "city": city, "epoch": epoch, "crime_type": crime,
                        "mean_raw": mean_raw, "n_days_epoch": n_e,
                        "R2_A": R2A, "R2_B": R2B,
                        "trigger": name, "trigger_family": fam, "lag": lag,
                        "n_target": nt, "n_ctrl": nc,
                    }

                    if nt < MIN_N_TARGET or nc < MIN_N_CTRL:
                        base["status_flag"] = "INSUFFICIENT_N"
                        rows.append(base)
                        continue

                    t_vals = resid_v[t_mask]
                    c_vals = resid_v[c_eff_mask]

                    g, var_g = hedges_g(t_vals, c_vals)
                    p_mw = float(mannwhitneyu(t_vals, c_vals, alternative="two-sided").pvalue)
                    delta = (np.exp(t_vals.mean() - c_vals.mean()) - 1) * 100

                    base.update({
                        "g": g, "var_g": var_g, "p_mw": p_mw,
                        "delta_pct": float(delta), "status_flag": "OK",
                    })
                    rows.append(base)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    # BH-FDR within (epoch, trigger_family, lag) across crime types of this city.
    out["p_mw_fdr"] = np.nan
    ok = out["status_flag"] == "OK"
    if ok.any():
        for _, idx in out[ok].groupby(["epoch", "trigger_family", "lag"], sort=False).groups.items():
            out.loc[idx, "p_mw_fdr"] = bh_fdr(out.loc[idx, "p_mw"].values)

    out["config_hash"] = CONFIG_HASH

    col_order = [
        "city", "epoch", "crime_type", "trigger", "trigger_family", "lag",
        "mean_raw", "n_days_epoch", "R2_A", "R2_B", "n_target", "n_ctrl",
        "g", "var_g", "p_mw", "p_mw_fdr", "delta_pct",
        "status_flag", "config_hash",
    ]
    out = out.reindex(columns=col_order)

    out_path = OUTPUT_DIR / f"sgr_{city}_results.csv"
    out.to_csv(out_path, index=False)
    print(f"[{city}] Done in {time.time() - t0:.1f}s -> {out_path} ({len(out)} rows)", flush=True)
    return out


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    print(f"SGR Correlator Initialized | Config Hash: {CONFIG_HASH}", flush=True)
    cities_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for c in cities_to_run:
        if c not in CITIES:
            print(f"Unknown city requested: {c}", flush=True)
            continue
        process_city(c)
    print("Execution complete.", flush=True)


if __name__ == "__main__":
    main()