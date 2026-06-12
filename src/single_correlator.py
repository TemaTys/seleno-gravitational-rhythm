# -*- coding: utf-8 -*-
"""
File: single_correlator.py
Author: Artem Tysiatskii
Python 3.10.11

================================================================================
SGR Single-Database Correlator — Lunar Tidal-Wave Effect Estimator
================================================================================

Purpose
-------
Quantifies the impact of impulsive lunar tidal-wave events (syzygy / quadrature,
pre-detected on d²F/dt² of the local-noon tidal scalar) on daily socio-behavioral
counts of a SINGLE database (e.g. NYC 911 dispatches, NYC EMS calls). Unlike the
cross-city pipeline, no meta-analysis is performed: each database is a unique
data-generating process, so robustness is established INTERNALLY via permutation,
placebo, bootstrap and slice-stability tests.

Methodology
-----------
1) Detrending / deseasoning (OLS on log1p(counts), per database × epoch):
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
       Reported WAVE_* effects are therefore LOWER BOUNDS on the true signal.

3) Symmetric winsorization (1% / 1%) on residuals. Robust against riots and
   mass-event spikes; does not bias the residual mean.

4) Effect-size estimation:
       • Hedges' g with pooled variance + J small-sample correction
       • Analytic Var(g) (reported but not aggregated — single-DB design)
       • Two-sided Mann-Whitney U as non-parametric backstop
       • Δ% on original counts via exp(Δresid) − 1

5) Lag scheme (biological response may jitter ±1 d around the trigger):
       lag = −1   : day before event
       lag =  0   : day of event
       lag = +1   : day after event
       lag = WIN3 : 3-day exposure window  =  union of {−1, 0, +1} days
                    — timing-robust primary estimate. Uses a fixed window
                    (not per-event argmax) to avoid selection-bias /
                    double-dipping (winner's curse).

6) Internal robustness battery (lag = 0 and lag = WIN3):
       • Block-permutation tests, block lengths 14 and 28 days
       • Circular-shift permutation (≥ 7-day shift)
       • Bootstrap 95% CI for Hedges' g
         (i.i.d. day resampling; may be mildly anticonservative under
         residual autocorrelation and should be interpreted as descriptive
         uncertainty, whereas permutation p-values provide the
         autocorrelation-robust inferential check)
       • Slice stability over 5 contiguous time slices
       • Placebo resampling: random control subsets of size n_target
       • Quasi-Poisson GLM sanity check on raw counts

7) Multiple-testing: Benjamini-Hochberg FDR is COMPUTED and REPORTED per
   (epoch, family, lag) stratum, but tiers are assigned on raw p_mw — this
   is the exploratory/characterization layer; confirmatory control lives in
   the cross-city meta-analysis.

8) Strict gatekeeping:
       • mean(raw counts) ≥ 8/day
       • n_target ≥ 15 exposed days, n_ctrl ≥ 60 control days

Epochs (independent per database):
       PRE2014  — pre-2015 (replication cohort)
       PRE      — full pre-COVID timeline
       POST     — post-COVID timeline
       COVID    — optional, off by default (extreme regime shift)
       NB. PRE2014 is a NESTED sub-epoch of PRE (sensitivity stratum, not
       an independent cohort) — reported separately for diagnostic purposes.

Output: ../outputs/correlations/sgr_<db>_results.csv

References
----------
Hedges & Olkin (1985); Cohen (1988); Benjamini & Hochberg (1995);
Good (2005) Permutation, Parametric and Bootstrap Tests of Hypotheses.
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
    "nyc_911": {
        "crime_file":  DATA_DIR  / "nyc_911_daily.csv",
        "waves_file":  WAVES_DIR / "waves_nyc_2005_2025.csv",
    },
    "nyc_ems": {
        "crime_file":  DATA_DIR  / "nyc_ems_daily.csv",
        "waves_file":  WAVES_DIR / "waves_nyc_2005_2025.csv",
    },
}

INCLUDE_COVID_EPOCH = 0
COVID_START = "2020-03-01"
COVID_END   = "2021-06-30"

# OLS Detrending
POLY_DEGREE       = 4
FOURIER_HARMONICS = 3

# Statistical Gatekeeping
MIN_MEAN_RAW = 8.0
MIN_N_TARGET = 15
MIN_N_CTRL   = 60

# Heavy-statistics configuration
CALC_HEAVY_MATH_ALL_LAGS = 0
BOOTSTRAP_ITER  = 1000
PERM_ITER       = 1000
N_SLICES        = 5
MIN_N_PER_SLICE = 5
PLACEBO_ITER    = 200

# Lags: −1 (before), 0 (day of), +1 (after), WIN3 (3-day union window)
LAGS = [-1, 0, 1, 'WIN3']

# Symmetric winsorization on residuals
WINSOR_LIMITS = (0.01, 0.01)


def _config_hash() -> str:
    h = hashlib.md5()
    keys = ["POLY_DEGREE", "FOURIER_HARMONICS", "MIN_MEAN_RAW", "MIN_N_TARGET",
            "MIN_N_CTRL", "LAGS", "BOOTSTRAP_ITER", "PERM_ITER", "N_SLICES",
            "MIN_N_PER_SLICE", "PLACEBO_ITER", "WINSOR_LIMITS",
            "COVID_START", "COVID_END", "SEED"]
    for k in keys:
        h.update(f"{k}={globals()[k]};".encode())
    return h.hexdigest()[:10]


CONFIG_HASH = _config_hash()

# ════════════════════════════════════════════════════════════════════════
# TRIGGER REGISTRY
# Branch A (no lunar dummies)  -> PHASE_* triggers   (phase vs non-phase)
# Branch B (with lunar dummies)-> WAVE_*  triggers   (wave above phase)
# ════════════════════════════════════════════════════════════════════════
TRIGGERS = [
    # WAVE — Branch B: dynamic ddF impulse beyond static phase baseline
    ("WAVE_event",             "wave_event",              "B", "WAVE"),
    ("WAVE_syzygy",            "wave_syzygy",             "B", "WAVE"),
    ("WAVE_quadrature",        "wave_quadrature",         "B", "WAVE"),
    ("WAVE_before_syzygy",     "wave_before_syzygy",      "B", "WAVE"),
    ("WAVE_after_syzygy",      "wave_after_syzygy",       "B", "WAVE"),
    ("WAVE_before_quadrature", "wave_before_quadrature",  "B", "WAVE"),
    ("WAVE_after_quadrature",  "wave_after_quadrature",   "B", "WAVE"),

    # PHASE — Branch A: pure phase-vs-non-phase contrast
    ("PHASE_Syzygy",           "is_syzygy",               "A", "PHASE"),
    ("PHASE_Quadrature",       "is_quadrature",           "A", "PHASE"),
    ("PHASE_Full",             "is_full",                 "A", "PHASE"),
    ("PHASE_New",              "is_new",                  "A", "PHASE"),
    ("PHASE_FirstQ",           "is_first",                "A", "PHASE"),
    ("PHASE_LastQ",            "is_last",                 "A", "PHASE"),
]

WAVES_COLS = {
    "dF_dt_micro", "d2F_dt2_micro", "lunar_declination", "perigee_proximity",
    "nodal_modulation", "apsidal_modulation", "lunar_phase",
    "wave_event", "wave_syzygy", "wave_quadrature",
    "wave_before_syzygy", "wave_after_syzygy",
    "wave_before_quadrature", "wave_after_quadrature",
}


# ════════════════════════════════════════════════════════════════════════
# DATA PROCESSING
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

    for phase in ["is_full", "is_new", "is_first", "is_last",
                  "is_syzygy", "is_quadrature"]:
        df[f"{phase}_p1"] = df[phase].shift(1, fill_value=0)
        df[f"{phase}_m1"] = df[phase].shift(-1, fill_value=0)

    cal = USFederalHolidayCalendar()
    hol = pd.DatetimeIndex(cal.holidays(start=df.index.min(), end=df.index.max()))
    df["is_holiday"]    = df.index.isin(hol).astype(int)
    df["is_holiday_p1"] = df.index.isin(hol + pd.Timedelta(days=1)).astype(int)
    df["is_holiday_m1"] = df.index.isin(hol - pd.Timedelta(days=1)).astype(int)

    return df, crime_cols


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


def winsorize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Symmetric two-sided winsorization for residuals."""
    ql = np.quantile(x, lo)
    qh = np.quantile(x, 1.0 - hi)
    return np.clip(x, ql, qh)


def build_ctrl_mask(df_e: pd.DataFrame, family: str) -> np.ndarray:
    if family == "PHASE":
        # Pure control: exclude any phase activity (including ±1 days)
        phase_any = np.zeros(len(df_e), dtype=bool)
        for phase in ["is_full", "is_new", "is_first", "is_last"]:
            phase_any |= (df_e[phase].values > 0)
            phase_any |= (df_e[f"{phase}_p1"].values > 0)
            phase_any |= (df_e[f"{phase}_m1"].values > 0)
        return ~phase_any
    # WAVE control: Branch B already partials out phase means -> all non-target days
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
    """Union of {-1, 0, +1} days around each event (fixed exposure window)."""
    n = len(mask)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.zeros(n, dtype=bool)
    win = np.concatenate([idx - 1, idx, idx + 1])
    win = win[(win >= 0) & (win < n)]
    out = np.zeros(n, dtype=bool)
    out[np.unique(win)] = True
    return out


# ════════════════════════════════════════════════════════════════════════
# STATISTICS
# ════════════════════════════════════════════════════════════════════════
def hedges_g(t: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    nt, nc = len(t), len(c)
    if nt < 2 or nc < 2:
        return np.nan, np.nan
    vt, vc = t.var(ddof=1), c.var(ddof=1)
    sp2 = ((nt - 1) * vt + (nc - 1) * vc) / (nt + nc - 2)
    if sp2 <= 0:
        return 0.0, np.nan
    d = (t.mean() - c.mean()) / np.sqrt(sp2)
    J = 1.0 - 3.0 / (4.0 * (nt + nc) - 9.0)
    g_val = J * d
    var_g_val = (nt + nc) / (nt * nc) + (g_val ** 2) / (2.0 * (nt + nc - 2))
    return float(g_val), float(var_g_val)


def quick_g(t: np.ndarray, c: np.ndarray) -> float:
    nt, nc = len(t), len(c)
    sp2 = ((nt - 1) * t.var(ddof=1) + (nc - 1) * c.var(ddof=1)) / (nt + nc - 2)
    if sp2 <= 0:
        return 0.0
    return (1 - 3.0 / (4 * (nt + nc) - 9)) * (t.mean() - c.mean()) / np.sqrt(sp2)


def bootstrap_ci(t: np.ndarray, c: np.ndarray, n_iter: int, rng) -> tuple[float, float]:
    nt, nc = len(t), len(c)
    it = rng.integers(0, nt, size=(n_iter, nt))
    ic = rng.integers(0, nc, size=(n_iter, nc))
    bt, bc = t[it], c[ic]
    sp = np.sqrt(((nt - 1) * bt.var(axis=1, ddof=1) +
                  (nc - 1) * bc.var(axis=1, ddof=1)) / (nt + nc - 2))
    sp[sp == 0] = np.nan
    J = 1 - 3.0 / (4 * (nt + nc) - 9)
    g = J * (bt.mean(axis=1) - bc.mean(axis=1)) / sp
    g = g[~np.isnan(g)]
    if len(g) < 10:
        return np.nan, np.nan
    lo, hi = np.percentile(g, [2.5, 97.5])
    return float(lo), float(hi)


def perm_block(resid: np.ndarray, t_mask: np.ndarray, c_mask: np.ndarray,
               block: int, n_iter: int, g_obs: float, rng) -> float:
    n  = len(resid)
    nb = n // block
    if nb < 4:
        return np.nan
    n_use = nb * block
    blk = resid[:n_use].reshape(nb, block)
    tm, cm = t_mask[:n_use], c_mask[:n_use]
    if tm.sum() < 2 or cm.sum() < 2:
        return np.nan
    null = np.empty(n_iter)
    for i in range(n_iter):
        r = blk[rng.permutation(nb)].ravel()
        null[i] = quick_g(r[tm], r[cm])
    return float(np.mean(np.abs(null) >= np.abs(g_obs)))


def perm_circ(resid: np.ndarray, t_mask: np.ndarray, c_mask: np.ndarray,
              n_iter: int, g_obs: float, rng) -> float:
    n = len(resid)
    if n < 30:
        return np.nan
    null   = np.empty(n_iter)
    shifts = rng.integers(7, n - 7, size=n_iter)
    for i in range(n_iter):
        r = np.roll(resid, shifts[i])
        null[i] = quick_g(r[t_mask], r[c_mask])
    return float(np.mean(np.abs(null) >= np.abs(g_obs)))


def slice_stability(resid: np.ndarray, t_mask: np.ndarray, c_mask: np.ndarray,
                    n_slices: int, min_n: int):
    n  = len(resid)
    sz = n // n_slices
    gs = []
    for i in range(n_slices):
        a = i * sz
        b = (i + 1) * sz if i < n_slices - 1 else n
        tm, cm = t_mask[a:b], c_mask[a:b]
        if tm.sum() < min_n or cm.sum() < min_n * 4:
            gs.append(np.nan)
            continue
        g, _ = hedges_g(resid[a:b][tm], resid[a:b][cm])
        gs.append(g)
    arr = np.array(gs)
    valid = ~np.isnan(arr)
    if valid.sum() < 2:
        return gs, np.nan, np.nan, int(valid.sum())
    overall_sign = np.sign(np.nanmean(arr))
    sign_frac    = float(np.mean(np.sign(arr[valid]) == overall_sign))
    iqr          = float(np.percentile(arr[valid], 75) - np.percentile(arr[valid], 25))
    return gs, sign_frac, iqr, int(valid.sum())


def placebo_test(resid: np.ndarray, c_mask: np.ndarray, n_target: int,
                 n_iter: int, g_obs: float, rng) -> tuple[float, float]:
    c_idx = np.where(c_mask)[0]
    if len(c_idx) <= n_target * 2:
        return np.nan, np.nan
    g_plac = np.empty(n_iter)
    for i in range(n_iter):
        sel = rng.choice(c_idx, size=n_target, replace=False)
        m = np.zeros(len(resid), dtype=bool); m[sel] = True
        c_eff = c_mask & ~m
        g_plac[i] = quick_g(resid[m], resid[c_eff])
    return float(np.mean(g_plac)), float(np.mean(np.abs(g_plac) >= np.abs(g_obs)))


def quasi_poisson_sanity(y_count: np.ndarray, X: np.ndarray, trig: np.ndarray):
    Xt = np.column_stack([X, trig.astype(float)])
    try:
        res = sm.GLM(y_count, Xt, family=sm.families.Poisson()).fit(scale="X2")
        return float(res.params[-1]), float(res.pvalues[-1])
    except Exception:
        return np.nan, np.nan


def bh_fdr(p: np.ndarray) -> np.ndarray:
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

    epochs_to_run = (["PRE2014", "PRE", "COVID", "POST"]
                     if INCLUDE_COVID_EPOCH else ["PRE2014", "PRE", "POST"])

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

        XA = build_design(df_e, with_lunar=False)   # Branch A: no lunar dummies
        XB = build_design(df_e, with_lunar=True)    # Branch B: full lunar dummies

        trig_src = {}
        for name, col, _, _ in TRIGGERS:
            trig_src[name] = ((df_e[col].values > 0) if col in df_e.columns
                              else np.zeros(n_e, dtype=bool))

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
                resid_v = rA_w if branch == "A" else rB_w
                X_qp    = XA   if branch == "A" else XB
                t_src   = trig_src[name]
                ctrl_mask = build_ctrl_mask(df_e, fam)
                lag_idx = []

                for lag in LAGS:
                    if lag == 'WIN3':
                        # Fixed 3-day exposure window (union of ±1 around each event).
                        # No per-event argmax -> no winner's curse.
                        t_mask = win3_mask(t_src)
                        c_eff_mask = ctrl_mask & ~t_mask
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
                        lag_idx.append(len(rows) - 1)
                        continue

                    t_vals = resid_v[t_mask]
                    c_vals = resid_v[c_eff_mask]

                    g, var_g = hedges_g(t_vals, c_vals)
                    p_mw  = float(mannwhitneyu(t_vals, c_vals,
                                               alternative="two-sided").pvalue)
                    delta = (np.exp(t_vals.mean() - c_vals.mean()) - 1) * 100

                    base.update({
                        "g": g, "var_g": var_g, "p_mw": p_mw,
                        "delta_pct": float(delta), "status_flag": "OK",
                    })

                    heavy = (lag == 0) or (lag == 'WIN3') or CALC_HEAVY_MATH_ALL_LAGS
                    if heavy:
                        g_lo, g_hi = bootstrap_ci(t_vals, c_vals, BOOTSTRAP_ITER, RNG)
                        base["g_lo"] = g_lo; base["g_hi"] = g_hi

                        qc, qp = quasi_poisson_sanity(
                            df_e[crime].values.astype(float), X_qp, t_mask.astype(int))
                        base["qp_coef"] = qc; base["qp_p"] = qp
                        base["model_disagree"] = int(
                            (not np.isnan(qc)) and (not np.isnan(g))
                            and np.sign(qc) != np.sign(g))

                        base["p_perm_14"]   = perm_block(resid_v, t_mask, c_eff_mask,
                                                         14, PERM_ITER, g, RNG)
                        base["p_perm_28"]   = perm_block(resid_v, t_mask, c_eff_mask,
                                                         28, PERM_ITER, g, RNG)
                        base["p_perm_circ"] = perm_circ(resid_v, t_mask, c_eff_mask,
                                                        PERM_ITER, g, RNG)

                        gs, sf, iqr, nv = slice_stability(
                            resid_v, t_mask, c_eff_mask, N_SLICES, MIN_N_PER_SLICE)
                        for i in range(N_SLICES):
                            base[f"slice_g_{i+1}"] = (gs[i] if i < len(gs) else np.nan)
                        base["slice_sign_frac"] = sf
                        base["slice_g_iqr"]     = iqr
                        base["n_slices_valid"]  = nv

                        gp_m, p_pl = placebo_test(resid_v, c_eff_mask, nt,
                                                  PLACEBO_ITER, g, RNG)
                        base["g_placebo_mean"] = gp_m
                        base["p_placebo_emp"]  = p_pl
                    else:
                        for k in ("g_lo", "g_hi", "qp_coef", "qp_p",
                                  "model_disagree", "p_perm_14", "p_perm_28",
                                  "p_perm_circ", "slice_sign_frac", "slice_g_iqr",
                                  "n_slices_valid", "g_placebo_mean", "p_placebo_emp"):
                            base[k] = np.nan
                        for i in range(N_SLICES):
                            base[f"slice_g_{i+1}"] = np.nan

                    rows.append(base)
                    lag_idx.append(len(rows) - 1)

                # argmax_lag diagnostic: best signed-int lag by |g| (informational)
                gs_lag = [(rows[ri]["lag"], rows[ri].get("g", np.nan)) for ri in lag_idx]
                vals = [(l, abs(g)) for l, g in gs_lag
                        if not (isinstance(g, float) and np.isnan(g)) and isinstance(l, int)]
                am = (sorted(vals, key=lambda x: -x[1])[0][0]) if vals else np.nan
                for ri in lag_idx:
                    rows[ri]["argmax_lag"] = am

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    out["p_mw_fdr"] = np.nan
    ok = out["status_flag"] == "OK"
    if ok.any():
        for _, idx in out[ok].groupby(
                ["epoch", "trigger_family", "lag"], sort=False).groups.items():
            out.loc[idx, "p_mw_fdr"] = bh_fdr(out.loc[idx, "p_mw"].values)

    out["config_hash"] = CONFIG_HASH

    col_order = [
        "city", "epoch", "crime_type", "trigger", "trigger_family", "lag",
        "mean_raw", "n_days_epoch", "R2_A", "R2_B", "n_target", "n_ctrl",
        "g", "var_g", "p_mw", "p_mw_fdr", "delta_pct",
        "g_lo", "g_hi", "qp_coef", "qp_p", "model_disagree",
        "p_perm_14", "p_perm_28", "p_perm_circ",
        "slice_g_1", "slice_g_2", "slice_g_3", "slice_g_4", "slice_g_5",
        "slice_sign_frac", "slice_g_iqr", "n_slices_valid",
        "g_placebo_mean", "p_placebo_emp",
        "argmax_lag", "status_flag", "config_hash",
    ]
    out = out.reindex(columns=col_order)

    out_path = OUTPUT_DIR / f"sgr_{city}_results.csv"
    out.to_csv(out_path, index=False)
    print(f"[{city}] Done in {time.time() - t0:.1f}s -> {out_path} "
          f"({len(out)} rows)", flush=True)
    return out


def main():
    print(f"SGR Single Correlator Initialized | Config Hash: {CONFIG_HASH}",
          flush=True)
    cities_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for c in cities_to_run:
        if c not in CITIES:
            print(f"Unknown city requested: {c}", flush=True)
            continue
        process_city(c)
    print("Execution complete.", flush=True)


if __name__ == "__main__":
    main()