# -*- coding: utf-8 -*-
"""
File: verdict.py
Author: Artem Tysiatskii
Python 3.10.11

================================================================================
SGR Verdict — Cross-City Random-Effects Meta-Analysis & Tier Assignment
================================================================================

Purpose
-------
Aggregates per-city Hedges' g effect sizes (produced by correlator.py) into
a global random-effects meta-analysis. Two levels:
       • Family-level   : semantically-grouped crime aggregates (AGGRESSION,
                          PROPERTY_CRIME, NARCOTICS, ...).
       • Crime-type     : per individual crime type (diagnostic table).

Methodology
-----------
• Random-effects model with iterative Empirical-Bayes / REML τ² estimation.
• Hartung-Knapp-Sidik-Jonkman (HKSJ) variance adjustment — current standard
  for small-k meta-analysis (k ≤ 10 cohorts). Prevents anti-conservative
  standard errors under non-trivial between-study heterogeneity.
• 95% CI from Student's t (df = k − 1).
• Heterogeneity: Cochran's Q, its p-value, and I² (Higgins & Thompson, 2002).

Publication tiers (only PRE / PRE2014 epochs, primary triggers):
       TIER1_PUBLISH   : p_meta < 0.005  AND  k_cities ≥ 3  AND  I² < 40%
       TIER2_PROMISING : p_meta < 0.010  AND  k_cities ≥ 3  AND  I² < 60%
       TIER3_SIGNAL    : p_meta < 0.050  AND  k_cities ≥ 2
       NULL            : otherwise

Conservative interpretation note
--------------------------------
WAVE_* effects are estimated on Branch-B residuals (with static lunar-phase
dummies partialled out). Because impulsive wave events fall within ±2 d of
phase, the phase dummies absorb some of the wave variance — therefore WAVE_*
estimates here are LOWER BOUNDS on the true signal. PHASE_* effects are
estimated on Branch-A residuals (no lunar dummies) and represent the pure
phase contrast.

Outputs
-------
../outputs/verdicts/sgr_meta_family.csv      (one row per family×trigger×epoch×lag)
../outputs/verdicts/sgr_meta_crimetype.csv   (diagnostic, per individual crime)
../outputs/verdicts/sgr_forest_data.csv      (raw inputs for each meta-row)

References
----------
DerSimonian & Laird (1986); Hartung & Knapp (2001); Sidik & Jonkman (2002);
IntHout, Ioannidis & Borm (2014); Higgins & Thompson (2002).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════════════════════════════════
CORRELATIONS_DIR = Path("../outputs/correlations")
VERDICTS_DIR     = Path("../outputs/verdicts")
VERDICTS_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# SEMANTIC MAPPING
# ════════════════════════════════════════════════════════════════════════
SEMANTIC_MAP = {
    "AGGRESSION_TOTAL": {
        "Chicago": ["ASSAULT", "BATTERY", "HOMICIDE"],
        "Philly":  ["Aggravated Assault Firearm", "Aggravated Assault No Firearm",
                    "Homicide - Criminal"],
        "NYC":     ["FELONY ASSAULT", "ASSAULT 3 & RELATED OFFENSES",
                    "MURDER & NON-NEGL. MANSLAUGHTER"],
        "LA":      ["ASSAULT WITH DEADLY WEAPON AGGRAVATED ASSAULT",
                    "BATTERY - SIMPLE ASSAULT", "CRIMINAL HOMICIDE"],
        "SF":      ["ASSAULT"],
    },
    "SEXUAL_ASSAULT_CORE": {
        "Chicago": ["CRIM SEXUAL ASSAULT"],
        "Philly":  ["Rape"],
        "NYC":     ["RAPE"],
        "LA":      ["RAPE FORCIBLE"],
        "SF":      ["SEX OFFENSES FORCIBLE"],
    },
    "LIBIDO_BROAD": {
        "Chicago": ["CRIM SEXUAL ASSAULT", "SEX OFFENSE"],
        "Philly":  ["Rape", "Other Sex Offenses (Not Commercialized)"],
        "NYC":     ["RAPE", "SEX CRIMES"],
        "LA":      ["RAPE FORCIBLE", "RAPE ATTEMPTED",
                    "BATTERY WITH SEXUAL CONTACT", "ORAL COPULATION",
                    "SEXUAL PENETRATION W/FOREIGN OBJECT"],
        "SF":      ["SEX OFFENSES FORCIBLE"],
    },
    "PROSTITUTION_PROXY": {
        "Philly": ["Prostitution and Commercialized Vice"],
        "SF":     ["PROSTITUTION"],
        "NYC":    ["PROSTITUTION & RELATED OFFENSES"],
    },
    "NARCOTICS": {
        "Chicago": ["NARCOTICS"],
        "Philly":  ["Narcotic / Drug Law Violations"],
        "NYC":     ["DANGEROUS DRUGS"],
        "SF":      ["DRUG/NARCOTIC"],
    },
    "PROPERTY_CRIME": {
        "Chicago": ["THEFT", "BURGLARY", "CRIMINAL DAMAGE"],
        "Philly":  ["Thefts", "Burglary Residential", "Burglary Non-Residential",
                    "Vandalism/Criminal Mischief", "Theft from Vehicle"],
        "NYC":     ["PETIT LARCENY", "GRAND LARCENY", "BURGLARY",
                    "CRIMINAL MISCHIEF & RELATED OF"],
        "LA":      ["THEFT PLAIN - PETTY ($950 & UNDER)",
                    "THEFT-GRAND ($950.01 & OVER)EXCPTGUNSFOWLLIVESTKPROD",
                    "SHOPLIFTING - PETTY THEFT ($950 & UNDER)",
                    "BURGLARY", "BURGLARY FROM VEHICLE",
                    "VANDALISM - FELONY ($400 & OVER ALL CHURCH VANDALISMS)",
                    "VANDALISM - MISDEAMEANOR ($399 OR UNDER)"],
        "SF":      ["LARCENY/THEFT", "BURGLARY", "VANDALISM"],
    },
    "WEAPONS_STRESS": {
        "Chicago": ["WEAPONS VIOLATION"],
        "Philly":  ["Weapon Violations"],
        "NYC":     ["DANGEROUS WEAPONS"],
        "LA":      ["BRANDISH WEAPON"],
        "SF":      ["WEAPON LAWS"],
    },
    "DUI_EXTENDED": {
        "Philly": ["DRIVING UNDER THE INFLUENCE"],
        "NYC":    ["INTOXICATED & IMPAIRED DRIVING"],
        "SF":     ["DRIVING UNDER THE INFLUENCE"],
    },
    "DISORDERLY_PUBLIC": {
        "Philly": ["Disorderly Conduct"],
        "SF":     ["DISORDERLY CONDUCT"],
    },
    "PURE_BURGLARY": {
        "Chicago": ["BURGLARY"],
        "Philly":  ["Burglary Residential"],
        "NYC":     ["BURGLARY"],
        "LA":      ["BURGLARY"],
        "SF":      ["BURGLARY"],
    },
    "PURE_THEFT": {
        "Chicago": ["THEFT"],
        "Philly":  ["Thefts"],
        "NYC":     ["PETIT LARCENY"],
        "LA":      ["THEFT PLAIN - PETTY ($950 & UNDER)"],
        "SF":      ["LARCENY/THEFT"],
    },
    "PURE_VEHICLE_THEFT": {
        "Chicago": ["MOTOR VEHICLE THEFT"],
        "Philly":  ["Motor Vehicle Theft"],
        "NYC":     ["GRAND LARCENY OF MOTOR VEHICLE"],
        "LA":      ["VEHICLE - STOLEN"],
        "SF":      ["VEHICLE THEFT"],
    },
    "PURE_ASSAULT": {
        "Chicago": ["BATTERY"],
        "Philly":  ["Aggravated Assault No Firearm"],
        "NYC":     ["ASSAULT 3 & RELATED OFFENSES"],
        "LA":      ["BATTERY - SIMPLE ASSAULT"],
        "SF":      ["ASSAULT"],
    },
    "PURE_VANDALISM": {
        "Chicago": ["CRIMINAL DAMAGE"],
        "Philly":  ["Vandalism/Criminal Mischief"],
        "NYC":     ["CRIMINAL MISCHIEF & RELATED OF"],
        "LA":      ["VANDALISM - FELONY ($400 & OVER ALL CHURCH VANDALISMS)"],
        "SF":      ["VANDALISM"],
    },
    "PURE_ROBBERY": {
        "Chicago": ["ROBBERY"],
        "Philly":  ["Robbery No Firearm"],
        "NYC":     ["ROBBERY"],
        "LA":      ["ROBBERY"],
        "SF":      ["ROBBERY"],
    },
}

CITY_FROM_SEMANTIC = {
    "Chicago": "chicago", "LA": "la", "Philly": "philly",
    "NYC": "nyc", "SF": "sf",
}

# Strictly synchronized with correlator.py::TRIGGERS
PRIMARY_TRIGGERS = [
    "WAVE_event", "WAVE_syzygy", "WAVE_quadrature",
    "WAVE_before_syzygy", "WAVE_after_syzygy",
    "WAVE_before_quadrature", "WAVE_after_quadrature",
    "PHASE_Syzygy", "PHASE_Quadrature",
    "PHASE_Full", "PHASE_New", "PHASE_FirstQ", "PHASE_LastQ",
]

ENABLE_LAGS = True

# ════════════════════════════════════════════════════════════════════════
# RANDOM-EFFECTS META (REML τ² + HKSJ adjustment)
# ════════════════════════════════════════════════════════════════════════
def hksj_meta(y: np.ndarray, v: np.ndarray) -> dict | None:
    """Random-effects meta with iterative REML τ² and HKSJ SE adjustment.
    Returns None when k < 2 or weights collapse."""
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    valid = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[valid], v[valid]
    k = len(y)
    if k < 2:
        return None

    w0 = 1.0 / v
    w0_sum = w0.sum()
    mu0 = (w0 * y).sum() / w0_sum
    Q   = float((w0 * (y - mu0) ** 2).sum())
    df  = k - 1

    Cval = w0_sum - (w0 ** 2).sum() / w0_sum
    tau2 = max(0.0, (Q - df) / Cval) if Cval > 0 else 0.0

    for _ in range(200):
        w = 1.0 / (v + tau2)
        w_sum = w.sum()
        if w_sum <= 0:
            break
        mu = (w * y).sum() / w_sum
        num = (w ** 2 * ((y - mu) ** 2 - v + 1.0 / w_sum)).sum()
        den = (w ** 2).sum()
        tau2_new = max(0.0, num / den) if den > 0 else 0.0
        if abs(tau2_new - tau2) < 1e-8:
            tau2 = tau2_new
            break
        tau2 = tau2_new

    w = 1.0 / (v + tau2)
    w_sum = w.sum()
    if w_sum <= 0:
        return None
    mu = (w * y).sum() / w_sum

    q_hk  = max(1.0, (w * (y - mu) ** 2).sum() / df) if df > 0 else 1.0
    se_hk = float(np.sqrt(q_hk / w_sum))

    t_stat = mu / se_hk if se_hk > 0 else np.nan
    p_meta = 2.0 * float(stats.t.sf(abs(t_stat), df)) if df > 0 else np.nan
    t_crit = float(stats.t.ppf(0.975, df)) if df > 0 else np.nan

    I2  = max(0.0, (Q - df) / Q * 100.0) if Q > 0 else 0.0
    p_Q = float(stats.chi2.sf(Q, df)) if df > 0 else np.nan

    return {
        "k": int(k), "g_meta": float(mu), "se_hk": se_hk,
        "ci_lo": float(mu - t_crit * se_hk), "ci_hi": float(mu + t_crit * se_hk),
        "p_meta": p_meta, "tau2": float(tau2),
        "Q": Q, "p_Q": p_Q, "I2": I2,
    }

# ════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ════════════════════════════════════════════════════════════════════════
def load_all_results() -> pd.DataFrame:
    parts = []
    for code in set(CITY_FROM_SEMANTIC.values()):
        path = CORRELATIONS_DIR / f"sgr_{code}_results.csv"
        if not path.exists():
            print(f"  [warn] missing {path}")
            continue
        parts.append(pd.read_csv(path))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def map_families(df0: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, mapping in SEMANTIC_MAP.items():
        for sem_city, crimes in mapping.items():
            code = CITY_FROM_SEMANTIC.get(sem_city)
            if code is None:
                continue
            sub = df0[(df0["city"] == code) & (df0["crime_type"].isin(crimes))]
            for _, r in sub.iterrows():
                d = r.to_dict()
                d["family"] = fam
                rows.append(d)
    return pd.DataFrame(rows)


def assign_tier(meta: dict, k_cities: int) -> tuple[str, str]:
    p, I2 = meta["p_meta"], meta["I2"]
    if p < 0.005 and k_cities >= 3 and I2 < 40:
        return "TIER1_PUBLISH",   "Strong meta-signal, low heterogeneity"
    if p < 0.01  and k_cities >= 3 and I2 < 60:
        return "TIER2_PROMISING", "Significant meta-signal, moderate heterogeneity"
    if p < 0.05  and k_cities >= 2:
        return "TIER3_SIGNAL",    "p_meta < 0.05 across ≥ 2 cities"
    return "NULL", "p_meta ≥ 0.05 or k_cities < 2"


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    print(f"SGR Verdict Engine | Reading correlator outputs from {CORRELATIONS_DIR}", flush=True)
    df = load_all_results()
    if df.empty:
        print("No correlator results found. Aborting.")
        return

    if ENABLE_LAGS:
        df0 = df[df["status_flag"] == "OK"].copy()
        print("  [INFO] Lags ENABLED (processing -1, 0, +1, WIN3)")
    else:
        df0 = df[(df["status_flag"] == "OK") & (df["lag"].astype(str) == "0")].copy()
        print("  [INFO] Lags DISABLED (processing only lag = 0)")

    if df0.empty:
        print("No OK rows. Nothing to meta-analyze.")
        return

    print(f"  {len(df0)} valid rows | cities: {sorted(df0['city'].unique())}")

    df_fam = map_families(df0)
    print(f"  {len(df_fam)} family-mapped rows | families: {df_fam['family'].nunique()}")

    rows, forest = [], []
    for (family, trigger, epoch, lag), grp in df_fam.groupby(["family", "trigger", "epoch", "lag"]):
        meta = hksj_meta(grp["g"].values, grp["var_g"].values)
        if meta is None:
            continue

        is_pre     = epoch in ["PRE", "PRE2014"]
        is_primary = trigger in PRIMARY_TRIGGERS
        k_cities   = int(grp["city"].nunique())

        if is_pre and is_primary:
            tier, notes = assign_tier(meta, k_cities)
        else:
            tier, notes = "N/A", "Secondary trigger or non-PRE epoch"

        city_details_list = []
        for c, c_grp in grp.groupby('city'):
            c_grp_sorted = c_grp.sort_values('g', ascending=False)
            gs = [f"{row['g']:.3f}" for _, row in c_grp_sorted.iterrows()]
            max_g = c_grp_sorted['g'].max()
            city_details_list.append((max_g, f"{c.upper()}[{', '.join(gs)}]"))
        city_details_list.sort(key=lambda x: x[0], reverse=True)
        city_str = " ".join(x[1] for x in city_details_list)

        rows.append({
            "family": family, "trigger": trigger, "lag": lag,
            "trigger_family_class": grp["trigger_family"].iloc[0],
            "epoch": epoch, "is_primary": bool(is_primary),
            "k_total": meta["k"], "k_cities": k_cities,
            "g_meta": meta["g_meta"], "ci_lo": meta["ci_lo"], "ci_hi": meta["ci_hi"],
            "se_hk": meta["se_hk"], "p_meta": meta["p_meta"],
            "tau2": meta["tau2"], "Q": meta["Q"], "p_Q": meta["p_Q"], "I2": meta["I2"],
            "TIER": tier, "tier_notes": notes, "city_details": city_str,
        })

        for _, r in grp.iterrows():
            forest.append({
                "family": family, "trigger": trigger, "epoch": epoch, "lag": lag,
                "city": r["city"], "crime_type": r["crime_type"],
                "g": r["g"], "var_g": r["var_g"],
                "n_target": r["n_target"], "n_ctrl": r["n_ctrl"],
                "p_mw": r["p_mw"], "p_mw_fdr": r.get("p_mw_fdr"),
            })

    out_fam = pd.DataFrame(rows)

    cm_rows = []
    for (crime, trigger, tfam, epoch, lag), grp in df0.groupby(
            ["crime_type", "trigger", "trigger_family", "epoch", "lag"]):
        meta = hksj_meta(grp["g"].values, grp["var_g"].values)
        if meta is None:
            continue
        cm_rows.append({
            "crime_type": crime, "trigger": trigger, "lag": lag,
            "trigger_family_class": tfam, "epoch": epoch,
            "k_cities": int(grp["city"].nunique()),
            "g_meta": meta["g_meta"], "ci_lo": meta["ci_lo"], "ci_hi": meta["ci_hi"],
            "se_hk": meta["se_hk"], "p_meta": meta["p_meta"],
            "tau2": meta["tau2"], "Q": meta["Q"], "p_Q": meta["p_Q"], "I2": meta["I2"],
            "cities": ";".join(sorted(grp["city"].unique())),
        })

    out_cm     = pd.DataFrame(cm_rows)
    out_forest = pd.DataFrame(forest)

    out_fam.to_csv(VERDICTS_DIR / "sgr_meta_family.csv",    index=False)
    out_cm.to_csv (VERDICTS_DIR / "sgr_meta_crimetype.csv", index=False)
    out_forest.to_csv(VERDICTS_DIR / "sgr_forest_data.csv", index=False)

    print_summary(out_fam)


# ════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════════════
def _fmt(x, fmt="+.3f") -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "  nan"
        return format(x, fmt)
    except Exception:
        return str(x)


def print_summary(out_fam: pd.DataFrame):
    print("\n" + "═" * 115)
    print("FAMILY-LEVEL META — PRE & PRE2014 epochs, PRIMARY triggers only")
    print("═" * 115)

    sub = out_fam[out_fam["epoch"].isin(["PRE", "PRE2014"]) & out_fam["is_primary"]].copy()
    if sub.empty:
        print("  (no rows)")
        return

    sub["lag_sort"] = sub["lag"].astype(str)
    sub = sub.sort_values(["TIER", "lag_sort", "p_meta"]).drop(columns=["lag_sort"])

    tier_order = ["TIER1_PUBLISH", "TIER2_PROMISING", "TIER3_SIGNAL"]

    if os.name == 'nt':
        os.system('')
    GREEN, GRAY, RESET = '\033[92m', '\033[90m', '\033[0m'

    for tier in tier_order:
        chunk = sub[sub["TIER"] == tier]
        if chunk.empty:
            continue

        print(f"\n── {tier}  ({len(chunk)} rows)")
        print(f"  {'family':<20} {'epoch':<7} {'trigger':<22} {'lag':>5}  {'g_meta':>8}  "
              f"{'95%CI':>20}  {'p_meta':>10}  {'I2':>5}  {'k':>5}")

        for _, r in chunk.iterrows():
            ci_str  = f"[{_fmt(r['ci_lo'])},{_fmt(r['ci_hi'])}]"
            lag_val = str(r['lag'])
            print(f"{GREEN}  {r['family']:<20} {r['epoch']:<7} {r['trigger']:<22} {lag_val:>5}  "
                  f"{_fmt(r['g_meta']):>8}  {ci_str:>20}  "
                  f"{r['p_meta']:>10.2e}  {r['I2']:>4.0f}%  "
                  f"{int(r['k_total'])}({int(r['k_cities'])}c){RESET}")
            if "city_details" in r and pd.notna(r["city_details"]):
                print(f"{GRAY}      ↳ Contributions: {r['city_details']}{RESET}")

    print("\n" + "═" * 115)
    counts = sub["TIER"].value_counts()
    parts = [f"{t}={int(counts.get(t, 0))}" for t in tier_order + ["NULL"]]
    print("Summary:  " + " | ".join(parts))
    print(f"\nFiles written to {VERDICTS_DIR}/")
    print(f"  sgr_meta_family.csv     ({len(out_fam)} rows)")
    print(f"  sgr_meta_crimetype.csv")
    print(f"  sgr_forest_data.csv\n")


if __name__ == "__main__":
    main()