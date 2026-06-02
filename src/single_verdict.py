# -*- coding: utf-8 -*-
"""
File: single_verdict.py
Author: Artem Tysiatskii
Python 3.10.11

================================================================================
SGR Single-Database Verdict — Internal Robustness Scanner & Tier Assignment
================================================================================

Purpose
-------
Consumes per-database outputs of single_correlator.py and assigns publication
tiers based on INTERNAL robustness only (no cross-cohort meta-analysis,
because each database is a unique data-generating process). Designed to be
read directly into a "Robustness" table for the manuscript.

Methodology
-----------
For each (database × crime_type × trigger × epoch × lag) row from the
correlator, the verdict aggregates four complementary robustness criteria:

  • Significance        : Mann-Whitney U two-sided p-value (p_mw).
  • Temporal robustness : block-permutation tests at block sizes 14 and
                          28 days + circular-shift permutation. The pass
                          count {0..3} measures resistance to autocorrelation
                          and seasonal alignment artefacts.
  • Stability           : slice sign-fraction over 5 contiguous time slices
                          (rejects accidental concentration in one era).
  • Placebo control     : empirical p-value from random subsets of the
                          control set (rejects "any-window-works" artefacts).

Publication tiers (PRE / PRE2014 epochs, primary triggers only):
       TIER1_PUBLISH   : p_mw < 0.005  AND  slice_sign_frac ≥ 0.80
                         AND 3/3 permutation tests pass  AND  placebo passes
       TIER2_PROMISING : p_mw < 0.010  AND  slice_sign_frac ≥ 0.75
                         AND ≥2/3 permutation tests pass AND  placebo passes
       TIER3_SIGNAL    : p_mw < 0.050  AND  slice_sign_frac ≥ 0.50
       DISCARDED_UNSTABLE : significant but slice-sign instability
       NULL            : otherwise

Conservative interpretation note
--------------------------------
WAVE_* effects are estimated on Branch-B residuals (with static lunar-phase
dummies partialled out). Because impulsive wave events fall within ±2 d of
phase, the phase dummies absorb some of the wave variance — therefore WAVE_*
estimates here are LOWER BOUNDS on the true signal. PHASE_* effects are
estimated on Branch-A residuals (no lunar dummies) and represent the pure
phase contrast.

Headline lag: WIN3
------------------
The 3-day exposure window WIN3 (union of {−1, 0, +1} days around each event)
is the timing-robust primary estimate. Individual integer lags {−1, 0, +1}
are reported alongside as a lag profile for biological plausibility.

Outputs
-------
../outputs/verdicts/sgr_single_verdict_signals.csv   (master, all rows)
../outputs/verdicts/sgr_<db>_single_results.csv      (per database)
"""
from __future__ import annotations

import os
import csv
from pathlib import Path
import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════════════════════════════════
CORRELATIONS_DIR = Path("../outputs/correlations")
VERDICTS_DIR     = Path("../outputs/verdicts")
VERDICTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# SEMANTIC MAP (TSV Loader)
# ==============================================================================
REPORT_LANG = "en" # "en" or "ru"
SEMANTIC_MAP_PATH = Path("../docs/semantic_map_single.tsv")

def _load_semantic_map(path: Path, lang: str) -> dict:
    if not path.exists():
        print(f"WARN: Semantic map file not found at {path}")
        return {}

    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = row["signal_key"]
            db = row["db_name"]
            code = row["code"]

            family = row["family_en"] if lang == "en" else row["family_ru"]
            label = row["label_en"] if lang == "en" else row["label_ru"]

            if db == "NYC_EMS":
                title = f"{row['emoji']} {family} | {code} — {label}"
            else:
                title = f"{row['emoji']} {family} | {label}"

            out[key] = {
                db: [code],
                "TITLE": title,
            }
    return out

SEMANTIC_MAP = _load_semantic_map(SEMANTIC_MAP_PATH, REPORT_LANG)

# Strictly synchronized with TRIGGERS in single_correlator.py
PRIMARY_TRIGGERS = [
    # WAVE — Branch B (controlled for static phase baseline)
    "WAVE_event", "WAVE_syzygy", "WAVE_quadrature",
    "WAVE_before_syzygy", "WAVE_after_syzygy",
    "WAVE_before_quadrature", "WAVE_after_quadrature",
    # PHASE — Branch A (pure phase contrast)
    "PHASE_Syzygy", "PHASE_Quadrature",
    "PHASE_Full", "PHASE_New", "PHASE_FirstQ", "PHASE_LastQ",
]

ENABLE_LAGS = True   # if False, only lag = "0" is considered


# ════════════════════════════════════════════════════════════════════════
# DATA LOADERS & MAPPERS
# ════════════════════════════════════════════════════════════════════════
def get_target_databases() -> list[str]:
    db_set = set()
    for _, cfg in SEMANTIC_MAP.items():
        for k in cfg.keys():
            if k != "TITLE":
                db_set.add(k.lower())
    return sorted(db_set)


def load_all_results() -> pd.DataFrame:
    parts = []
    dbs = get_target_databases()
    if not dbs:
        print("  [WARN] SEMANTIC_MAP is empty. Loading all sgr_*_results.csv files.")
        for path in CORRELATIONS_DIR.glob("sgr_*_results.csv"):
            parts.append(pd.read_csv(path))
    else:
        for code in dbs:
            path = CORRELATIONS_DIR / f"sgr_{code}_results.csv"
            if not path.exists():
                print(f"  [WARN] Missing {path}")
                continue
            parts.append(pd.read_csv(path))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def map_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["db_name"]      = df["city"].str.upper()
    df["signal_title"] = df["db_name"] + " | " + df["crime_type"]
    df["signal_key"]   = "RAW_" + df["crime_type"]

    for key, cfg in SEMANTIC_MAP.items():
        title = cfg.get("TITLE", key)
        for db_name, crimes in cfg.items():
            if db_name == "TITLE":
                continue
            city_code = db_name.lower()
            mask = (df["city"] == city_code) & (df["crime_type"].isin(crimes))
            df.loc[mask, "signal_key"]   = key
            df.loc[mask, "signal_title"] = title
            df.loc[mask, "db_name"]      = db_name
    return df


# ════════════════════════════════════════════════════════════════════════
# TIER ASSIGNMENT
# ════════════════════════════════════════════════════════════════════════
def assign_tier(r: pd.Series, is_primary: bool, is_target_epoch: bool) -> tuple[str, str]:
    p  = float(r.get("p_mw", 1.0))
    sf = float(r.get("slice_sign_frac", 0.0)) if pd.notna(r.get("slice_sign_frac")) else 0.0

    p14 = r.get("p_perm_14", np.nan)
    p28 = r.get("p_perm_28", np.nan)
    pcr = r.get("p_perm_circ", np.nan)
    perm_pass = sum(int(pd.notna(pv) and float(pv) < 0.05) for pv in (p14, p28, pcr))

    pl = r.get("p_placebo_emp", np.nan)
    plac_pass = bool(pd.notna(pl) and float(pl) < 0.05)

    if not (is_primary and is_target_epoch):
        return "N/A", "Secondary trigger or non-target epoch"

    if sf < 0.4:
        return "DISCARDED_UNSTABLE", f"Slice sign-frac {sf:.2f} < 0.40"

    if p < 0.005 and sf >= 0.8 and perm_pass == 3 and plac_pass:
        return "TIER1_PUBLISH", "p < 0.005, full robustness (3/3 perm, placebo OK)"

    if p < 0.01 and sf >= 0.75 and perm_pass >= 2 and plac_pass:
        return "TIER2_PROMISING", "p < 0.01, high robustness (>=2 perm, placebo OK)"

    if p < 0.05:
        if sf < 0.5:
            return "DISCARDED_UNSTABLE", f"Slice sign-frac {sf:.2f} < 0.50"
        return "TIER3_SIGNAL", "p < 0.05, partial robustness"

    return "NULL", "p_mw >= 0.05"


# ════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════
def main():
    print(f"SGR Single-DB Verdict Engine | Reading from {CORRELATIONS_DIR}", flush=True)
    df = load_all_results()
    if df.empty:
        print("No correlator results found. Aborting.")
        return

    if ENABLE_LAGS:
        df0 = df[df["status_flag"] == "OK"].copy()
        print("  [INFO] Lags ENABLED (all lags incl. WIN3)")
    else:
        df0 = df[(df["status_flag"] == "OK") & (df["lag"].astype(str) == "0")].copy()
        print("  [INFO] Lags DISABLED (lag = 0 only)")

    if df0.empty:
        print("No OK rows. Nothing to analyze.")
        return

    df_mapped = map_signals(df0)
    print(f"  {len(df_mapped)} valid rows | databases: "
          f"{sorted(df_mapped['db_name'].unique())}")

    results = []
    for _, r in df_mapped.iterrows():
        trigger = r["trigger"]
        epoch   = r["epoch"]
        is_pre  = epoch in ("PRE", "PRE2014")
        is_pri  = trigger in PRIMARY_TRIGGERS

        tier, notes = assign_tier(r, is_pri, is_pre)

        p14 = r.get("p_perm_14", np.nan)
        p28 = r.get("p_perm_28", np.nan)
        pcr = r.get("p_perm_circ", np.nan)
        perm_pass = sum(int(pd.notna(pv) and float(pv) < 0.05) for pv in (p14, p28, pcr))

        results.append({
            "db_name":              r["db_name"],
            "signal_title":         r["signal_title"],
            "crime_type":           r["crime_type"],
            "trigger":              trigger,
            "trigger_family_class": r.get("trigger_family", ""),
            "epoch":                epoch,
            "lag":                  r["lag"],
            "is_primary":           bool(is_pri),
            "n_target":             r.get("n_target", np.nan),
            "n_ctrl":               r.get("n_ctrl",   np.nan),
            "g":                    r.get("g",    np.nan),
            "ci_lo":                r.get("g_lo", np.nan),
            "ci_hi":                r.get("g_hi", np.nan),
            "p_mw":                 r.get("p_mw",     np.nan),
            "p_mw_fdr":             r.get("p_mw_fdr", np.nan),
            "delta_pct":            r.get("delta_pct", np.nan),
            "slice_sign_frac":      r.get("slice_sign_frac", np.nan),
            "perm_pass_count":      perm_pass,
            "p_placebo_emp":        r.get("p_placebo_emp", np.nan),
            "TIER":                 tier,
            "tier_notes":           notes,
        })

    out_df = pd.DataFrame(results)

    # Per-database outputs
    for db_name, grp in out_df.groupby("db_name"):
        out_path = VERDICTS_DIR / f"sgr_{db_name.lower()}_single_results.csv"
        grp.to_csv(out_path, index=False)

    # Master consolidated output
    global_out_path = VERDICTS_DIR / "sgr_single_verdict_signals.csv"
    out_df.to_csv(global_out_path, index=False)

    print_summary(out_df)


# ════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════════════
def _fmt(x, fmt="+.3f"):
    try:
        if pd.isna(x):
            return "  nan"
        return format(x, fmt)
    except Exception:
        return str(x)


def print_summary(df: pd.DataFrame):
    print("\n" + "═" * 140)
    print("SINGLE-DATABASE VERDICT  —  PRE & PRE2014 epochs, PRIMARY triggers only")
    print("═" * 140)

    sub = df[df["epoch"].isin(["PRE", "PRE2014"]) & df["is_primary"]].copy()
    if sub.empty:
        print("  (no rows)")
        return

    sub["lag_sort"] = sub["lag"].astype(str)
    sub = sub.sort_values(["TIER", "lag_sort", "p_mw"])
    sub = sub.drop(columns=["lag_sort"])

    tier_order = ["TIER1_PUBLISH", "TIER2_PROMISING", "TIER3_SIGNAL"]

    if os.name == 'nt':
        os.system('')
    GREEN = '\033[92m'
    RESET = '\033[0m'

    for tier in tier_order:
        chunk = sub[sub["TIER"] == tier]
        if chunk.empty:
            continue
        print(f"\n── {tier}  ({len(chunk)} rows)")
        print(f"  {'Signal':<65} {'Epoch':<7} {'Trigger':<22} {'Lag':>4}  "
              f"{'g':>8}  {'95% CI':>16}  {'p_MW':>8}  {'Sf':>4}  "
              f"{'Perm':>4} {'Plac':>4}")
        print("-" * 140)
        for _, r in chunk.iterrows():
            title_short = ((str(r['signal_title'])[:62] + '...')
                           if len(str(r['signal_title'])) > 65 else str(r['signal_title']))
            ci_str   = f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}]"
            plac_str = "YES" if (pd.notna(r['p_placebo_emp'])
                                 and r['p_placebo_emp'] < 0.05) else "NO"
            lag_val  = str(r['lag'])
            print(f"{GREEN}  {title_short:<65} {r['epoch']:<7} "
                  f"{r['trigger']:<22} {lag_val:>4}  "
                  f"{_fmt(r['g']):>8}  {ci_str:>16}  "
                  f"{r['p_mw']:>8.2e}  {_fmt(r['slice_sign_frac'], '.2f'):>4}  "
                  f"{r['perm_pass_count']:>2}/3  {plac_str:>4}{RESET}")

    discarded = sub[sub["TIER"].astype(str).str.startswith("DISCARDED")]
    if len(discarded):
        print(f"\n── DISCARDED  ({len(discarded)} rows)")
        for _, r in discarded.head(15).iterrows():
            title_short = ((str(r['signal_title'])[:55] + '..')
                           if len(str(r['signal_title'])) > 57
                           else str(r['signal_title']))
            print(f"  {title_short:<57} {r['epoch']:<7} {r['trigger']:<22} "
                  f"lag={str(r['lag']):>4}  {r['TIER']:<20}  {r['tier_notes']}")
        if len(discarded) > 15:
            print(f"  ... and {len(discarded) - 15} more discarded signals.")

    print("\n" + "═" * 140)
    counts = sub["TIER"].value_counts()
    parts = [f"{t}={int(counts.get(t, 0))}" for t in tier_order + ["NULL", "N/A"]]
    parts += [f"DISCARDED={int(sum(counts.get(t, 0) for t in counts.index if str(t).startswith('DISCARDED')))}"]
    print("Summary:  " + " | ".join(parts))
    print(f"\nFiles written to {VERDICTS_DIR}/")
    print(f"  sgr_single_verdict_signals.csv     ({len(df)} total rows)")
    print(f"  sgr_*_single_results.csv           (per database)\n")


if __name__ == "__main__":
    main()