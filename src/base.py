# -*- coding: utf-8 -*-
"""
File: base.py — Raw crime / EMS / 911-call preprocessor for the SGR
(Seleno-Gravitational Rhythm) analytical pipeline.
Author : Artem Tysiatskii
Python : 3.10.11
Description: Reads heterogeneous raw incident CSVs (Chicago, Philadelphia, Los Angeles,
NYC crime, NYC 911 calls, NYC EMS, San Francisco), deduplicates by record
ID across all parts, parses local datetimes (handling tz-aware and mixed
formats), restricts to a configured date window and aggregates into tidy
daily tables of shape  date × category → count.  Output files land in
../data/interim and are ready for downstream correlation / TIER analysis.
"""

import os, sys, glob, time
from collections import defaultdict
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ───────────────────────── PATHS & GLOBALS ─────────────────────────
RAW_DIR      = "../data/raw"
OUT_DIR      = "../data/interim"
CHUNK        = 500_000
MIN_PER_YEAR = 30      # drop categories with average yearly count below this

NULL_LABELS  = {"", "nan", "NaN", "(null)", "NULL", "None", "none", "<NA>"}

# ─────────────────────────── DATASETS ──────────────────────────────
# To add a new source: copy a block and adjust input / output / column
# names / date format / window.  date_fmt = None lets pandas auto-detect.
DATASETS = {
    "chicago": {
        "input":     "chicago_base.csv, chicago_base2.csv, chicago_base3.csv",
        "output":    "chicago_daily.csv",
        "date_col":  "date",
        "label_col": "primary_type",
        "dedup_col": "case_number",
        "date_fmt":  "ISO8601",
        "start":     "2001-01-01",
        "end":       "2026-02-20",
    },
    "philly": {
        "input":     "Philly_*.csv",
        "output":    "philly_daily.csv",
        "date_col":  "dispatch_date",
        "label_col": "text_general_code",
        "dedup_col": "dc_key",
        "date_fmt":  "%Y-%m-%d",
        "start":     "2006-01-01",
        "end":       "2026-02-26",
    },
    "la": {
        "input":     "LA_Raw_2010_2019.csv, LA_Raw_2020_plus.csv",
        "output":    "la_daily.csv",
        "date_col":  "DATE OCC",
        "label_col": "Crm Cd Desc",
        "dedup_col": "DR_NO",
        "date_fmt":  "%m/%d/%Y %I:%M:%S %p",
        "start":     "2010-01-01",
        "end":       "2022-12-31",
    },
    "nyc_crime": {
        "input":     "NYC_Raw.csv",
        "output":    "nyc_crime_daily.csv",
        "date_col":  "CMPLNT_FR_DT",
        "label_col": "OFNS_DESC",
        "dedup_col": "CMPLNT_NUM",
        "date_fmt":  "%m/%d/%Y",
        "start":     "2008-01-01",
        "end":       "2023-12-31",
    },
    "nyc_911": {
        "input":     "NYPD_Calls_for_Service__Historic_.csv, NYPD_Calls_for_Service__Year_to_Date_.csv",
        "output":    "nyc_911_daily.csv",
        "date_col":  "INCIDENT_DATE",
        "label_col": "TYP_DESC",
        "dedup_col": "CAD_EVNT_ID",
        "date_fmt":  "%m/%d/%Y",
        "start":     "2018-01-01",
        "end":       "2025-12-31",
    },
    "nyc_ems": {
        "input":     "EMS_Incident_Dispatch_Data_20260305.csv",
        "output":    "nyc_ems_daily.csv",
        "date_col":  "INCIDENT_DATETIME",
        "label_col": "FINAL_CALL_TYPE",
        "dedup_col": "CAD_INCIDENT_ID",
        "date_fmt":  "%m/%d/%Y %I:%M:%S %p",
        "start":     "2005-01-01",
        "end":       "2025-08-31",
    },
    "sf": {
        "input":     "SF_base_historical.csv",
        "output":    "sf_daily.csv",
        "date_col":  "Date",
        "label_col": "Category",
        "dedup_col": "PdId",
        "date_fmt":  "%m/%d/%Y",
        "start":     "2003-01-01",
        "end":       "2017-12-31",
    },
}

WARNING = (
    "  WARNING: this script is tailored for the seven specific raw sources\n"
    "  listed below.  Each one has its own schema (column names, datetime\n"
    "  format, dedup key) — it is NOT a generic CSV converter.  Download\n"
    "  the required raw files into ../data/raw with the exact filenames.\n"
    "  To add a new source, inspect its raw CSV and append an entry to the\n"
    "  DATASETS dict: input, output, date_col, label_col, dedup_col,\n"
    "  date_fmt, start, end.\n"
)

# ─────────────────────────── HELPERS ───────────────────────────────
def resolve_files(pattern_str, base_dir):
    out = []
    for p in pattern_str.split(","):
        p = p.strip()
        if not p:
            continue
        matched = sorted(glob.glob(os.path.join(base_dir, p)))
        out.extend(matched if matched else [os.path.join(base_dir, p)])
    return out


def files_status(cfg):
    files = resolve_files(cfg["input"], RAW_DIR)
    if not files:
        return False, []
    return all(os.path.isfile(f) for f in files), files


def parse_dates(series, fmt):
    if fmt:
        d = pd.to_datetime(series, format=fmt, errors="coerce")
    else:
        d = pd.to_datetime(series, errors="coerce")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    return d.dt.normalize()


# ─────────────────────────── CORE LOGIC ────────────────────────────
def process(key, cfg):
    files     = resolve_files(cfg["input"], RAW_DIR)
    date_col  = cfg["date_col"]
    label_col = cfg["label_col"]
    dedup_col = cfg.get("dedup_col")
    fmt       = cfg.get("date_fmt")
    d_min     = pd.Timestamp(cfg["start"])
    d_max     = pd.Timestamp(cfg["end"])
    usecols   = {date_col, label_col} | ({dedup_col} if dedup_col else set())

    counts = defaultdict(int)
    seen   = set()
    n_read = n_dup = n_bad = n_oor = 0
    t0 = time.time()

    for fpath in files:
        if not os.path.isfile(fpath):
            print(f"    ! missing : {os.path.basename(fpath)}")
            continue
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        fname   = os.path.basename(fpath)
        print(f"    > {fname}  ({size_mb:,.0f} MB)")

        reader = pd.read_csv(
            fpath,
            usecols=lambda c: c in usecols,
            chunksize=CHUNK,
            low_memory=False,
            dtype={dedup_col: "string"} if dedup_col else None,
            on_bad_lines="skip",
            encoding_errors="replace",
        )

        f_rows = 0
        for i, ch in enumerate(reader, 1):
            if date_col not in ch.columns or label_col not in ch.columns:
                raise KeyError(
                    f"required column missing in {fname}: "
                    f"need [{date_col}] and [{label_col}]")
            n = len(ch)
            f_rows += n
            n_read += n

            # 1) cross-file deduplication (BEFORE date parse — IDs are unique)
            if dedup_col and dedup_col in ch.columns:
                ids = (ch[dedup_col].astype("string").str.strip()
                                    .str.replace(r"\.0+$", "", regex=True))
                mask = ids.notna() & (ids != "") & ~ids.isin(seen)
                n_dup += int(n - mask.sum())
                ch    = ch.loc[mask]
                seen.update(ids[mask].tolist())

            if not len(ch):
                print(f"\r      chunk {i} | rows {f_rows:,}", end="", flush=True)
                continue

            # 2) date parsing + window filter
            dates = parse_dates(ch[date_col], fmt)
            bad   = dates.isna()
            n_bad += int(bad.sum())

            in_range = ~bad & (dates >= d_min) & (dates <= d_max)
            n_oor   += int((~bad & ~in_range).sum())
            if not in_range.any():
                print(f"\r      chunk {i} | rows {f_rows:,}", end="", flush=True)
                continue

            # 3) label clean-up
            labels = ch.loc[in_range, label_col].astype("string").str.strip()
            keep   = labels.notna() & ~labels.isin(NULL_LABELS)
            if not keep.any():
                print(f"\r      chunk {i} | rows {f_rows:,}", end="", flush=True)
                continue

            valid_idx = labels.index[keep]
            sub = pd.DataFrame({
                "date":  dates.loc[valid_idx].values,
                "label": labels.loc[valid_idx].values,
            })
            grp = sub.groupby(["date", "label"], sort=False).size()
            for k_, v in grp.items():
                counts[k_] += int(v)

            print(f"\r      chunk {i} | rows {f_rows:,}", end="", flush=True)

        print(f"\r      {fname}: {f_rows:,} rows processed" + " " * 24)

    if not counts:
        print("    ! no data after filtering")
        return

    # 4) pivot to daily table over the *configured* window
    df = pd.DataFrame(
        ((d, l, c) for (d, l), c in counts.items()),
        columns=["date", "label", "count"],
    )
    daily = df.pivot_table(index="date", columns="label", values="count",
                           aggfunc="sum", fill_value=0)
    daily.columns.name = None
    daily = daily.reindex(pd.date_range(d_min, d_max, freq="D"), fill_value=0)
    daily.index.name = "date"
    daily = daily.astype("int64")

    # 5) prune rare categories
    n_years = max(1.0, (d_max - d_min).days / 365.25)
    totals  = daily.sum()
    kept    = sorted(totals[totals / n_years >= MIN_PER_YEAR].index.tolist())
    dropped = sorted(set(daily.columns) - set(kept))
    daily   = daily[kept]
    daily.columns = [
        str(c).replace(",", "").replace(";", "")
              .replace("\n", " ").replace("\r", "").strip()
        for c in daily.columns
    ]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, cfg["output"])
    daily.to_csv(out_path, date_format="%Y-%m-%d")

    elapsed = time.time() - t0
    print(f"    rows read       : {n_read:,}")
    print(f"    duplicates      : {n_dup:,}")
    print(f"    bad dates       : {n_bad:,}")
    print(f"    outside window  : {n_oor:,}")
    print(f"    days            : {len(daily):,}  "
          f"({daily.index.min().date()} -> {daily.index.max().date()})")
    print(f"    categories kept : {len(kept)}   dropped : {len(dropped)}")
    if dropped:
        print(f"      dropped -> {', '.join(dropped[:10])}"
              + (" ..." if len(dropped) > 10 else ""))
    print(f"    saved -> {out_path}")
    print(f"    elapsed         : {elapsed:.1f}s")


# ─────────────────────────────  UI  ────────────────────────────────
def menu():
    print("=" * 72)
    print("  SGR :: raw dataset preprocessor")
    print("=" * 72)
    print(WARNING)
    print(f"  RAW dir : {os.path.abspath(RAW_DIR)}")
    print(f"  OUT dir : {os.path.abspath(OUT_DIR)}")
    print()
    keys = list(DATASETS.keys())
    print("  [0] ALL")
    for i, k in enumerate(keys, 1):
        ok, files = files_status(DATASETS[k])
        mark = "[+]" if ok else "[-]"
        suf  = f"{len(files)} file{'s' if len(files) != 1 else ''}" if files else "no match"
        print(f"  [{i}] {mark} {k:<10}  ({suf})  -> {DATASETS[k]['output']}")
    print()
    sel = input("  select dataset number: ").strip()
    if sel == "0":
        return keys
    if sel.isdigit():
        idx = int(sel) - 1
        if 0 <= idx < len(keys):
            return [keys[idx]]
    print("  invalid selection.")
    sys.exit(1)


def main():
    selected = menu()
    t_all = time.time()
    for k in selected:
        print("\n" + "-" * 72)
        print(f"  >>> {k}")
        print("-" * 72)
        try:
            process(k, DATASETS[k])
        except FileNotFoundError as e:
            print(f"    [ERROR] file not found: {e}")
        except KeyError as e:
            print(f"    [ERROR] schema mismatch: {e}")
        except Exception as e:
            print(f"    [ERROR] {type(e).__name__}: {e}")
    print("\n" + "=" * 72)
    print(f"  total elapsed: {time.time() - t_all:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()