# -*- coding: utf-8 -*-
"""
File: gen.py
Author: Artem Tysiatskii
Python 3.10.11
Description: SGR - Lunar Gravitational Wave Generator.
Calculates local noon tidal forces (dF/dt, d2F/dt2), astronomical parameters, 
and detects localized wave events relative to syzygy and quadrature phases.
Directly generates highly accurate UTC-mapped data per city timezone.
"""

import datetime
import math
import os
import csv
from datetime import timezone, timedelta
from skyfield.api import load as sf_load
from skyfield import almanac

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────

EPH_PATH = "../data/external/de421.bsp"

OUT_DIR = "../data/waves"
os.makedirs(OUT_DIR, exist_ok=True)

CITIES = [
    {"name": "SF",          "tz": -8, "start": "2003-01-01", "end": "2017-12-31"},
    {"name": "Chicago",     "tz": -6, "start": "2001-01-01", "end": "2026-02-20"},
    {"name": "LA",          "tz": -8, "start": "2010-01-01", "end": "2022-12-31"},
    {"name": "Philly",      "tz": -5, "start": "2006-01-01", "end": "2026-02-26"},
    {"name": "NYC",         "tz": -5, "start": "2005-01-01", "end": "2025-12-31"},
]

# ──────────────────────────────────────────────────────────
# PHYSICS & CONSTANTS
# ──────────────────────────────────────────────────────────

MOON_MASS = 7.342e22
SUN_MASS = 1.989e30
NODAL_CYCLE_YEARS = 18.6134
APSIDAL_CYCLE_YEARS = 8.8501
EPOCH_J2000 = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
NODAL_MULT_RANGE = (0.86, 1.14)
NODAL_PEAK_YEARS = 25.3
APSIDAL_MULT_RANGE = (0.97, 1.03)

THR_BASE = 0.035
THR_STEP = 0.020
LOOKBACK_START = 4
COOLDOWN = 2

# ──────────────────────────────────────────────────────────
# ASTRONOMY ENGINE
# ──────────────────────────────────────────────────────────

def _nodal_modulation(dt):
    years = (dt - EPOCH_J2000).total_seconds() / (365.25 * 86400)
    mn, mx = NODAL_MULT_RANGE
    return (mn + mx) / 2.0 + (mx - mn) / 2.0 * math.cos(
        2 * math.pi * (years - NODAL_PEAK_YEARS) / NODAL_CYCLE_YEARS
    )

def _apsidal_modulation(dt):
    years = (dt - EPOCH_J2000).total_seconds() / (365.25 * 86400)
    _PERIGEE_J2000 = 83.3532
    _PERIHELION = 102.9372
    angle_rad = math.radians((_PERIGEE_J2000 - _PERIHELION) + 40.690 * years)
    mn, mx = APSIDAL_MULT_RANGE
    return (mn + mx) / 2.0 + (mx - mn) / 2.0 * math.cos(angle_rad)

class AstroGlobal:
    def __init__(self):
        self.eph = sf_load(EPH_PATH)
        self.ts = sf_load.timescale()
        self.earth = self.eph["earth"]
        self.moon = self.eph["moon"]
        self.sun = self.eph["sun"]
        self._cache = {}

    def _at(self, dt_utc):
        key = dt_utc.isoformat()
        if key in self._cache:
            return self._cache[key]
        
        t = self.ts.from_datetime(dt_utc)
        e = self.earth.at(t)
        
        r_moon_km = e.observe(self.moon).distance().km
        r_sun_km = e.observe(self.sun).distance().km
        angle_deg = e.observe(self.moon).apparent().separation_from(
            e.observe(self.sun).apparent()
        ).degrees
        _, dec_m, _ = e.observe(self.moon).radec()
        
        phase_f = (3.0 * math.cos(math.radians(angle_deg)) ** 2 - 1.0) / 2.0
        F_moon = MOON_MASS / ((r_moon_km * 1000.0) ** 3)
        F_sun = SUN_MASS / ((r_sun_km * 1000.0) ** 3)
        F_total = F_moon + F_sun * phase_f
        
        res = {
            "F": F_total,
            "r_moon_km": r_moon_km,
            "moon_dec": dec_m.degrees
        }
        self._cache[key] = res
        return res

    def get_point(self, dt_utc):
        c0 = self._at(dt_utc)
        c1 = self._at(dt_utc - timedelta(hours=1))
        c2 = self._at(dt_utc - timedelta(hours=2))
        
        dF_dt = c0["F"] - c1["F"]
        d2F_dt2 = (c0["F"] - c1["F"]) - (c1["F"] - c2["F"])
        
        apx = max(0.0, min(1.0, (c0["r_moon_km"] - 356500.0) / (406700.0 - 356500.0)))
        per = 1.0 - apx
        
        return {
            "dF_dt_micro": dF_dt * 1e6,
            "d2F_dt2_micro": d2F_dt2 * 1e6,
            "lunar_declination": c0["moon_dec"],
            "perigee_proximity": per,
            "nodal_modulation": _nodal_modulation(dt_utc),
            "apsidal_modulation": _apsidal_modulation(dt_utc)
        }

    def get_phases_for_range(self, start_utc, end_utc):
        t0 = self.ts.from_datetime(start_utc - timedelta(days=30))
        t1 = self.ts.from_datetime(end_utc + timedelta(days=30))
        times, phases = almanac.find_discrete(t0, t1, almanac.moon_phases(self.eph))
        
        phase_list = []
        phase_names = {0: "New Moon", 1: "First Quarter", 2: "Full Moon", 3: "Last Quarter"}
        for t, ph in zip(times, phases):
            dt_utc = t.utc_datetime()
            is_syzygy = (ph == 0 or ph == 2)
            phase_list.append({
                "utc_time": dt_utc,
                "type": "syzygy" if is_syzygy else "quadrature",
                "name": phase_names.get(ph, "")
            })
        return phase_list

# ──────────────────────────────────────────────────────────
# DETECTORS & LOCALIZERS
# ──────────────────────────────────────────────────────────

def get_nearest_phase(utc_time, phase_list, target_type=None):
    if target_type is not None:
        valid_phases = [p for p in phase_list if p["type"] == target_type]
    else:
        valid_phases = phase_list
        
    closest = min(valid_phases, key=lambda p: abs((p["utc_time"] - utc_time).total_seconds()))
    time_diff = (closest["utc_time"] - utc_time).total_seconds()
    pos = "before" if time_diff > 0 else "after"
    return closest["type"], pos

def detect_local_waves(data_rows, phase_list):
    n = len(data_rows)
    last_up = -999
    last_down = -999

    def is_down_start(k):
        if k < 2: return False
        x0, x1, x2 = data_rows[k-2]["d2F_dt2_micro"], data_rows[k-1]["d2F_dt2_micro"], data_rows[k]["d2F_dt2_micro"]
        return (x0 > -THR_BASE and x1 <= -THR_BASE and x2 <= x1 - THR_STEP)

    def is_up_start(k):
        if k < 2: return False
        x0, x1, x2 = data_rows[k-2]["d2F_dt2_micro"], data_rows[k-1]["d2F_dt2_micro"], data_rows[k]["d2F_dt2_micro"]
        return (x0 < THR_BASE and x1 >= THR_BASE and x2 >= x1 + THR_STEP)

    for i in range(2, n):
        row = data_rows[i]
        utc_time = row["_utc_time"]
        
        df0, df1, df2 = data_rows[i-2]["dF_dt_micro"], data_rows[i-1]["dF_dt_micro"], data_rows[i]["dF_dt_micro"]
        
        down_st = is_down_start(i)
        up_st = is_up_start(i)
        
        dF_falling_3 = (df0 > df1 > df2)
        dF_rising_3 = (df0 < df1 < df2)
        
        recent_down = any(is_down_start(k) for k in range(max(2, i - LOOKBACK_START), i + 1))
        recent_up = any(is_up_start(k) for k in range(max(2, i - LOOKBACK_START), i + 1))
        
        mark_down = down_st or (recent_down and dF_falling_3 and (i - last_down > COOLDOWN))
        mark_up = up_st or (recent_up and dF_rising_3 and (i - last_up > COOLDOWN))

        if mark_down and not mark_up:
            _, pos = get_nearest_phase(utc_time, phase_list, target_type="syzygy")
            row["wave_event"] = 1
            row["wave_syzygy"] = 1
            row[f"wave_{pos}_syzygy"] = 1
            last_down = i
            
        elif mark_up and not mark_down:
            _, pos = get_nearest_phase(utc_time, phase_list, target_type="quadrature")
            row["wave_event"] = 1
            row["wave_quadrature"] = 1
            row[f"wave_{pos}_quadrature"] = 1
            last_up = i

# ──────────────────────────────────────────────────────────
# MAIN GENERATOR
# ──────────────────────────────────────────────────────────

def main():
    print("Initialize Skyfield AstroGlobal...")
    astro = AstroGlobal()
    
    global_start = min(datetime.date.fromisoformat(c["start"]) for c in CITIES)
    global_end = max(datetime.date.fromisoformat(c["end"]) for c in CITIES)
    
    start_utc = datetime.datetime(global_start.year, global_start.month, global_start.day, tzinfo=timezone.utc)
    end_utc = datetime.datetime(global_end.year, global_end.month, global_end.day, tzinfo=timezone.utc)
    
    print("Loading Moon phases for global timeline...")
    phase_list = astro.get_phases_for_range(start_utc, end_utc)
    
    columns = [
        "date", "dF_dt_micro", "d2F_dt2_micro", "lunar_declination", "perigee_proximity",
        "nodal_modulation", "apsidal_modulation", "lunar_phase",
        "wave_event", "wave_syzygy", "wave_quadrature",
        "wave_before_syzygy", "wave_after_syzygy",
        "wave_before_quadrature", "wave_after_quadrature"
    ]

    for c in CITIES:
        city = c["name"]
        tz_offset = c["tz"]
        c_start = datetime.date.fromisoformat(c["start"])
        c_end = datetime.date.fromisoformat(c["end"])
        
        print(f"Generating localized timeline for {city} (UTC{tz_offset:+} 12:00)...")
        
        city_tz = timezone(timedelta(hours=tz_offset))
        curr_date = c_start
        city_data = []
        
        while curr_date <= c_end:
            local_noon = datetime.datetime(curr_date.year, curr_date.month, curr_date.day, 12, 0, 0, tzinfo=city_tz)
            utc_noon = local_noon.astimezone(timezone.utc)
            
            p = astro.get_point(utc_noon)
            
            local_phase_name = ""
            for phase_info in phase_list:
                phase_local = phase_info["utc_time"].astimezone(city_tz)
                if phase_local.date() == curr_date:
                    local_phase_name = phase_info["name"]
                    break
            
            row = {
                "date": curr_date.isoformat(),
                "_utc_time": utc_noon,
                "dF_dt_micro": round(p["dF_dt_micro"], 3),
                "d2F_dt2_micro": round(p["d2F_dt2_micro"], 4),
                "lunar_declination": round(p["lunar_declination"], 3),
                "perigee_proximity": round(p["perigee_proximity"], 4),
                "nodal_modulation": round(p["nodal_modulation"], 4),
                "apsidal_modulation": round(p["apsidal_modulation"], 4),
                "lunar_phase": local_phase_name,
                "wave_event": "", "wave_syzygy": "", "wave_quadrature": "",
                "wave_before_syzygy": "", "wave_after_syzygy": "",
                "wave_before_quadrature": "", "wave_after_quadrature": ""
            }
            city_data.append(row)
            curr_date += timedelta(days=1)
            
        detect_local_waves(city_data, phase_list)
        
        for row in city_data:
            del row["_utc_time"]
            
        out_file = os.path.join(OUT_DIR, f"waves_{city.lower()}_{c_start.year}_{c_end.year}.csv")
        with open(out_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(city_data)
            
        print(f"[{city}] Created {len(city_data)} points -> {out_file}")

if __name__ == "__main__":
    main()