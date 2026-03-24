"""
Main QC procedure for CoMeT 2024 data.
Replaces: comet_qc_2024.pro  +  comet_mergeqcd_createncdf_2024.pro

Workflow: run_qc() processes each file, writes a QC'd ASCII, then after all
files for a given day+vehicle are done it writes one combined NetCDF whose
format matches comet_ascii_2_cfnetcdf_torus_v2.pro exactly.

Error flag accumulation (bitwise):
    Level 1 → bit value 1
    Level 2 → bit value 2
    Level 3 → bit value 4
    Level 4 → bit value 8
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from .data_format import get_format
from .read_ascii import read_comet_ascii
from .read_raw import get_raw, ERRVAL
from .epoch import to_epoch, from_epoch, epoch_to_date_time_str
from .thermo import (sat_vap_pres, mixing_ratio, dewpoint,
                     potential_temp, virtual_potential_temp,
                     equivalent_potential_temp)
from .wind import wind_from_raw, find_winddir_offset
from .spike_filter import spike_filter
from .pressure_corr import pressure_correction
from .netcdf_writer import write_netcdf

# ---------------------------------------------------------------------------
# Header text written to every output ASCII file
# ---------------------------------------------------------------------------

HEADER_INTRO = [
    "# Combined Mesonet and Tracker (CoMeT-{comet}) synchronized data file",
    "# Data logger script version",
    "#   2024",
]
HEADER_INSTRUMENTATION = [
    "# Instrumentation",
    "#   Heading: KVH Industries C-100 fluxgate compass",
    "#   RH and slow temperature: Vaisala HMP155A-L-PT",
    "#   Fast temperature: Campbell Scientific 10922-L Thermistor",
    "#   Pressure: Vaisala PTB210 Barometer",
    "#   Wind velocity: RM Young Wind Monitor - 05103-L-PT",
]
HEADER_ERRFLAGS = [
    "# Error flags -- \"a#-g#-p#-tf#-ts#-rh#-w#-f#\"",
    "#   The number will accumulate bitwise using the following (where * corresponds to a parameter label)",
    "#    *1 --> 1, *2 --> 2, *3 --> 4, *4 --> 8",
    "#   a (All): 0-No error; 1-Exact correction, entire record reprocessed from raw data because missing",
    "#   g (GPS): 0-No error; 1-Exact correction, reprocessed from raw because of error; "
    "2-Exact correction, time corrected; 3-Exact correction, vehicle speed reprocessed from raw data",
    "#   p (Pressure): 0-No error; 1-Missing data, malfunctioning sensor, all dependent variables changed to missing; "
    "2-approximation, Gill port bias due to flow, all dependent variables recalculated",
    "#   tf (Tfast): 0-No error; 1-Missing data, malfunctioning sensor; 2-No correction, possible contamination from vehicle heat",
    "#   ts (Tslow): 0-No error; 1-Missing data, malfunctioning sensor; 2-No correction, possible contamination from vehicle heat",
    "#   rh (RH): 0-No error; 1-Missing data, malfunctioning sensor; 2-No correction, possible contamination from vehicle heat",
    "#   f (Fluxgate): 0-No error; 1-Missing data, sensor inoperable, vehicle heading replaced with GPS-based heading",
    "#   w (Wind): 0-No error; 1-Missing data, malfunctioning sensor; "
    "2-Approximate correction, reprocessed from raw because of error dir offset; "
    "3-Approximate correction, spikes in wind speed removed; "
    "4-No correction, possible errors due to vehicle acceleration",
]
HEADER_DATAKEY_23 = [
    "# Data key",
    "#   Date, Time (UTC HHMMSS), Latitude, Longitude, Altitude (Meters), Pressure (hPa), Temperature fast (Celsius),",
    "#   Temperature slow (Celsius), Logger RH, Calculated Corrected RH, Calculated Dewpoint (Celsius),",
    "#   Calculated Mixing Ratio (g/kg), Calculated Theta (Kelvin), Calculated Theta-V (Kelvin),",
    "#   Calculated Theta-E (Kelvin), Calculated Wind Speed (m/s), Calculated Wind Direction, Fluxgate Heading,",
    "#   Vehicle Speed (m/s), Computer Time (epoch), Logger Time (epoch), Error Flags",
]
HEADER_DATAKEY_1 = [
    "# Data key",
    "#   Date, Time (UTC HHMMSS), Latitude, Longitude, Altitude (Meters), Pressure (hPa), Temperature fast (Celsius),",
    "#   Temperature slow (Celsius), Logger RH, Calculated Corrected RH, Calculated Dewpoint (Celsius),",
    "#   Calculated Mixing Ratio (g/kg), Calculated Theta (Kelvin), Calculated Theta-V (Kelvin),",
    "#   Calculated Theta-E (Kelvin), Anemometer Speed (m/s), Anemometer Direction, Calculated Wind Speed (m/s), "
    "Calculated Wind Direction, Fluxgate Heading,",
    "#   Vehicle Speed (m/s), Number of Satellites, GPS Magnetic Variation, Estimated Horizontal Position Error (meters),",
    "#   Estimated Vertical Position Error (meters), Overall Spherical Equivalent Position Error, GPS Computer Time (epoch),",
    "#   Pressure Computer Time (epoch), THV Computer Time (epoch), Fluxgate Computer Time (epoch), Error Flags",
]

# Tag names order matches IDL tag_names(err_str_template) where template = {g,p,tf,ts,rh,f,w,a}
ERR_FIELDS = ('g', 'p', 'tf', 'ts', 'rh', 'f', 'w', 'a')


# ---------------------------------------------------------------------------
# Helper: build directory path from catalog row and template
# ---------------------------------------------------------------------------

def _build_dir(cat_row: pd.Series, comet_str: str, dir_template: str,
               dir_type: str) -> str:
    date_str = (f"{int(cat_row['year']):04d}"
                f"{int(cat_row['month']):02d}"
                f"{int(cat_row['day']):02d}")
    time_str = f"{int(cat_row['time']):04d}"

    directory = dir_template.replace('[project]', str(cat_row['project']))

    if int(time_str) < 600:
        yr, mo, dy = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:])
        ep   = to_epoch(yr, mo, dy, 0, 0, 0)
        yr2, mo2, dy2, *_ = from_epoch(ep - 3600)
        new_date = f"{yr2:04d}{mo2:02d}{dy2:02d}"
        directory = directory.replace('[YYYYMMDD]', new_date)
    else:
        directory = directory.replace('[YYYYMMDD]', date_str)

    directory = directory.replace('[*]', comet_str)
    directory = directory.replace('[type]', dir_type)

    site = str(cat_row.get('site', '')).strip()
    if site and site.lower() not in ('nan', ''):
        directory = directory.replace('CoMeT', f"{site}/CoMeT")

    return directory


# ---------------------------------------------------------------------------
# Helper: parse time window specification (replicates IDL find_window)
# ---------------------------------------------------------------------------

def _find_window(time_str: str, date_str: str,
                 epoch_arr: np.ndarray, quiet: bool = False) -> np.ndarray:
    nrec = len(epoch_arr)
    doit = np.zeros(nrec, dtype=bool)
    ts   = str(time_str).strip()

    if ts == 'no':
        return doit
    if ts == 'all':
        doit[:] = True
        return doit

    yr = int(date_str[0:4])
    mo = int(date_str[4:6])
    dy = int(date_str[6:8])

    for win in ts.split(';'):
        parts = win.strip().split('-')
        if len(parts) < 2:
            continue
        t0p = [float(x) for x in parts[0].strip().split(':')]
        t1p = [float(x) for x in parts[1].strip().split(':')]
        ep0 = to_epoch(yr, mo, dy, t0p[0], t0p[1], t0p[2] if len(t0p) > 2 else 0)
        ep1 = to_epoch(yr, mo, dy, t1p[0], t1p[1], t1p[2] if len(t1p) > 2 else 0)
        if ep0 < epoch_arr[0]: ep0 += 86400
        if ep1 < epoch_arr[0]: ep1 += 86400
        mask = (epoch_arr >= ep0) & (epoch_arr <= ep1)
        if not mask.any():
            print(f"  <!!!!> No times in QC window {parts[0]} – {parts[1]}")
        else:
            if not quiet:
                print(f"  <----> Applying correction between {parts[0]} and {parts[1]}")
            doit |= mask

    return doit


# ---------------------------------------------------------------------------
# Helper: sync raw/full time alignment
# ---------------------------------------------------------------------------

def _find_offset(raw_epoch: np.ndarray, full_epoch: np.ndarray,
                 full_time: pd.Series, full_date: pd.Series, idx: int) -> int:
    print(f"  <----> Full/raw times out of sync at {full_date.iloc[idx]} {full_time.iloc[idx]}")
    matches = np.where(raw_epoch == full_epoch[idx])[0]
    if len(matches) == 0:
        raise RuntimeError(f"No raw GPS time matches full epoch {full_epoch[idx]}")
    off = int(matches[0]) - idx
    print(f"  <----> Offset of {off} applied")
    return off


# ---------------------------------------------------------------------------
# Helper: vehicle heat contamination check
# ---------------------------------------------------------------------------

def _bad_wind_for_thermo(anem_spd: float, anem_dir: float,
                          anem_dir_off: float) -> bool:
    direction = anem_dir - anem_dir_off
    if direction < 0:
        direction += 360.0
    return (90.0 < direction < 270.0) or (anem_spd < 1.0)


# ---------------------------------------------------------------------------
# CRITICAL: Conditional recalculation — mirrors IDL logic exactly.
#
# IDL defines:
#   doit_p  = doit_p2           (True where P2 correction was applied)
#   doit_tf = bytarr(nrec)      (always zero — tf1 only sets to errval)
#   doit_ts = doit_ts3          (True where TS3 bias correction was applied)
#   doit_rh = bytarr(nrec)      (always zero — rh1 only sets to errval)
#   dont_p  = doit_p1 + errv_p  (True where pressure is missing)
#   dont_tf = doit_tf1 + errv_tf
#   dont_ts = doit_ts1 + errv_ts
#   dont_rh = doit_rh1 + errv_rh
#
# The recalculation is then:
#   theta:        recalc if P2 applied; errval if p or tf missing
#   rh_fast/td:   recalc if TS3 applied; errval if tf/ts/rh missing
#   qv/thv/the:   recalc if P2 or TS3 applied; errval if any relevant missing
#
# Records that were NOT affected by P2 or TS3 keep their ORIGINAL logger values.
# This is exactly why Python produced "very close but slightly different" results
# — it was unconditionally recalculating everything.
# ---------------------------------------------------------------------------

def _recalc_derived(df: pd.DataFrame,
                    doit_p2:  np.ndarray,
                    dont_p:   np.ndarray,
                    dont_tf:  np.ndarray,
                    doit_ts3: np.ndarray,
                    dont_ts:  np.ndarray,
                    dont_rh:  np.ndarray) -> pd.DataFrame:
    """
    Conditionally recalculate derived thermodynamic variables, replicating
    the IDL recalculation block in comet_qc_2024.pro exactly.

    Parameters mirror the IDL doit_* / dont_* arrays.
    - doit_p2:  True where P2 pressure correction was applied
    - dont_p:   True where pressure == errval (from p1 or pre-existing)
    - dont_tf:  True where temp_fast == errval
    - doit_ts3: True where TS3 bias correction was applied
    - dont_ts:  True where temp_slow == errval
    - dont_rh:  True where rh_slow == errval
    """
    errv = ERRVAL
    nrec = len(df)

    # doit_tf and doit_rh are always zero in IDL (tf1/rh1 only set to errval,
    # tf2/rh2 only flag — neither changes the value).
    doit_tf = np.zeros(nrec, dtype=bool)
    doit_rh = np.zeros(nrec, dtype=bool)

    for j in range(nrec):
        tf   = df.at[j, 'temperature_fast']
        ts   = df.at[j, 'temperature_slow']
        rhs  = df.at[j, 'rh_slow']
        pres = df.at[j, 'pressure']

        # ── theta ─────────────────────────────────────────────────────────
        # IDL: if ((doit_p[j]>0 or doit_tf[j]>0) and (dont_p[j]==0 or dont_tf[j]==0)):
        #          recalc theta
        #      if (dont_p[j]>0 or dont_tf[j]>0):
        #          theta = errval
        if (doit_p2[j] or doit_tf[j]) and (not dont_p[j] or not dont_tf[j]):
            df.at[j, 'theta'] = float(
                potential_temp(tf + 273.15, pres * 100.0, 0.0))
        if dont_p[j] or dont_tf[j]:
            df.at[j, 'theta'] = errv

        # ── rh_fast & dewpoint ────────────────────────────────────────────
        # IDL: if ((doit_tf>0 or doit_ts>0 or doit_rh>0) and (dont_tf==0 or dont_ts==0 or dont_rh==0)):
        #          recalc via sound_td / sound_satvappres('arm')
        #      if (dont_tf>0 or dont_ts>0 or dont_rh>0):
        #          rh_fast = dewpoint = errval
        if ((doit_tf[j] or doit_ts3[j] or doit_rh[j]) and
                (not dont_tf[j] or not dont_ts[j] or not dont_rh[j])):
            T_d_slow = float(dewpoint(rhs, ts))                          # °C
            e_hPa    = 0.01 * float(sat_vap_pres(T_d_slow + 273.15, 'arm'))  # hPa
            es_hPa   = 0.01 * float(sat_vap_pres(tf + 273.15,        'arm'))  # hPa
            rhf = 100.0 * e_hPa / es_hPa
            td  = float(dewpoint(rhf, tf))
            df.at[j, 'rh_fast']  = rhf
            df.at[j, 'dewpoint'] = td
        if dont_tf[j] or dont_ts[j] or dont_rh[j]:
            df.at[j, 'rh_fast']  = errv
            df.at[j, 'dewpoint'] = errv

        # ── qv, theta_v, theta_e ──────────────────────────────────────────
        # IDL: if ((doit_p>0 or doit_tf>0 or doit_ts>0 or doit_rh>0) and
        #          (dont_p==0 or dont_tf==0 or dont_ts==0 or dont_rh==0)):
        #          recalc
        #      if (dont_p>0 or dont_ts>0 or dont_rh>0 or dont_tf>0):
        #          qv = theta_v = theta_e = errval
        if ((doit_p2[j] or doit_tf[j] or doit_ts3[j] or doit_rh[j]) and
                (not dont_p[j] or not dont_tf[j] or not dont_ts[j] or not dont_rh[j])):
            td_j   = df.at[j, 'dewpoint']
            rhf_j  = df.at[j, 'rh_fast']
            theta_j = df.at[j, 'theta']
            # IDL: e_f = 0.01*sound_satvappres(273.15+dewpoint, 'arm')  ; hPa
            #      qv  = sound_mixr(pressure_hPa, e_f_hPa)              ; g/kg (units cancel)
            e_f_hPa = 0.01 * float(sat_vap_pres(td_j + 273.15, 'arm'))
            qv      = float(mixing_ratio(pres, e_f_hPa))                 # g/kg
            df.at[j, 'water_vapor_mixing_ratio'] = qv
            # IDL: theta_e = sound_thte(temp_f+273.15, 100*pressure, rh_fast)
            df.at[j, 'theta_e'] = float(
                equivalent_potential_temp(tf + 273.15, pres * 100.0, rhf_j))
            # IDL: theta_v = sound_thtv(theta, qv_g_per_kg)
            df.at[j, 'theta_v'] = float(virtual_potential_temp(theta_j, qv))
        if dont_p[j] or dont_ts[j] or dont_rh[j] or dont_tf[j]:
            df.at[j, 'water_vapor_mixing_ratio'] = errv
            df.at[j, 'theta_v']                  = errv
            df.at[j, 'theta_e']                  = errv

    return df


# ---------------------------------------------------------------------------
# Record formatting for ASCII output
# ---------------------------------------------------------------------------

def _format_record(row: pd.Series, frmt: pd.DataFrame, err_str: str) -> str:
    parts = []
    for _, frow in frmt.iterrows():
        name  = frow['parameter_name']
        ttype = str(frow['type']).lower().strip()
        val   = row.get(name, ERRVAL)

        if ttype == 'string':
            parts.append(str(val).strip())
        elif ttype in ('float', 'double'):
            fval = float(val) if str(val) not in ('', 'nan') else ERRVAL
            if name in ('latitude', 'longitude'):
                parts.append(f"{fval:11.6f}")
            elif ttype == 'double':
                parts.append(f"{fval:13.2f}")
            else:
                parts.append(f"{fval:7.2f}")
        elif ttype == 'long':
            try:
                parts.append(str(int(float(val))))
            except (ValueError, TypeError):
                parts.append(str(int(ERRVAL)))
        else:
            parts.append(str(val))

    parts.append(err_str)
    return ','.join(p.strip() for p in parts)


# ---------------------------------------------------------------------------
# Build error string for one record
# ---------------------------------------------------------------------------

def _build_err_str(erow: pd.Series) -> str:
    return '-'.join(f"{k}{int(erow[k]):02d}" for k in ERR_FIELDS)


# ---------------------------------------------------------------------------
# Main QC function
# ---------------------------------------------------------------------------

def run_qc(
    catalog_file: str,
    format_file: str,
    dir_template: str,
    project: str = 'all',
    day: str = 'all',
    time_filter: str = 'all',
    comet_filter: str = 'all',
    site_filter: str = 'all',
    nodump: bool = False,
    noplot: bool = True,
    skipqcd: bool = False,
) -> None:
    """
    Quality-control CoMeT data files and write combined NetCDF output.

    Replicates comet_qc_2024.pro + comet_mergeqcd_createncdf_2024.pro.

    After all files for a given day+vehicle are processed, a single combined
    NetCDF is written (format matches comet_ascii_2_cfnetcdf_torus_v2.pro).

    Args:
        catalog_file:  Path to comet_qc_2024.csv.
        format_file:   Path to CoMeT_data_format.csv.
        dir_template:  E.g. '/data/[project]/[YYYYMMDD]/CoMeT-[*]/[type]/'.
        project/day/time_filter/comet_filter/site_filter: Catalog filters.
        nodump:    Skip writing output files (dry run).
        noplot:    Skip plots (default True; plotting not yet ported).
        skipqcd:   Skip files already marked as QC'd without prompting.
    """
    errval = ERRVAL

    # -----------------------------------------------------------------------
    # Load and filter catalog
    # -----------------------------------------------------------------------
    cat = pd.read_csv(catalog_file)
    cat.columns = [c.strip().lower().replace(' ', '_') for c in cat.columns]

    flag_cols = [c for c in cat.columns if c.startswith('flag_')]
    flag_df   = cat[flag_cols].copy()

    cat['date_str'] = (cat['year'].astype(int).map(lambda y: f"{y:04d}") +
                       cat['month'].astype(int).map(lambda m: f"{m:02d}") +
                       cat['day'].astype(int).map(lambda d: f"{d:02d}"))
    cat['time_str'] = cat['time'].astype(int).map(lambda t: f"{t:04d}")

    mask = pd.Series(True, index=cat.index)
    for col, val in [('project', project), ('day', day), ('time', time_filter),
                     ('comet', comet_filter), ('site', site_filter)]:
        if val != 'all' and col in cat.columns:
            mask &= cat[col].astype(str).str.strip() == str(val).strip()

    selected = cat[mask].copy()
    if selected.empty:
        print("  <!> No matching entries in catalog.")
        return

    # Check already-QC'd
    if 'qc_txt' in selected.columns:
        already = selected['qc_txt'].astype(str).str.strip().isin(['1', 'True', 'true'])
        if skipqcd:
            selected = selected[~already]
        else:
            drop_idx = []
            for idx, row in selected[already].iterrows():
                ans = input(
                    f"  -> CoMeT {row['comet']} on {row['date_str']} at "
                    f"{row['time_str']} already QC'd. Proceed? [y/N] ")
                if ans.strip().upper() not in ('Y', 'YES'):
                    drop_idx.append(idx)
            selected = selected.drop(drop_idx)

    if selected.empty:
        print("  No files to process.")
        return

    # -----------------------------------------------------------------------
    # Group by (date_str, comet) — one NetCDF per day+vehicle
    # -----------------------------------------------------------------------
    selected = selected.sort_values(['date_str', 'comet', 'time_str']).reset_index(drop=True)

    # Use a grouping key; catalog index is preserved in selected
    group_keys = selected.groupby(['date_str', 'comet'], sort=False)

    for (grp_date, grp_comet), group_df in group_keys:
        comet_str  = str(grp_comet).strip()
        date_str   = str(grp_date)
        print(f"\n{'='*60}")
        print(f"  <> Day+Vehicle: CoMeT-{comet_str}  {date_str}")
        print(f"{'='*60}")

        # Collect processed DataFrames and union of active flags for this group
        group_processed: List[pd.DataFrame] = []
        group_active_flags: List[str]        = []
        first_time_str: Optional[str]        = None

        for _, cat_row in group_df.iterrows():
            # We need the original catalog index to index flag_df
            cat_idx   = cat_row.name   # pandas .name = original row index
            date_str  = str(cat_row['date_str'])
            time_str  = str(cat_row['time_str'])

            # Active flags for this file
            active_flags = []
            for fc in flag_cols:
                if str(flag_df.at[cat_idx, fc]).strip() != 'no':
                    active_flags.append(fc.replace('flag_', '').upper())

            print(f"\n  <> Processing CoMeT {comet_str} on {date_str} at {time_str}")
            print(f"  <--> Flags: {' '.join(active_flags) or 'none'}")

            # Build paths
            dir_orig  = _build_dir(cat_row, comet_str, dir_template, 'Original data')
            file_orig = Path(dir_orig) / f"CoMeT{comet_str}_full_{date_str}_{time_str}.txt"
            if not file_orig.exists():
                alt = Path(dir_orig) / f"IMeT{comet_str}_full_{date_str}_{time_str}.txt"
                print(f"  <> {file_orig.name} not found, trying {alt.name}")
                if not alt.exists():
                    print("  <> Neither found. Skipping.")
                    continue
                file_orig = alt

            fileformat = {'comet': comet_str, 'version': '2024', 'qc': 'orig'}
            try:
                get_format(fileformat, format_file)
            except ValueError as e:
                print(f"  <!> {e}")
                continue

            try:
                df = read_comet_ascii(str(file_orig), fileformat, format_file, quiet=True)
            except Exception as e:
                print(f"  <!> Error reading {file_orig}: {e}")
                continue

            nrec      = len(df)
            epoch_arr = df['epoch_time'].values.astype(float)

            # Error flags (one dict per record; matches IDL err_str_template field order)
            err_flags = pd.DataFrame(
                {k: np.zeros(nrec, dtype=int) for k in ERR_FIELDS},
                index=df.index)

            raw_loaded  = False
            raw_records = []
            gprmc_list  = []
            gpgga_list  = []

            def _ensure_raw():
                nonlocal raw_loaded, raw_records, gprmc_list, gpgga_list
                if not raw_loaded:
                    raw_records, gprmc_list, gpgga_list = get_raw(
                        comet_str, fileformat, str(file_orig), df)
                    raw_loaded = True

            # -------------------------------------------------------------------
            # Track which records were actually CHANGED (for conditional recalc)
            # These mirror IDL's doit_p2 and doit_ts3 arrays.
            # -------------------------------------------------------------------
            doit_p2_mask  = np.zeros(nrec, dtype=bool)
            doit_ts3_mask = np.zeros(nrec, dtype=bool)

            def _flag_val(name):
                return str(flag_df.at[cat_idx, name]).strip() if name in flag_df.columns else 'no'

            # ==============================================================
            # a1 – Fill gaps from raw data
            # ==============================================================
            dt_gap = 2
            gaps   = np.where(np.diff(epoch_arr) > dt_gap)[0]
            if len(gaps) > 0:
                print(f"  <--> {len(gaps)} data gap(s) found (>{dt_gap}s)")
                _ensure_raw()

            if len(gaps) > 0 and _flag_val('flag_a1') != 'no':
                print("  <--> A1 error correction")
                gprmc_times = np.array([r.time for r in gprmc_list])
                new_rows, new_errs = [], []

                for p0 in gaps:
                    t0, t1    = epoch_arr[p0], epoch_arr[p0 + 1]
                    fill_idx  = np.where((gprmc_times > t0) & (gprmc_times < t1))[0]
                    if not len(fill_idx):
                        continue
                    print(f"  <----> Filling {len(fill_idx)} record(s) at ~epoch {t0:.0f}")
                    for fi in fill_idx:
                        g  = gprmc_list[fi]
                        ga = gpgga_list[fi]  if fi < len(gpgga_list)  else None
                        rr = raw_records[fi] if fi < len(raw_records) else None
                        new_row = df.iloc[p0].copy()
                        new_row['date'], new_row['time'] = epoch_to_date_time_str(g.time)
                        new_row['epoch_time']   = g.time
                        new_row['latitude']     = g.latitude
                        new_row['longitude']    = g.longitude
                        new_row['altitude']     = ga.alt if ga else ERRVAL
                        if rr:
                            new_row['pressure']         = rr.pres
                            new_row['temperature_fast'] = rr.temp_f
                            new_row['temperature_slow'] = rr.temp_s
                            new_row['rh_slow']          = rr.rh_s
                            new_row['rh_fast']          = rr.rh_f
                            new_row['dewpoint']         = rr.dewpt
                            if rr.pres != ERRVAL and rr.temp_f != ERRVAL:
                                new_row['water_vapor_mixing_ratio'] = rr.dewpt  # placeholder; recalc below
                                new_row['theta']   = float(potential_temp(rr.temp_f+273.15, rr.pres*100.0, 0.0))
                                new_row['theta_e'] = float(equivalent_potential_temp(rr.temp_f+273.15, rr.pres*100.0, rr.rh_f))
                                new_row['theta_v'] = float(virtual_potential_temp(new_row['theta'], rr.rh_f))
                            new_row['wind_speed']      = rr.wndspd
                            new_row['wind_direction']  = rr.wnddir
                            new_row['vehicle_heading'] = rr.fluxdir
                            new_row['vehicle_speed']   = g.spd * 0.5144 if g.spd != ERRVAL else ERRVAL
                            if 'computer_time' in new_row.index:
                                new_row['computer_time'] = rr.compt1
                        new_rows.append(new_row)
                        new_errs.append({k: 0 for k in ERR_FIELDS})
                        new_errs[-1]['a'] = 1

                if new_rows:
                    df        = pd.concat([df, pd.DataFrame(new_rows)]).sort_values(
                        'epoch_time').reset_index(drop=True)
                    err_flags = pd.concat([err_flags, pd.DataFrame(new_errs)]).reset_index(drop=True)
                    epoch_arr = df['epoch_time'].values.astype(float)
                    nrec      = len(df)
                    # Resize doit masks
                    doit_p2_mask  = np.zeros(nrec, dtype=bool)
                    doit_ts3_mask = np.zeros(nrec, dtype=bool)

            # ==============================================================
            # f1 – Inoperable fluxgate: replace heading with GPS heading
            # ==============================================================
            fv_f1 = _flag_val('flag_f1')
            if fv_f1 != 'no':
                print("  <--> F1 error correction")
                _ensure_raw()
                doit = _find_window(fv_f1, date_str, epoch_arr)
                gprmc_times = np.array([r.time for r in gprmc_list])
                off = 0
                err_cnt = 0
                for j in range(nrec):
                    if j + off >= len(gprmc_list): break
                    if gprmc_times[j+off] != epoch_arr[j]:
                        off = _find_offset(gprmc_times, epoch_arr, df['time'], df['date'], j)
                    if doit[j]:
                        df.at[j, 'vehicle_heading'] = gprmc_list[j+off].track
                        err_flags.at[j, 'f'] |= 1
                        err_cnt += 1
                print(f"  <----> F1 flag set {err_cnt} times")

            # ==============================================================
            # g1 – GPS position error
            # ==============================================================
            fv_g1 = _flag_val('flag_g1')
            if fv_g1 != 'no':
                print("  <--> G1 error")
                _ensure_raw()
                doit = _find_window(fv_g1, date_str, epoch_arr)
                gprmc_times = np.array([r.time for r in gprmc_list])
                off = 0
                err_cnt = 0
                for j in range(nrec):
                    if j+off >= len(gprmc_list): break
                    if gprmc_times[j+off] != epoch_arr[j]:
                        off = _find_offset(gprmc_times, epoch_arr, df['time'], df['date'], j)
                    if doit[j]:
                        g = gprmc_list[j+off]
                        if df.at[j,'latitude'] != g.latitude or df.at[j,'longitude'] != g.longitude:
                            df.at[j,'latitude']  = g.latitude
                            df.at[j,'longitude'] = g.longitude
                            err_flags.at[j,'g'] |= 1
                            err_cnt += 1
                print(f"  <----> G1 flag set {err_cnt} times")

            # ==============================================================
            # g2 – GPS time warp
            # ==============================================================
            fv_g2 = _flag_val('flag_g2')
            print("  <--> Checking for time warps")
            j = 1
            while j <= nrec-2 and (epoch_arr[j] - epoch_arr[j-1]) > 0:
                j += 1
            if j < nrec and epoch_arr[j] - epoch_arr[j-1] < 0:
                jbad = j
                if fv_g2 != 'no':
                    print(f"  <--> G2 error correction at {df.at[j-1,'date']} {df.at[j-1,'time']}")
                    _ensure_raw()
                    toff = epoch_arr[jbad-1] - epoch_arr[jbad]
                    cnt  = 0
                    while j < nrec and (epoch_arr[j] - epoch_arr[j-1]) < 0:
                        epoch_arr[j] += toff
                        df.at[j,'epoch_time'] = epoch_arr[j]
                        ds, ts_val = epoch_to_date_time_str(epoch_arr[j])
                        df.at[j,'date'] = ds;  df.at[j,'time'] = ts_val
                        err_flags.at[j,'g'] += 2
                        j += 1;  cnt += 1
                    print(f"  <----> G2 flag set {cnt} times")
                else:
                    print(f"  <!!!!> TIME WARP at {df.at[j-1,'date']} {df.at[j-1,'time']} but G2 not set.")
            else:
                if fv_g2 != 'no':
                    print("  <!!!!> G2 set but no time warp detected.")

            # ==============================================================
            # g3 – Vehicle speed logged incorrectly
            # ==============================================================
            fv_g3 = _flag_val('flag_g3')
            if fv_g3 != 'no':
                print("  <--> G3 error")
                _ensure_raw()
                doit = _find_window(fv_g3, date_str, epoch_arr)
                gprmc_times = np.array([r.time for r in gprmc_list])
                off = 0
                err_cnt = 0
                for j in range(nrec):
                    if j+off >= len(gprmc_list): break
                    if gprmc_times[j+off] != epoch_arr[j]:
                        off = _find_offset(gprmc_times, epoch_arr, df['time'], df['date'], j)
                    if doit[j]:
                        df.at[j,'vehicle_speed'] = gprmc_list[j+off].spd * 0.514444
                        err_flags.at[j,'g'] |= 4
                        err_cnt += 1
                print(f"  <----> G3 flag set {err_cnt} times")

            # ==============================================================
            # p1 – Malfunctioning pressure sensor
            # ==============================================================
            fv_p1 = _flag_val('flag_p1')
            doit_p1_mask = np.zeros(nrec, dtype=bool)
            if fv_p1 != 'no':
                print("  <--> P1 correction")
                doit = _find_window(fv_p1, date_str, epoch_arr)
                df.loc[doit, 'pressure'] = errval
                err_flags.loc[doit, 'p'] |= 1
                doit_p1_mask = doit
                print(f"  <----> P1 flag set {int(doit.sum())} times")

            # ==============================================================
            # p2 – Gill pressure port flow bias correction
            # ==============================================================
            fv_p2 = _flag_val('flag_p2')
            if fv_p2 != 'no':
                print("  <--> P2 correction")
                _ensure_raw()
                gprmc_times = np.array([r.time for r in gprmc_list])
                off = 0
                err_cnt = 0
                for j in range(nrec):
                    if j+off >= len(raw_records): break
                    if gprmc_times[j+off] != epoch_arr[j]:
                        off = _find_offset(gprmc_times, epoch_arr, df['time'], df['date'], j)
                    rr = raw_records[j+off]
                    if df.at[j,'pressure'] != errval and rr.wndspdraw != errval:
                        df.at[j,'pressure'] = float(
                            pressure_correction(df.at[j,'pressure'], rr.wndspdraw, 'gill'))
                        err_flags.at[j,'p'] |= 2
                        doit_p2_mask[j] = True
                    err_cnt += 1
                print(f"  <----> P2 flag set {err_cnt} times")

            # ==============================================================
            # w1 – Malfunctioning wind monitor
            # ==============================================================
            fv_w1 = _flag_val('flag_w1')
            if fv_w1 != 'no':
                print("  <--> W1 correction")
                doit = _find_window(fv_w1, date_str, epoch_arr)
                for col in ('u','v','wind_speed','wind_direction'):
                    if col in df.columns:
                        df.loc[doit, col] = errval
                err_flags.loc[doit, 'w'] |= 1
                print(f"  <----> W1 flag set {int(doit.sum())} times")

            # ==============================================================
            # w2 – Wind direction offset correction
            # ==============================================================
            _ensure_raw()
            print("  <--> Checking wind direction offset")
            find_winddir_offset(gprmc_list, raw_records)

            fv_w2 = _flag_val('flag_w2')
            if fv_w2 != 'no':
                print("  <--> W2 correction")
                doit = _find_window(fv_w2, date_str, epoch_arr)
                w2_col   = 'w2_offset' if 'w2_offset' in cat.columns else None
                wnddiroff = float(cat_row[w2_col]) if w2_col else float(raw_records[0].wnddiroff)
                print(f"  <----> WNDDIROFF {raw_records[0].wnddiroff:.1f} → {wnddiroff:.1f}")
                gprmc_times = np.array([r.time for r in gprmc_list])
                off = 0
                err_cnt = 0
                for j in range(nrec):
                    if j+off >= len(raw_records): break
                    if gprmc_times[j+off] != epoch_arr[j]:
                        off = _find_offset(gprmc_times, epoch_arr, df['time'], df['date'], j)
                    if not doit[j]: continue
                    g  = gprmc_list[j+off];  rr = raw_records[j+off]
                    if any(v == errval for v in [g.track, g.spd, rr.wnddirraw, rr.wndspdraw]):
                        continue
                    ws, wd, u, v = wind_from_raw(g.track, g.spd, rr.wnddirraw, rr.wndspdraw, wnddiroff)
                    df.at[j,'wind_speed'] = ws;  df.at[j,'wind_direction'] = wd
                    df.at[j,'u'] = u;            df.at[j,'v'] = v
                    err_flags.at[j,'w'] += 2;    err_cnt += 1
                print(f"  <----> W2 flag set {err_cnt} times")

            # ==============================================================
            # w3 – Spike removal
            # ==============================================================
            fv_w3 = _flag_val('flag_w3')
            if fv_w3 != 'no':
                print("  <--> W3 correction")
                u_arr = df['u'].values.astype(float).copy()
                v_arr = df['v'].values.astype(float).copy()
                bad   = (u_arr == errval)
                u_arr[bad] = np.nan;  v_arr[bad] = np.nan
                uf = spike_filter(u_arr, hwidth=30, std_f=2)
                vf = spike_filter(v_arr, hwidth=30, std_f=2)
                spd_f = np.where(~np.isnan(uf) & ~np.isnan(vf), np.sqrt(uf**2+vf**2), errval)
                dir_f = np.where(~np.isnan(uf) & ~np.isnan(vf),
                                 np.degrees(np.arctan2(uf, vf)) + 180.0, errval)
                uf[bad] = errval;  vf[bad] = errval
                df['u'] = uf;  df['v'] = vf
                df['wind_speed'] = spd_f;  df['wind_direction'] = dir_f
                good_cnt = int((~bad).sum())
                err_flags.loc[~bad, 'w'] = err_flags.loc[~bad, 'w'] + 4
                print(f"  <----> W3 flag set {good_cnt} times")

            # ==============================================================
            # w4 – Vehicle acceleration flag
            # ==============================================================
            fv_w4 = _flag_val('flag_w4')
            if fv_w4 != 'no':
                print("  <--> W4 flagging")
                _ensure_raw()
                gprmc_times = np.array([r.time for r in gprmc_list])
                off = 0
                dir_old = spd_old = t_old = 0.0
                err_cnt = 0
                for j in range(nrec):
                    if j+off >= len(gprmc_list): break
                    if gprmc_times[j+off] != epoch_arr[j]:
                        off = _find_offset(gprmc_times, epoch_arr, df['time'], df['date'], j)
                    g = gprmc_list[j+off]
                    if g.track == errval or g.spd == errval: continue
                    dir_new = g.track
                    spd_new = g.spd * 0.514444
                    t_new   = g.time
                    dt = t_new - t_old
                    if dt > 0:
                        dd = min((dir_new-dir_old) % 360, (dir_old-dir_new) % 360) / dt
                        ds = abs(spd_new - spd_old) / dt
                        if dd > 2.0 or ds > 2.0:
                            err_flags.at[j,'w'] |= 8
                            err_cnt += 1
                    dir_old, spd_old, t_old = dir_new, spd_new, t_new
                print(f"  <----> W4 flag set {err_cnt} times")

            # ==============================================================
            # rh1 – Malfunctioning RH sensor
            # ==============================================================
            fv_rh1 = _flag_val('flag_rh1')
            doit_rh1_mask = np.zeros(nrec, dtype=bool)
            if fv_rh1 != 'no':
                print("  <--> RH1 correction")
                doit = _find_window(fv_rh1, date_str, epoch_arr)
                df.loc[doit, 'rh_slow'] = errval
                err_flags.loc[doit, 'rh'] |= 1
                doit_rh1_mask = doit
                print(f"  <----> RH1 flag set {int(doit.sum())} times")

            # ==============================================================
            # tf1 – Malfunctioning fast temperature sensor
            # ==============================================================
            fv_tf1 = _flag_val('flag_tf1')
            doit_tf1_mask = np.zeros(nrec, dtype=bool)
            if fv_tf1 != 'no':
                print("  <--> TF1 correction")
                doit = _find_window(fv_tf1, date_str, epoch_arr)
                df.loc[doit, 'temperature_fast'] = errval
                err_flags.loc[doit, 'tf'] |= 1
                doit_tf1_mask = doit
                print(f"  <----> TF1 flag set {int(doit.sum())} times")

            # ==============================================================
            # ts1 – Malfunctioning slow temperature sensor
            # ==============================================================
            fv_ts1 = _flag_val('flag_ts1')
            doit_ts1_mask = np.zeros(nrec, dtype=bool)
            if fv_ts1 != 'no':
                print("  <--> TS1 correction")
                doit = _find_window(fv_ts1, date_str, epoch_arr)
                df.loc[doit, 'temperature_slow'] = errval
                err_flags.loc[doit, 'ts'] |= 1
                doit_ts1_mask = doit
                print(f"  <----> TS1 flag set {int(doit.sum())} times")

            # ==============================================================
            # rh2, tf2, ts2 – Heat contamination flags (flag only, no change)
            # ==============================================================
            fv_rh2 = _flag_val('flag_rh2')
            fv_tf2 = _flag_val('flag_tf2')
            fv_ts2 = _flag_val('flag_ts2')
            if any(f != 'no' for f in [fv_rh2, fv_tf2, fv_ts2]):
                print("  <--> RH2/TF2/TS2 flagging")
                _ensure_raw()
                gprmc_times = np.array([r.time for r in gprmc_list])
                off = 0
                err_cnt = 0
                for j in range(nrec):
                    if j+off >= len(raw_records): break
                    if gprmc_times[j+off] != epoch_arr[j]:
                        off = _find_offset(gprmc_times, epoch_arr, df['time'], df['date'], j)
                    rr = raw_records[j+off]
                    if rr.wndspdraw != errval and rr.wnddirraw != errval:
                        if _bad_wind_for_thermo(rr.wndspdraw, rr.wnddirraw, rr.wnddiroff):
                            err_flags.at[j,'rh'] |= 2
                            err_flags.at[j,'tf'] |= 2
                            err_flags.at[j,'ts'] |= 2
                            err_cnt += 1
                print(f"  <----> RH2/TF2/TS2 flag set {err_cnt} times")

            # ==============================================================
            # ts3 – Slow temperature bias correction (+0.3 °C)
            # ==============================================================
            fv_ts3 = _flag_val('flag_ts3')
            if fv_ts3 != 'no':
                print("  <--> TS3 correction")
                doit = _find_window(fv_ts3, date_str, epoch_arr)
                valid = doit & (df['temperature_slow'] != errval)
                df.loc[valid, 'temperature_slow'] += 0.3
                err_flags.loc[valid, 'ts'] |= 4
                doit_ts3_mask[valid] = True
                print(f"  <----> TS3 flag set {int(valid.sum())} times")

            # ==============================================================
            # Conditional recalculation of derived variables
            # Replicates the IDL recalculation block EXACTLY:
            #   - Only recalculate where P2 or TS3 actually changed values
            #   - Set errval where source measurements are missing
            #   - All other records keep their original logger-computed values
            # ==============================================================
            errv_p  = (df['pressure'].values         == errval)
            errv_tf = (df['temperature_fast'].values  == errval)
            errv_ts = (df['temperature_slow'].values  == errval)
            errv_rh = (df['rh_slow'].values           == errval)

            dont_p  = doit_p1_mask  | errv_p
            dont_tf = doit_tf1_mask | errv_tf
            dont_ts = doit_ts1_mask | errv_ts
            dont_rh = doit_rh1_mask | errv_rh

            df = _recalc_derived(df, doit_p2_mask, dont_p, dont_tf,
                                 doit_ts3_mask, dont_ts, dont_rh)

            # Summary statistics (mirrors IDL output)
            for vname in ('theta', 'theta_e', 'theta_v', 'water_vapor_mixing_ratio',
                          'dewpoint', 'rh_fast'):
                if vname in df.columns:
                    valid = df[vname][df[vname] != errval]
                    pct   = 100.0 * (nrec - len(valid)) / nrec
                    mean  = valid.mean() if len(valid) else float('nan')
                    print(f"  <----> {vname}: mean={mean:.3f}, %missing={pct:.1f}%")

            # Sanity check
            bad_te = df[(df['theta_e'] < 0) & (df['theta_e'] != errval)]
            bad_ws = df[(df['wind_speed'] < 0) & (df['wind_speed'] != errval)]
            if len(bad_te): print(f"  <!!> Bad theta_e at {len(bad_te)} points")
            if len(bad_ws): print(f"  <!!> Bad wind_speed at {len(bad_ws)} points")

            # ==============================================================
            # Write QC'd ASCII file
            # ==============================================================
            if not nodump:
                fileformat_qcd = {'comet': comet_str, 'version': '2024', 'qc': 'qcd'}
                try:
                    frmt_qcd = get_format(fileformat_qcd, format_file)
                except ValueError:
                    frmt_qcd = get_format(fileformat, format_file)

                err_str_filename = '_'.join(f.lower() for f in active_flags)
                dir_qcd  = _build_dir(cat_row, comet_str, dir_template, "QC'd data")
                out_name = (f"CoMeT{comet_str}_full_{date_str}_{time_str}"
                            f"_L2_{err_str_filename}.txt")
                out_path = Path(dir_qcd) / out_name
                Path(dir_qcd).mkdir(parents=True, exist_ok=True)

                print(f"  <--> Writing ASCII: {out_path.name}")
                with open(out_path, 'w') as fh:
                    for line in HEADER_INTRO:
                        fh.write(line.replace('{comet}', comet_str) + '\n')
                    for line in HEADER_INSTRUMENTATION:
                        fh.write(line + '\n')
                    fh.write(f"# Missing data value\n#   {int(errval)}\n")
                    for line in (HEADER_DATAKEY_1 if comet_str == '1' else HEADER_DATAKEY_23):
                        fh.write(line + '\n')
                    for line in HEADER_ERRFLAGS:
                        fh.write(line + '\n')

                    err_str_col = []
                    for j in range(nrec):
                        es = _build_err_str(err_flags.iloc[j])
                        err_str_col.append(es)
                        fh.write(_format_record(df.iloc[j], frmt_qcd, es) + '\n')

                # Store error strings on the df for NetCDF writing
                df['error_string'] = err_str_col

                # Update catalog
                if 'qc_txt' in cat.columns:
                    cat.at[cat_idx, 'qc_txt'] = 1
                    cat.to_csv(catalog_file, index=False)
                    print("  <--> Catalog updated")
            else:
                # Even in nodump mode, build error strings so NetCDF can be written
                df['error_string'] = [_build_err_str(err_flags.iloc[j]) for j in range(nrec)]

            # Collect for group-level NetCDF
            group_processed.append(df)
            for f in active_flags:
                if f not in group_active_flags:
                    group_active_flags.append(f)
            if first_time_str is None:
                first_time_str = time_str

        # -------------------------------------------------------------------
        # After all files in this day+vehicle group: write ONE combined NetCDF
        # -------------------------------------------------------------------
        if group_processed and not nodump:
            merged = pd.concat(group_processed, ignore_index=True).sort_values(
                'epoch_time').reset_index(drop=True)

            err_str_filename = '_'.join(f.lower() for f in group_active_flags)

            # Use directory of the first processed file for output location
            first_row = group_df.iloc[0]
            nc_dir    = _build_dir(first_row, comet_str, dir_template, "QC'd data")
            Path(nc_dir).mkdir(parents=True, exist_ok=True)

            nc_name = (f"UNL.CoMeT{comet_str}.{date_str}.{first_time_str}"
                       f".L2_2024.{err_str_filename.replace('_', '.')}.nc")
            nc_path = str(Path(nc_dir) / nc_name)

            if comet_str == 'alpha':
                institution = 'Central Michigan University'
                pi_comment  = 'PI Contact Info: Jason Keeler (keele1j@cmich.edu)'
            else:
                institution = 'University of Nebraska-Lincoln'
                pi_comment  = 'PI Contact Info: Adam Houston (ahouston2@unl.edu)'

            yr_s = date_str[0:4]; mo_s = date_str[4:6]; dy_s = date_str[6:8]
            global_attrs = {
                'title':       f"{yr_s}-{mo_s}-{dy_s} Combined Mesonet and Tracker synchronized data file",
                'source':      f"Combined Mesonet and Tracker {comet_str} (CoMeT-{comet_str})",
                'institution': institution,
                'comment':     pi_comment,
            }

            fileformat_nc = {'comet': comet_str, 'version': '2024', 'qc': 'qcd'}
            print(f"\n  <--> Writing combined NetCDF: {nc_name}")
            write_netcdf(merged, nc_path, global_attrs, fileformat_nc, errval=errval)

    print("\n  Done.")


# ===========================================================================
# run_qc_dir  –  simplified entry point used by the new CLI


# ===========================================================================
# run_qc_dir  –  simplified entry point used by "comet-qc process"
# ===========================================================================

import re as _re


def _parse_full_filename(fname: str):
    """
    Extract (comet_str, date_str, time_str) from a CoMeT full filename.

    Accepted patterns:
        CoMeT<id>_full_<YYYYMMDD>_<HHMM>.txt
        IMeT<id>_full_<YYYYMMDD>_<HHMM>.txt   (CoMeT-1 alias)

    Returns None if the filename does not match.
    """
    m = _re.match(
        r'^(?:CoMeT|IMeT)([^_]+)_full_(\d{8})_(\d{4})(?:_.*)?\.txt$',
        fname, _re.IGNORECASE)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)   # comet_id, YYYYMMDD, HHMM


def run_qc_dir(
    catalog_file: str,
    format_file: str,
    input_dir: str,
    output_dir: Optional[str] = None,
    nodump: bool = False,
    skipqcd: bool = False,
) -> None:
    """
    QC all CoMeT full files found in *input_dir* and write one merged
    NetCDF per date + vehicle.

    This is the simplified entry point called by ``comet-qc process``.
    It replaces the old two-command workflow (``qc`` then ``merge``) and
    the confusing ``--dir-template`` argument.

    Parameters
    ----------
    catalog_file:
        Path to comet_qc_2024.csv.
    format_file:
        Path to CoMeT_data_format.csv.
    input_dir:
        Directory containing the un-QC'd ``*_full_*.txt`` (and
        ``*_raw_*.txt``) files.  The raw files are expected in the same
        directory (they are found by replacing ``full`` with ``raw`` in
        the filename, exactly as the existing code does).
    output_dir:
        Where to write QC'd ASCII and NetCDF output.  Defaults to a
        sibling ``QC'd data`` folder next to *input_dir*.
    nodump:
        If True, run QC but skip all file writes (dry run).
    skipqcd:
        If True, silently skip files already marked as QC'd in the
        catalog instead of prompting.
    """
    errval    = ERRVAL
    in_path   = Path(input_dir).resolve()
    out_path  = Path(output_dir).resolve() if output_dir else in_path.parent / "QC'd data"

    # -----------------------------------------------------------------------
    # Discover full files
    # -----------------------------------------------------------------------
    candidates = sorted(
        list(in_path.glob('CoMeT*_full_*.txt')) +
        list(in_path.glob('IMeT*_full_*.txt'))
    )
    if not candidates:
        print(f"  <!> No CoMeT*_full_*.txt files found in {in_path}")
        return

    # -----------------------------------------------------------------------
    # Load catalog
    # -----------------------------------------------------------------------
    cat = pd.read_csv(catalog_file)
    cat.columns = [c.strip().lower().replace(' ', '_') for c in cat.columns]
    flag_cols = [c for c in cat.columns if c.startswith('flag_')]
    flag_df   = cat[flag_cols].copy()

    cat['date_str'] = (cat['year'].astype(int).map(lambda y: f"{y:04d}") +
                       cat['month'].astype(int).map(lambda m: f"{m:02d}") +
                       cat['day'].astype(int).map(lambda d: f"{d:02d}"))
    cat['time_str'] = cat['time'].astype(int).map(lambda t: f"{t:04d}")

    # Build lookup: (comet_str, date_str, time_str) -> catalog row index
    catalog_key: Dict[Tuple[str, str, str], int] = {}
    for idx, row in cat.iterrows():
        key = (str(row['comet']).strip(),
               str(row['date_str']).strip(),
               str(row['time_str']).strip())
        catalog_key[key] = idx

    # -----------------------------------------------------------------------
    # Match files → catalog rows
    # -----------------------------------------------------------------------
    matched: List[Tuple[Path, int]] = []
    for fp in candidates:
        parsed = _parse_full_filename(fp.name)
        if parsed is None:
            print(f"  <-> Skipping unrecognised filename: {fp.name}")
            continue
        comet_id, date_str, time_str = parsed
        key = (comet_id, date_str, time_str)
        if key not in catalog_key:
            print(f"  <!> {fp.name}: no catalog entry for "
                  f"CoMeT={comet_id} date={date_str} time={time_str} — skipping.")
            continue
        matched.append((fp, catalog_key[key]))

    if not matched:
        print("  <!> No files matched the catalog. Nothing to do.")
        return

    # -----------------------------------------------------------------------
    # Handle already-QC'd files
    # -----------------------------------------------------------------------
    if 'qc_txt' in cat.columns:
        keep: List[Tuple[Path, int]] = []
        for fp, cat_idx in matched:
            already = str(cat.at[cat_idx, 'qc_txt']).strip() in ('1', 'True', 'true')
            if not already:
                keep.append((fp, cat_idx))
                continue
            if skipqcd:
                print(f"  -> Skipping already-QC'd: {fp.name}")
                continue
            comet_id  = str(cat.at[cat_idx, 'comet']).strip()
            date_str_p = str(cat.at[cat_idx, 'date_str'])
            time_str_p = str(cat.at[cat_idx, 'time_str'])
            ans = input(
                f"  -> CoMeT {comet_id} on {date_str_p} at {time_str_p} "
                f"already QC'd. Proceed? [y/N] ")
            if ans.strip().upper() in ('Y', 'YES'):
                keep.append((fp, cat_idx))
        matched = keep

    if not matched:
        print("  No files to process.")
        return

    # -----------------------------------------------------------------------
    # Group by (date_str, comet_str) — one NetCDF per day + vehicle
    # -----------------------------------------------------------------------
    from collections import defaultdict
    groups: Dict[Tuple[str, str], List[Tuple[Path, int]]] = defaultdict(list)
    for fp, cat_idx in matched:
        comet_id = str(cat.at[cat_idx, 'comet']).strip()
        date_str  = str(cat.at[cat_idx, 'date_str'])
        groups[(date_str, comet_id)].append((fp, cat_idx))

    for (date_str, comet_str), file_list in sorted(groups.items()):
        print(f"\n{'='*60}")
        print(f"  <> Day+Vehicle: CoMeT-{comet_str}  {date_str}")
        print(f"{'='*60}")

        group_processed: List[pd.DataFrame] = []
        group_active_flags: List[str]        = []
        first_time_str: Optional[str]        = None

        # Sort files within group chronologically by filename
        for file_orig, cat_idx in sorted(file_list, key=lambda x: x[0].name):
            cat_row  = cat.loc[cat_idx]
            time_str = str(cat.at[cat_idx, 'time_str'])

            # Active flags for this file
            active_flags = [
                fc.replace('flag_', '').upper()
                for fc in flag_cols
                if str(flag_df.at[cat_idx, fc]).strip() != 'no'
            ]

            print(f"\n  <> Processing {file_orig.name}")
            print(f"  <--> Flags: {' '.join(active_flags) or 'none'}")

            fileformat = {'comet': comet_str, 'version': '2024', 'qc': 'orig'}
            try:
                get_format(fileformat, format_file)
            except ValueError as e:
                print(f"  <!> {e}")
                continue

            try:
                df = read_comet_ascii(str(file_orig), fileformat, format_file, quiet=True)
            except Exception as e:
                print(f"  <!> Error reading {file_orig}: {e}")
                continue

            nrec      = len(df)
            epoch_arr = df['epoch_time'].values.astype(float)

            err_flags = pd.DataFrame(
                {k: np.zeros(nrec, dtype=int) for k in ERR_FIELDS},
                index=df.index)

            raw_loaded  = False
            raw_records = []
            gprmc_list  = []
            gpgga_list  = []

            def _ensure_raw():
                nonlocal raw_loaded, raw_records, gprmc_list, gpgga_list
                if not raw_loaded:
                    raw_records, gprmc_list, gpgga_list = get_raw(
                        comet_str, fileformat, str(file_orig), df)
                    raw_loaded = True

            doit_p2_mask  = np.zeros(nrec, dtype=bool)
            doit_ts3_mask = np.zeros(nrec, dtype=bool)

            def _flag_val(name):
                return str(flag_df.at[cat_idx, name]).strip() if name in flag_df.columns else 'no'

            # ==============================================================
            # a1 – Fill gaps from raw data
            # ==============================================================
            dt_gap = 2
            gaps   = np.where(np.diff(epoch_arr) > dt_gap)[0]
            if len(gaps) > 0:
                print(f"  <--> {len(gaps)} data gap(s) found (>{dt_gap}s)")
                _ensure_raw()

            if len(gaps) > 0 and _flag_val('flag_a1') != 'no':
                print("  <--> A1 error correction")
                gprmc_times = np.array([r.time for r in gprmc_list])
                new_rows, new_errs = [], []

                for p0 in gaps:
                    t0, t1    = epoch_arr[p0], epoch_arr[p0 + 1]
                    fill_idx  = np.where((gprmc_times > t0) & (gprmc_times < t1))[0]
                    if not len(fill_idx):
                        continue
                    print(f"  <----> Filling {len(fill_idx)} record(s) at ~epoch {t0:.0f}")
                    for fi in fill_idx:
                        g  = gprmc_list[fi]
                        ga = gpgga_list[fi]  if fi < len(gpgga_list)  else None
                        rr = raw_records[fi] if fi < len(raw_records) else None
                        new_row = df.iloc[p0].copy()
                        new_row['date'], new_row['time'] = epoch_to_date_time_str(g.time)
                        new_row['epoch_time']   = g.time
                        new_row['latitude']     = g.latitude
                        new_row['longitude']    = g.longitude
                        new_row['altitude']     = ga.alt if ga else ERRVAL
                        if rr:
                            new_row['pressure']         = rr.pres
                            new_row['temperature_fast'] = rr.temp_f
                            new_row['temperature_slow'] = rr.temp_s
                            new_row['rh_slow']          = rr.rh_s
                            new_row['rh_fast']          = rr.rh_f
                            new_row['dewpoint']         = rr.dewpt
                            if rr.pres != ERRVAL and rr.temp_f != ERRVAL:
                                new_row['water_vapor_mixing_ratio'] = rr.dewpt
                                new_row['theta']   = float(potential_temp(rr.temp_f+273.15, rr.pres*100.0, 0.0))
                                new_row['theta_e'] = float(equivalent_potential_temp(rr.temp_f+273.15, rr.pres*100.0, rr.rh_f))
                                new_row['theta_v'] = float(virtual_potential_temp(new_row['theta'], rr.rh_f))
                            new_row['wind_speed']      = rr.wndspd
                            new_row['wind_direction']  = rr.wnddir
                            new_row['vehicle_heading'] = rr.fluxdir
                            new_row['vehicle_speed']   = g.spd * 0.5144 if g.spd != ERRVAL else ERRVAL
                            if 'computer_time' in new_row.index:
                                new_row['computer_time'] = rr.compt1
                        new_rows.append(new_row)
                        new_errs.append({k: 0 for k in ERR_FIELDS})
                        new_errs[-1]['a'] = 1

                if new_rows:
                    df        = pd.concat([df, pd.DataFrame(new_rows)]).sort_values(
                        'epoch_time').reset_index(drop=True)
                    err_flags = pd.concat([err_flags, pd.DataFrame(new_errs)]).reset_index(drop=True)
                    epoch_arr = df['epoch_time'].values.astype(float)
                    nrec      = len(df)
                    doit_p2_mask  = np.zeros(nrec, dtype=bool)
                    doit_ts3_mask = np.zeros(nrec, dtype=bool)

            # ==============================================================
            # f1 – Inoperable fluxgate: replace heading with GPS heading
            # ==============================================================
            fv_f1 = _flag_val('flag_f1')
            if fv_f1 != 'no':
                print("  <--> F1 error correction")
                _ensure_raw()
                doit = _find_window(fv_f1, date_str, epoch_arr)
                gprmc_times = np.array([r.time for r in gprmc_list])
                off = 0
                err_cnt = 0
                for j in range(nrec):
                    if j + off >= len(gprmc_list): break
                    if gprmc_times[j+off] != epoch_arr[j]:
                        off = _find_offset(gprmc_times, epoch_arr, df['time'], df['date'], j)
                    if doit[j]:
                        df.at[j, 'vehicle_heading'] = gprmc_list[j+off].track
                        err_flags.at[j, 'f'] |= 1
                        err_cnt += 1
                print(f"  <----> F1 flag set {err_cnt} times")

            # ==============================================================
            # g1 – GPS position error
            # ==============================================================
            fv_g1 = _flag_val('flag_g1')
            if fv_g1 != 'no':
                print("  <--> G1 error")
                _ensure_raw()
                doit = _find_window(fv_g1, date_str, epoch_arr)
                gprmc_times = np.array([r.time for r in gprmc_list])
                off = 0
                err_cnt = 0
                for j in range(nrec):
                    if j+off >= len(gprmc_list): break
                    if gprmc_times[j+off] != epoch_arr[j]:
                        off = _find_offset(gprmc_times, epoch_arr, df['time'], df['date'], j)
                    if doit[j]:
                        g = gprmc_list[j+off]
                        if df.at[j,'latitude'] != g.latitude or df.at[j,'longitude'] != g.longitude:
                            df.at[j,'latitude']  = g.latitude
                            df.at[j,'longitude'] = g.longitude
                            err_flags.at[j,'g'] |= 1
                            err_cnt += 1
                print(f"  <----> G1 flag set {err_cnt} times")

            # ==============================================================
            # g2 – GPS time warp
            # ==============================================================
            fv_g2 = _flag_val('flag_g2')
            print("  <--> Checking for time warps")
            j = 1
            while j <= nrec-2 and (epoch_arr[j] - epoch_arr[j-1]) > 0:
                j += 1
            if j < nrec and epoch_arr[j] - epoch_arr[j-1] < 0:
                if fv_g2 != 'no':
                    print(f"  <--> G2 error correction at {df.at[j-1,'date']} {df.at[j-1,'time']}")
                    _ensure_raw()
                    toff = epoch_arr[j-1] - epoch_arr[j]
                    cnt  = 0
                    while j < nrec and (epoch_arr[j] - epoch_arr[j-1]) < 0:
                        epoch_arr[j] += toff
                        df.at[j,'epoch_time'] = epoch_arr[j]
                        ds, ts_val = epoch_to_date_time_str(epoch_arr[j])
                        df.at[j,'date'] = ds;  df.at[j,'time'] = ts_val
                        err_flags.at[j,'g'] += 2
                        j += 1;  cnt += 1
                    print(f"  <----> G2 flag set {cnt} times")
                else:
                    print(f"  <!!!!> TIME WARP at {df.at[j-1,'date']} {df.at[j-1,'time']} but G2 not set.")
            else:
                if fv_g2 != 'no':
                    print("  <!!!!> G2 set but no time warp detected.")

            # ==============================================================
            # g3 – Vehicle speed logged incorrectly
            # ==============================================================
            fv_g3 = _flag_val('flag_g3')
            if fv_g3 != 'no':
                print("  <--> G3 error")
                _ensure_raw()
                doit = _find_window(fv_g3, date_str, epoch_arr)
                gprmc_times = np.array([r.time for r in gprmc_list])
                off = 0
                err_cnt = 0
                for j in range(nrec):
                    if j+off >= len(gprmc_list): break
                    if gprmc_times[j+off] != epoch_arr[j]:
                        off = _find_offset(gprmc_times, epoch_arr, df['time'], df['date'], j)
                    if doit[j]:
                        df.at[j,'vehicle_speed'] = gprmc_list[j+off].spd * 0.514444
                        err_flags.at[j,'g'] |= 4
                        err_cnt += 1
                print(f"  <----> G3 flag set {err_cnt} times")

            # ==============================================================
            # p1 – Malfunctioning pressure sensor
            # ==============================================================
            fv_p1 = _flag_val('flag_p1')
            doit_p1_mask = np.zeros(nrec, dtype=bool)
            if fv_p1 != 'no':
                print("  <--> P1 correction")
                doit = _find_window(fv_p1, date_str, epoch_arr)
                df.loc[doit, 'pressure'] = errval
                err_flags.loc[doit, 'p'] |= 1
                doit_p1_mask = doit
                print(f"  <----> P1 flag set {int(doit.sum())} times")

            # ==============================================================
            # p2 – Gill pressure port flow bias correction
            # ==============================================================
            fv_p2 = _flag_val('flag_p2')
            if fv_p2 != 'no':
                print("  <--> P2 correction")
                _ensure_raw()
                gprmc_times = np.array([r.time for r in gprmc_list])
                off = 0
                err_cnt = 0
                for j in range(nrec):
                    if j+off >= len(raw_records): break
                    if gprmc_times[j+off] != epoch_arr[j]:
                        off = _find_offset(gprmc_times, epoch_arr, df['time'], df['date'], j)
                    rr = raw_records[j+off]
                    if df.at[j,'pressure'] != errval and rr.wndspdraw != errval:
                        df.at[j,'pressure'] = float(
                            pressure_correction(df.at[j,'pressure'], rr.wndspdraw, 'gill'))
                        err_flags.at[j,'p'] |= 2
                        doit_p2_mask[j] = True
                    err_cnt += 1
                print(f"  <----> P2 flag set {err_cnt} times")

            # ==============================================================
            # w1 – Malfunctioning wind monitor
            # ==============================================================
            fv_w1 = _flag_val('flag_w1')
            if fv_w1 != 'no':
                print("  <--> W1 correction")
                doit = _find_window(fv_w1, date_str, epoch_arr)
                for col in ('u','v','wind_speed','wind_direction'):
                    if col in df.columns:
                        df.loc[doit, col] = errval
                err_flags.loc[doit, 'w'] |= 1
                print(f"  <----> W1 flag set {int(doit.sum())} times")

            # ==============================================================
            # w2 – Wind direction offset correction
            # ==============================================================
            _ensure_raw()
            print("  <--> Checking wind direction offset")
            find_winddir_offset(gprmc_list, raw_records)

            fv_w2 = _flag_val('flag_w2')
            if fv_w2 != 'no':
                print("  <--> W2 correction")
                doit = _find_window(fv_w2, date_str, epoch_arr)
                w2_col    = 'w2_offset' if 'w2_offset' in cat.columns else None
                wnddiroff = float(cat_row[w2_col]) if w2_col else float(raw_records[0].wnddiroff)
                print(f"  <----> WNDDIROFF {raw_records[0].wnddiroff:.1f} → {wnddiroff:.1f}")
                gprmc_times = np.array([r.time for r in gprmc_list])
                off = 0
                err_cnt = 0
                for j in range(nrec):
                    if j+off >= len(raw_records): break
                    if gprmc_times[j+off] != epoch_arr[j]:
                        off = _find_offset(gprmc_times, epoch_arr, df['time'], df['date'], j)
                    if not doit[j]: continue
                    g  = gprmc_list[j+off];  rr = raw_records[j+off]
                    if any(v == errval for v in [g.track, g.spd, rr.wnddirraw, rr.wndspdraw]):
                        continue
                    ws, wd, u, v = wind_from_raw(g.track, g.spd, rr.wnddirraw, rr.wndspdraw, wnddiroff)
                    df.at[j,'wind_speed'] = ws;  df.at[j,'wind_direction'] = wd
                    df.at[j,'u'] = u;            df.at[j,'v'] = v
                    err_flags.at[j,'w'] += 2;    err_cnt += 1
                print(f"  <----> W2 flag set {err_cnt} times")

            # ==============================================================
            # w3 – Spike removal
            # ==============================================================
            fv_w3 = _flag_val('flag_w3')
            if fv_w3 != 'no':
                print("  <--> W3 correction")
                u_arr = df['u'].values.astype(float).copy()
                v_arr = df['v'].values.astype(float).copy()
                bad   = (u_arr == errval)
                u_arr[bad] = np.nan;  v_arr[bad] = np.nan
                uf = spike_filter(u_arr, hwidth=30, std_f=2)
                vf = spike_filter(v_arr, hwidth=30, std_f=2)
                spd_f = np.where(~np.isnan(uf) & ~np.isnan(vf), np.sqrt(uf**2+vf**2), errval)
                dir_f = np.where(~np.isnan(uf) & ~np.isnan(vf),
                                 np.degrees(np.arctan2(uf, vf)) + 180.0, errval)
                uf[bad] = errval;  vf[bad] = errval
                df['u'] = uf;  df['v'] = vf
                df['wind_speed'] = spd_f;  df['wind_direction'] = dir_f
                good_cnt = int((~bad).sum())
                err_flags.loc[~bad, 'w'] = err_flags.loc[~bad, 'w'] + 4
                print(f"  <----> W3 flag set {good_cnt} times")

            # ==============================================================
            # w4 – Vehicle acceleration flag
            # ==============================================================
            fv_w4 = _flag_val('flag_w4')
            if fv_w4 != 'no':
                print("  <--> W4 flagging")
                _ensure_raw()
                gprmc_times = np.array([r.time for r in gprmc_list])
                off = 0
                dir_old = spd_old = t_old = 0.0
                err_cnt = 0
                for j in range(nrec):
                    if j+off >= len(gprmc_list): break
                    if gprmc_times[j+off] != epoch_arr[j]:
                        off = _find_offset(gprmc_times, epoch_arr, df['time'], df['date'], j)
                    g = gprmc_list[j+off]
                    if g.track == errval or g.spd == errval: continue
                    dir_new = g.track
                    spd_new = g.spd * 0.514444
                    t_new   = g.time
                    dt = t_new - t_old
                    if dt > 0:
                        dd = min((dir_new-dir_old) % 360, (dir_old-dir_new) % 360) / dt
                        ds = abs(spd_new - spd_old) / dt
                        if dd > 2.0 or ds > 2.0:
                            err_flags.at[j,'w'] |= 8
                            err_cnt += 1
                    dir_old, spd_old, t_old = dir_new, spd_new, t_new
                print(f"  <----> W4 flag set {err_cnt} times")

            # ==============================================================
            # rh1 – Malfunctioning RH sensor
            # ==============================================================
            fv_rh1 = _flag_val('flag_rh1')
            doit_rh1_mask = np.zeros(nrec, dtype=bool)
            if fv_rh1 != 'no':
                print("  <--> RH1 correction")
                doit = _find_window(fv_rh1, date_str, epoch_arr)
                df.loc[doit, 'rh_slow'] = errval
                err_flags.loc[doit, 'rh'] |= 1
                doit_rh1_mask = doit
                print(f"  <----> RH1 flag set {int(doit.sum())} times")

            # ==============================================================
            # tf1 – Malfunctioning fast temperature sensor
            # ==============================================================
            fv_tf1 = _flag_val('flag_tf1')
            doit_tf1_mask = np.zeros(nrec, dtype=bool)
            if fv_tf1 != 'no':
                print("  <--> TF1 correction")
                doit = _find_window(fv_tf1, date_str, epoch_arr)
                df.loc[doit, 'temperature_fast'] = errval
                err_flags.loc[doit, 'tf'] |= 1
                doit_tf1_mask = doit
                print(f"  <----> TF1 flag set {int(doit.sum())} times")

            # ==============================================================
            # ts1 – Malfunctioning slow temperature sensor
            # ==============================================================
            fv_ts1 = _flag_val('flag_ts1')
            doit_ts1_mask = np.zeros(nrec, dtype=bool)
            if fv_ts1 != 'no':
                print("  <--> TS1 correction")
                doit = _find_window(fv_ts1, date_str, epoch_arr)
                df.loc[doit, 'temperature_slow'] = errval
                err_flags.loc[doit, 'ts'] |= 1
                doit_ts1_mask = doit
                print(f"  <----> TS1 flag set {int(doit.sum())} times")

            # ==============================================================
            # rh2, tf2, ts2 – Heat contamination flags (flag only, no change)
            # ==============================================================
            fv_rh2 = _flag_val('flag_rh2')
            fv_tf2 = _flag_val('flag_tf2')
            fv_ts2 = _flag_val('flag_ts2')
            if any(f != 'no' for f in [fv_rh2, fv_tf2, fv_ts2]):
                print("  <--> RH2/TF2/TS2 flagging")
                _ensure_raw()
                gprmc_times = np.array([r.time for r in gprmc_list])
                off = 0
                err_cnt = 0
                for j in range(nrec):
                    if j+off >= len(raw_records): break
                    if gprmc_times[j+off] != epoch_arr[j]:
                        off = _find_offset(gprmc_times, epoch_arr, df['time'], df['date'], j)
                    rr = raw_records[j+off]
                    if rr.wndspdraw != errval and rr.wnddirraw != errval:
                        if _bad_wind_for_thermo(rr.wndspdraw, rr.wnddirraw, rr.wnddiroff):
                            err_flags.at[j,'rh'] |= 2
                            err_flags.at[j,'tf'] |= 2
                            err_flags.at[j,'ts'] |= 2
                            err_cnt += 1
                print(f"  <----> RH2/TF2/TS2 flag set {err_cnt} times")

            # ==============================================================
            # ts3 – Slow temperature bias correction (+0.3 °C)
            # ==============================================================
            fv_ts3 = _flag_val('flag_ts3')
            if fv_ts3 != 'no':
                print("  <--> TS3 correction")
                doit = _find_window(fv_ts3, date_str, epoch_arr)
                valid = doit & (df['temperature_slow'] != errval)
                df.loc[valid, 'temperature_slow'] += 0.3
                err_flags.loc[valid, 'ts'] |= 4
                doit_ts3_mask[valid] = True
                print(f"  <----> TS3 flag set {int(valid.sum())} times")

            # ==============================================================
            # Recalculate derived variables where source data was corrected
            # ==============================================================
            errv_p  = (df['pressure'].values         == errval)
            errv_tf = (df['temperature_fast'].values  == errval)
            errv_ts = (df['temperature_slow'].values  == errval)
            errv_rh = (df['rh_slow'].values           == errval)

            dont_p  = doit_p1_mask  | errv_p
            dont_tf = doit_tf1_mask | errv_tf
            dont_ts = doit_ts1_mask | errv_ts
            dont_rh = doit_rh1_mask | errv_rh

            df = _recalc_derived(df, doit_p2_mask, dont_p, dont_tf,
                                 doit_ts3_mask, dont_ts, dont_rh)

            # Summary statistics
            for vname in ('theta', 'theta_e', 'theta_v', 'water_vapor_mixing_ratio',
                          'dewpoint', 'rh_fast'):
                if vname in df.columns:
                    valid = df[vname][df[vname] != errval]
                    pct   = 100.0 * (nrec - len(valid)) / nrec
                    mean  = valid.mean() if len(valid) else float('nan')
                    print(f"  <----> {vname}: mean={mean:.3f}, %missing={pct:.1f}%")

            bad_te = df[(df['theta_e'] < 0) & (df['theta_e'] != errval)]
            bad_ws = df[(df['wind_speed'] < 0) & (df['wind_speed'] != errval)]
            if len(bad_te): print(f"  <!!> Bad theta_e at {len(bad_te)} points")
            if len(bad_ws): print(f"  <!!> Bad wind_speed at {len(bad_ws)} points")

            # ==============================================================
            # Write QC'd ASCII file
            # ==============================================================
            if not nodump:
                fileformat_qcd = {'comet': comet_str, 'version': '2024', 'qc': 'qcd'}
                try:
                    frmt_qcd = get_format(fileformat_qcd, format_file)
                except ValueError:
                    frmt_qcd = get_format(fileformat, format_file)

                err_str_filename = '_'.join(f.lower() for f in active_flags)
                out_path.mkdir(parents=True, exist_ok=True)
                ascii_name = (f"CoMeT{comet_str}_full_{date_str}_{time_str}"
                              f"_L2_{err_str_filename}.txt")
                ascii_out = out_path / ascii_name
                print(f"  <--> Writing ASCII: {ascii_out.name}")
                with open(ascii_out, 'w') as fh:
                    for line in HEADER_INTRO:
                        fh.write(line.replace('{comet}', comet_str) + '\n')
                    for line in HEADER_INSTRUMENTATION:
                        fh.write(line + '\n')
                    fh.write(f"# Missing data value\n#   {int(errval)}\n")
                    for line in (HEADER_DATAKEY_1 if comet_str == '1' else HEADER_DATAKEY_23):
                        fh.write(line + '\n')
                    for line in HEADER_ERRFLAGS:
                        fh.write(line + '\n')

                    err_str_col = []
                    for j in range(nrec):
                        es = _build_err_str(err_flags.iloc[j])
                        err_str_col.append(es)
                        fh.write(_format_record(df.iloc[j], frmt_qcd, es) + '\n')

                df['error_string'] = err_str_col

                if 'qc_txt' in cat.columns:
                    cat.at[cat_idx, 'qc_txt'] = 1
                    cat.to_csv(catalog_file, index=False)
                    print("  <--> Catalog updated")
            else:
                df['error_string'] = [_build_err_str(err_flags.iloc[j]) for j in range(nrec)]

            group_processed.append(df)
            for f in active_flags:
                if f not in group_active_flags:
                    group_active_flags.append(f)
            if first_time_str is None:
                first_time_str = time_str

        # -------------------------------------------------------------------
        # Write one combined NetCDF for this date + vehicle group
        # -------------------------------------------------------------------
        if group_processed and not nodump:
            merged = pd.concat(group_processed, ignore_index=True).sort_values(
                'epoch_time').reset_index(drop=True)

            err_str_filename = '_'.join(f.lower() for f in group_active_flags)
            out_path.mkdir(parents=True, exist_ok=True)

            # Output filename: CoMeT<id>_full_<YYYYMMDD>_<HHMM>_<flags>.nc
            nc_name = (f"CoMeT{comet_str}_full_{date_str}_{first_time_str}"
                       f"_{err_str_filename}.nc")
            nc_path = str(out_path / nc_name)

            if comet_str == 'alpha':
                institution = 'Central Michigan University'
                pi_comment  = 'PI Contact Info: Jason Keeler (keele1j@cmich.edu)'
            else:
                institution = 'University of Nebraska-Lincoln'
                pi_comment  = 'PI Contact Info: Adam Houston (ahouston2@unl.edu)'

            yr_s = date_str[0:4]; mo_s = date_str[4:6]; dy_s = date_str[6:8]
            global_attrs = {
                'title':       f"{yr_s}-{mo_s}-{dy_s} Combined Mesonet and Tracker synchronized data file",
                'source':      f"Combined Mesonet and Tracker {comet_str} (CoMeT-{comet_str})",
                'institution': institution,
                'comment':     pi_comment,
            }

            fileformat_nc = {'comet': comet_str, 'version': '2024', 'qc': 'qcd'}
            print(f"\n  <--> Writing combined NetCDF: {nc_name}")
            write_netcdf(merged, nc_path, global_attrs, fileformat_nc, errval=errval)

    print("\n  Done.")
