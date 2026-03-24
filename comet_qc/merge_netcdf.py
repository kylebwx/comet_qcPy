"""
Merge multiple QC'd CoMeT ASCII files and write a combined NetCDF file.
Replaces: comet_mergeqcd_createncdf_2024.pro
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from .read_ascii import read_comet_ascii
from .netcdf_writer import write_netcdf
from .epoch import to_epoch, from_epoch

ERRVAL = -999.0


def _build_dir_merge(cat_row: pd.Series, comet_str: str,
                     dir_template: str, dir_type: str) -> str:
    """Expand the directory template (same logic as in qc.py)."""
    from .qc import _build_dir  # reuse
    return _build_dir(cat_row, comet_str, dir_template, dir_type)


def _active_flags(flag_df: pd.DataFrame, idx) -> List[str]:
    """Return list of active flag names for a catalog row."""
    active = []
    for col in flag_df.columns:
        val = str(flag_df.at[idx, col]).strip()
        if val != 'no':
            active.append(col.replace('flag_', '').upper())
    return active


def run_merge_netcdf(
    catalog_file: str,
    format_file: str,
    dir_template: str,
    project: str = 'all',
    day: str = 'all',
    time_filter: str = 'all',
    comet_filter: str = 'all',
    site_filter: str = 'all',
) -> None:
    """
    Merge QC'd CoMeT text files that share a directory and create NetCDF output.

    Replicates comet_mergeqcd_createncdf_2024.pro.

    Args:
        catalog_file:   Path to comet_qc_2024.csv.
        format_file:    Path to CoMeT_data_format.csv.
        dir_template:   Directory template with tokens [project], [YYYYMMDD],
                        [*] (CoMeT id), [type].
        project/day/time_filter/comet_filter/site_filter:
                        Catalog filters (all default to 'all').
    """
    cat = pd.read_csv(catalog_file)
    cat.columns = [c.strip().lower().replace(' ', '_') for c in cat.columns]

    flag_cols = [c for c in cat.columns if c.startswith('flag_')]
    flag_df   = cat[flag_cols].copy()

    cat['date_str'] = (cat['year'].astype(int).map(lambda y: f"{y:04d}") +
                       cat['month'].astype(int).map(lambda m: f"{m:02d}") +
                       cat['day'].astype(int).map(lambda d: f"{d:02d}"))
    cat['time_str'] = cat['time'].astype(int).map(lambda t: f"{t:04d}")

    # Filter
    mask = pd.Series(True, index=cat.index)
    choice = {'project': project, 'day': day, 'time': time_filter,
              'comet': comet_filter, 'site': site_filter}
    for col, val in choice.items():
        if val != 'all' and col in cat.columns:
            mask &= cat[col].astype(str).str.strip() == str(val).strip()

    selected = cat[mask].copy()
    if selected.empty:
        print("  <!> No matching entries in catalog.")
        return

    # Build directory and file names for every selected row
    dirs  = []
    files = []
    for idx, row in selected.iterrows():
        comet_str = str(row['comet']).strip()
        active    = _active_flags(flag_df, idx)
        err_str_f = '_'.join(f.lower() for f in active)

        from .qc import _build_dir
        qcd_dir = _build_dir(row, comet_str, dir_template, "QC'd data")
        fname   = (f"CoMeT{comet_str}_full_{row['date_str']}_{row['time_str']}"
                   f"_L2_{err_str_f}.txt")
        dirs.append(qcd_dir)
        files.append(str(Path(qcd_dir) / fname))

    selected = selected.copy()
    selected['_dir']  = dirs
    selected['_file'] = files

    # Sort by directory, then by file name (ensures chronological order)
    selected = selected.sort_values(['_dir', '_file']).reset_index(drop=True)

    # Group by directory
    for qcd_dir, group in selected.groupby('_dir', sort=False):
        comet_str = str(group.iloc[0]['comet']).strip()
        date_str  = str(group.iloc[0]['date_str'])
        time_str  = str(group.iloc[0]['time_str'])
        fileformat = {'comet': comet_str, 'version': '2024', 'qc': 'qcd'}

        # Merge files
        data_all = []
        all_flags: List[str] = []
        for _, row in group.iterrows():
            fpath = row['_file']
            print(f"  <> Processing {fpath}")
            if not Path(fpath).exists():
                print(f"  <!> File not found, skipping: {fpath}")
                continue
            try:
                df = read_comet_ascii(fpath, fileformat, format_file, quiet=True)
                data_all.append(df)
            except Exception as e:
                print(f"  <!> Error reading {fpath}: {e}")
                continue
            # Collect active flags
            for fc in flag_cols:
                val = str(flag_df.at[row.name, fc]).strip()
                fn  = fc.replace('flag_', '').upper()
                if val != 'no' and fn not in all_flags:
                    all_flags.append(fn)

        if not data_all:
            print(f"  <!> No data loaded for directory {qcd_dir}")
            continue

        merged = pd.concat(data_all, ignore_index=True).sort_values(
            'epoch_time').reset_index(drop=True)

        err_str_filename = '_'.join(f.lower() for f in all_flags)

        # Global attributes
        yr = date_str[0:4]
        mo = date_str[4:6]
        dy = date_str[6:8]
        date_nice = f"{yr}-{mo}-{dy}"

        if comet_str == 'alpha':
            institution = 'Central Michigan University'
            pi_comment  = 'PI Contact Info: Jason Keeler (keele1j@cmich.edu)'
        else:
            institution = 'University of Nebraska-Lincoln'
            pi_comment  = 'PI Contact Info: Adam Houston (ahouston2@unl.edu)'

        global_attrs = {
            'title':       f"{date_nice} Combined Mesonet and Tracker synchronized data file",
            'source':      f"Combined Mesonet and Tracker {comet_str} (CoMeT-{comet_str})",
            'institution': institution,
            'comment':     pi_comment,
        }

        nc_name = (f"UNL.CoMeT{comet_str}.{date_str}.{time_str}.L2_2024"
                   f".{err_str_filename.replace('_', '.')}.nc")
        nc_path = str(Path(qcd_dir) / nc_name)

        print(f"  <--> Creating NetCDF file: {nc_path}")
        write_netcdf(merged, nc_path, global_attrs, fileformat)

    print("\n  Done.")
