"""
CoMeT ASCII data file reader.
Replaces: read_comet_ascii.pro
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

from .data_format import get_format
from .epoch import parse_date_time

ERRVAL = -999.0


def read_comet_ascii(
    filepath: str,
    fileformat: dict,
    format_file: str,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Read a CoMeT ASCII data file into a pandas DataFrame.

    Replicates read_comet_ascii.pro.  The returned DataFrame contains all
    columns from the data file plus three derived columns:
        epoch_time  – Unix timestamp [s]
        u           – Eastward wind component [m/s]
        v           – Northward wind component [m/s]

    Multiplication and additive factors from CoMeT_data_format.csv are applied.

    Args:
        filepath:    Path to the CoMeT data file (.txt).
        fileformat:  dict with keys 'comet', 'version', 'qc'.
        format_file: Path to CoMeT_data_format.csv.
        quiet:       If True, suppress informational printing.

    Returns:
        pandas DataFrame (one row per record).

    Raises:
        FileNotFoundError: if the data file does not exist.
    """
    if not quiet:
        print(f"  <> Processing CoMeT:  {fileformat['comet']}")
        print(f"  <> Version:           {fileformat['version']}")
        print(f"  <> QC status:         {fileformat['qc']}")

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    frmt      = get_format(fileformat, format_file)
    n_header  = int(frmt['headerlines'].iloc[0])
    n_fields  = len(frmt)

    if not quiet:
        print(f"  <> Reading data from {filepath}")

    # --- Build dtype map for read_csv ---
    dtype_map: Dict[str, Any] = {}
    col_names = list(frmt['parameter_name'])
    for _, row in frmt.iterrows():
        name  = row['parameter_name']
        ttype = row['type'].lower().strip()
        if ttype == 'string':
            dtype_map[name] = str
        elif ttype in ('float', 'double'):
            dtype_map[name] = float
        elif ttype == 'long':
            dtype_map[name] = pd.Int64Dtype()

    df = pd.read_csv(
        filepath,
        skiprows=n_header,
        header=None,
        names=col_names,
        dtype=dtype_map,
        na_values=['', 'nan', 'NaN'],
        keep_default_na=True,
        on_bad_lines='warn',
    )

    # --- Apply scale/offset factors ---
    for _, row in frmt.iterrows():
        name      = row['parameter_name']
        mult_fact = float(row.get('mult_fact', 1.0))
        add_fact  = float(row.get('add_fact', 0.0))
        ttype     = str(row['type']).lower().strip()
        if ttype != 'string' and name in df.columns:
            if mult_fact != 1.0:
                df[name] = pd.to_numeric(df[name], errors='coerce') * mult_fact
            if add_fact != 0.0:
                df[name] = pd.to_numeric(df[name], errors='coerce') + add_fact

    # --- Ensure string columns are stripped ---
    for _, row in frmt.iterrows():
        if str(row['type']).lower().strip() == 'string':
            name = row['parameter_name']
            if name in df.columns:
                df[name] = df[name].astype(str).str.strip()

    # --- Derived columns: epoch_time, u, v ---
    comet   = str(fileformat['comet'])
    qc      = str(fileformat['qc'])
    version = str(fileformat['version'])

    epoch_times = np.zeros(len(df), dtype=float)
    for i, row in df.iterrows():
        try:
            epoch_times[i] = parse_date_time(
                str(row['date']), str(row['time']), comet, qc, version
            )
        except Exception:
            epoch_times[i] = ERRVAL

    df['epoch_time'] = epoch_times

    # u = eastward wind = -ws * sin(wd_rad)  (meteorological convention)
    if 'wind_speed' in df.columns and 'wind_direction' in df.columns:
        ws = pd.to_numeric(df['wind_speed'], errors='coerce').fillna(ERRVAL)
        wd = pd.to_numeric(df['wind_direction'], errors='coerce').fillna(ERRVAL)
        u  = np.where(
            (ws != ERRVAL) & (wd != ERRVAL),
            -ws * np.sin(np.deg2rad(wd)),
            ERRVAL,
        )
        v  = np.where(
            (ws != ERRVAL) & (wd != ERRVAL),
            -ws * np.cos(np.deg2rad(wd)),
            ERRVAL,
        )
        df['u'] = u.astype(float)
        df['v'] = v.astype(float)
    else:
        df['u'] = ERRVAL
        df['v'] = ERRVAL

    return df.reset_index(drop=True)
