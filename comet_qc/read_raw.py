"""
Raw data file readers.
Replaces: read_comet_raw.pro, get_raw.pro
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import numpy as np

from .gps_parser import (
    GprmcRecord, GpggaRecord,
    parse_gprmc, parse_gpgga,
    read_comet1_gps_file,
    pgrme_epoch_times,
    create_gps_records,
)
from .epoch import to_epoch

ERRVAL = -999.0
NHEAD_23 = 4   # Header lines in raw files for CoMeT 2/3/alpha


# ---------------------------------------------------------------------------
# Raw record container
# ---------------------------------------------------------------------------

@dataclass
class RawRecord:
    """Single record from a CoMeT raw data file."""
    time:     str   = ''
    temp_f:   float = ERRVAL
    temp_s:   float = ERRVAL
    rh_s:     float = ERRVAL
    wndspd:   float = ERRVAL
    wnddir:   float = ERRVAL
    dewpt:    float = ERRVAL
    rh_f:     float = ERRVAL
    pres:     float = ERRVAL
    compt1:   float = ERRVAL
    paneltemp: float = ERRVAL
    battvolt:  float = ERRVAL
    wndspdraw: float = ERRVAL
    wnddirraw: float = ERRVAL
    fluxdir:   float = ERRVAL
    gprmc:     str  = ''
    gpgga:     str  = ''
    utcoff:    int  = 0
    wnddiroff: float = 0.0
    compt2:    float = ERRVAL


# ---------------------------------------------------------------------------
# Campbell logger record parser (strip function equivalent)
# ---------------------------------------------------------------------------

def _strip_all(rec: str, nelem: int, comet: str) -> List[str]:
    """
    Extract all fields from a concatenated Campbell logger raw record string.

    The Campbell CR6 logger outputs fields labeled with sequential two-digit
    indices ("  01 … 02 … 03 …").  For CoMeT 2/3/alpha the field ordering in
    the file is non-standard (fields 1-10, then 20, then 11-19, then 21+) to
    accommodate the GPS strings.  This replicates the IDL strip() function.

    Args:
        rec:   Concatenated record string (two logger lines joined).
        nelem: Number of fields to extract (n_tags of template1 = 21).
        comet: CoMeT designation string ('1', '2', '3', 'alpha').

    Returns:
        List of *nelem* field value strings.
    """
    elems = [str(ERRVAL)] * nelem
    for j in range(nelem):
        if comet != '1':
            if j == 9:
                srch0 = '10'
                srch1 = '  20'
            elif j == 10:
                srch0 = '20'
                srch1 = f'  {j + 1:02d}'   # '  11'
            elif j > 10:
                srch0 = f'{j:02d}'
                srch1 = f'  {j + 1:02d}'
            else:
                srch0 = f'{j + 1:02d}'
                srch1 = f'  {j + 2:02d}'
        else:
            srch0 = f'{j + 1:02d}'
            srch1 = f'  {j + 2:02d}'

        i0 = rec.find(srch0)
        if i0 < 0:
            continue

        if j == nelem - 1:
            chunk = rec[i0 + 2:]
            remaining = ''
        else:
            i1 = rec.find(srch1, i0)
            if i1 < 0:
                chunk = rec[i0 + 2:]
                remaining = ''
            else:
                chunk = rec[i0 + 2: i1]
                remaining = rec[i1 + 2:]

        # Clean up chunk
        chunk = ' '.join(chunk.split())    # collapse whitespace
        if chunk == '+':
            chunk = str(int(ERRVAL))
        if chunk.startswith('+$G'):
            chunk = chunk[1:]              # trim leading '+' from GPS strings

        rec = remaining
        elems[j] = chunk

    return elems


def _safe_float(s: str, default: float = ERRVAL) -> float:
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return default


def _parse_record_23alpha(rec1: str, rec2: str, comet: str) -> RawRecord:
    """Parse a two-line raw record from CoMeT 2, 3, or alpha."""
    NELEM = 21   # len of template1
    elems = _strip_all(rec1 + rec2, NELEM, comet)

    r = RawRecord()
    # elem[0]=hhmm, elem[1]=ss; combine into HHMMSS.s
    try:
        r.time = elems[0][:4] + elems[1]
    except Exception:
        pass

    r.temp_f    = _safe_float(elems[2])
    r.temp_s    = _safe_float(elems[3])
    r.rh_s      = _safe_float(elems[4])
    r.wndspd    = _safe_float(elems[5])
    r.wnddir    = _safe_float(elems[6])
    r.dewpt     = _safe_float(elems[7])
    r.rh_f      = _safe_float(elems[8])
    r.pres      = _safe_float(elems[9])
    r.compt1    = _safe_float(elems[10])
    r.paneltemp = _safe_float(elems[11])
    r.battvolt  = _safe_float(elems[12])
    r.wndspdraw = _safe_float(elems[13])
    r.wnddirraw = _safe_float(elems[14])
    r.fluxdir   = _safe_float(elems[15])
    r.gprmc     = elems[16]
    r.gpgga     = elems[17]
    try:
        r.utcoff = int(_safe_float(elems[18], 0))
    except Exception:
        pass
    r.wnddiroff = _safe_float(elems[19])
    r.compt2    = _safe_float(elems[20])

    return r


# ---------------------------------------------------------------------------
# Public reader functions
# ---------------------------------------------------------------------------

def read_raw_23alpha(filepath: str, comet: str) -> List[RawRecord]:
    """
    Read a raw data file for CoMeT 2, 3, or alpha.

    File naming: CoMeT[N]_raw_[YYYYMMDD]_[HHMM].txt
    Format: 4 header lines, then pairs of data lines per record.
    """
    with open(filepath, 'r', errors='replace') as fh:
        lines = fh.readlines()

    data_lines = lines[NHEAD_23:]
    records: List[RawRecord] = []

    for i in range(0, len(data_lines) - 1, 2):
        r1 = data_lines[i].rstrip('\r\n')
        r2 = data_lines[i + 1].rstrip('\r\n') if i + 1 < len(data_lines) else ''
        try:
            records.append(_parse_record_23alpha(r1, r2, comet))
        except Exception:
            records.append(RawRecord())

    return records


def _sync_scenarios(raw_times, data_times, i_raw, i_full, field_getter, setter):
    """
    Sync one raw data stream to the full data time axis (CoMeT-1 only).
    Replicates the IDL sync_scenarios() function.

    Returns updated i_raw and a 'bad' flag (1 if sync failed).
    """
    bad = 0
    n_raw  = len(raw_times)
    n_full = len(data_times)

    if i_raw >= n_raw:
        bad = 1
        return i_raw, bad

    if raw_times[i_raw] == data_times[i_full]:
        setter(i_full, field_getter(i_raw))
        i_raw += 1
    elif raw_times[i_raw] > data_times[i_full]:
        # Full data ahead – use previous raw or neighbour logic
        if (i_raw > 0 and i_full + 1 < n_full
                and raw_times[i_raw - 1] == data_times[i_full]
                and raw_times[i_raw + 1] == data_times[i_full + 1]):
            setter(i_full, field_getter(i_raw))
            i_raw += 1
        elif (i_raw > 0
              and raw_times[i_raw - 1] == data_times[i_full]):
            setter(i_full, field_getter(i_raw))
        else:
            setter(i_full, None)   # caller should fill with NaN
            bad = 1
    else:
        # raw is behind – search ahead
        matches = np.where(raw_times == data_times[i_full])[0]
        if len(matches) > 0:
            i_raw = int(matches[0])
            setter(i_full, field_getter(i_raw))
            i_raw += 1
        else:
            setter(i_full, None)
            bad = 1

    return i_raw, bad


def read_raw_comet1(filepath_gps: str, full_data) -> List[RawRecord]:
    """
    Read and sync raw CoMeT-1 data (GPS, fluxgate, pressure, THV streams).
    Replicates the CoMeT-1 branch of read_comet_raw.pro.

    Args:
        filepath_gps: Path to IMeT1_gpsraw_*.txt
        full_data:    DataFrame returned by read_comet_ascii (used for timing).

    Returns:
        List of RawRecord objects aligned to full_data.
    """
    base      = Path(filepath_gps)
    file_flx  = str(base).replace('gpsraw', 'fluxraw')
    file_prs  = str(base).replace('gpsraw', 'pressureraw')
    file_thv  = str(base).replace('gpsraw', 'thvraw')

    # -- GPS --
    gps_dict  = read_comet1_gps_file(filepath_gps)
    gprmc_recs = create_gps_records(gps_dict['gprmc'], 'gprmc')
    gpgga_recs = create_gps_records(gps_dict['gpgga'], 'gpgga')
    gps_epochs = pgrme_epoch_times(gps_dict['pgrme'])

    n_full = len(full_data)
    records = [RawRecord() for _ in range(n_full)]

    # -- Fluxgate --
    flx_dir   = []
    flx_epoch = []
    try:
        with open(file_flx, 'r', errors='replace') as fh:
            next(fh)   # header
            for line in fh:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    flx_dir.append(float(parts[0]) if parts[0].strip() else np.nan)
                    flx_epoch.append(float(parts[1]))
    except FileNotFoundError:
        print(f"  <!> Fluxgate file not found: {file_flx}")

    # -- Pressure --
    prs_val   = []
    prs_epoch = []
    try:
        with open(file_prs, 'r', errors='replace') as fh:
            next(fh)   # header
            for line in fh:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    prs_val.append(float(parts[0]) if parts[0].strip() else np.nan)
                    # Truncate to tenths digit (matches IDL behaviour)
                    ep_str = parts[1].strip()
                    dot = ep_str.find('.')
                    ep_trunc = ep_str[:dot + 2] if dot >= 0 else ep_str
                    prs_epoch.append(float(ep_trunc))
    except FileNotFoundError:
        print(f"  <!> Pressure file not found: {file_prs}")

    # -- THV (temp/humidity/wind) --
    thv_time     = []
    thv_temp_f   = []
    thv_temp_s   = []
    thv_rh_s     = []
    thv_wndspdraw = []
    thv_wnddirraw = []
    thv_epoch    = []
    NELEM_THV = 20

    try:
        with open(file_thv, 'r', errors='replace') as fh:
            next(fh)   # header
            for line in fh:
                elems = _strip_all(line.rstrip('\r\n'), NELEM_THV, '1')
                # indices follow IDL read_comet_raw CoMeT-1 branch
                thv_time.append(elems[1][:4] + elems[2] if len(elems) > 2 else '')
                thv_temp_f.append(_safe_float(elems[4]))
                thv_temp_s.append(_safe_float(elems[5]))
                thv_rh_s.append(_safe_float(elems[6]))
                thv_wndspdraw.append(_safe_float(elems[8]))
                thv_wnddirraw.append(_safe_float(elems[9]))
                thv_epoch.append(_safe_float(elems[19]))
    except FileNotFoundError:
        print(f"  <!> THV file not found: {file_thv}")

    print("  <----> Syncing CoMeT-1 data")

    # Align each stream to full_data epoch times
    full_ct_thv  = full_data['computer_time_thv'].values     if 'computer_time_thv'      in full_data.columns else np.zeros(n_full)
    full_ct_prs  = full_data['computer_time_pressure'].values if 'computer_time_pressure' in full_data.columns else np.zeros(n_full)
    full_ct_flx  = full_data['computer_time_fluxgate'].values if 'computer_time_fluxgate' in full_data.columns else np.zeros(n_full)
    full_ct_gps  = full_data['computer_time_gps'].values     if 'computer_time_gps'      in full_data.columns else np.zeros(n_full)

    # THV sync
    thv_ep = np.array(thv_epoch)
    i_thv  = 0
    for i in range(n_full):
        if i_thv < len(thv_ep):
            matches = np.where(thv_ep == full_ct_thv[i])[0]
            if len(matches) > 0:
                k = int(matches[0])
                records[i].temp_f    = thv_temp_f[k]   if k < len(thv_temp_f)    else ERRVAL
                records[i].temp_s    = thv_temp_s[k]   if k < len(thv_temp_s)    else ERRVAL
                records[i].rh_s      = thv_rh_s[k]     if k < len(thv_rh_s)      else ERRVAL
                records[i].wndspdraw = thv_wndspdraw[k] if k < len(thv_wndspdraw) else ERRVAL
                records[i].wnddirraw = thv_wnddirraw[k] if k < len(thv_wnddirraw) else ERRVAL
                records[i].time      = thv_time[k]      if k < len(thv_time)      else ''
                i_thv = k + 1

    # Pressure sync
    prs_ep = np.array(prs_epoch)
    for i in range(n_full):
        matches = np.where(prs_ep == full_ct_prs[i])[0]
        if len(matches) > 0:
            k = int(matches[0])
            records[i].pres = prs_val[k] if k < len(prs_val) else ERRVAL

    # Fluxgate sync
    flx_ep = np.array(flx_epoch)
    for i in range(n_full):
        matches = np.where(flx_ep == full_ct_flx[i])[0]
        if len(matches) > 0:
            k = int(matches[0])
            records[i].fluxdir = flx_dir[k] if k < len(flx_dir) else ERRVAL

    # GPS sync
    gps_ep = np.array(gps_epochs)
    for i in range(n_full):
        matches = np.where(gps_ep == full_ct_gps[i])[0]
        if len(matches) > 0:
            k = int(matches[0])
            records[i].gprmc = gps_dict['gprmc'][k] if k < len(gps_dict['gprmc']) else ''
            records[i].gpgga = gps_dict['gpgga'][k] if k < len(gps_dict['gpgga']) else ''

    return records


# ---------------------------------------------------------------------------
# get_raw (public entry point – replaces get_raw.pro)
# ---------------------------------------------------------------------------

def get_raw(comet: str, fileformat: dict, full_filepath: str, full_data=None
            ) -> Tuple[List[RawRecord], List[GprmcRecord], List[GpggaRecord]]:
    """
    Read raw data and build GPRMC / GPGGA record lists.
    Replicates get_raw.pro.

    Args:
        comet:          CoMeT designation ('1', '2', '3', 'alpha').
        fileformat:     dict with 'comet', 'version', 'qc'.
        full_filepath:  Path to the full (non-raw) data file.
        full_data:      DataFrame from read_comet_ascii (required for CoMeT-1 syncing).

    Returns:
        Tuple (raw_records, gprmc_list, gpgga_list)
    """
    if comet != '1':
        raw_filepath = full_filepath.replace('full', 'raw')
        raw_records  = read_raw_23alpha(raw_filepath, comet)
    else:
        raw_filepath = full_filepath.replace('full', 'gpsraw')
        raw_filepath = raw_filepath.replace('CoMeT1_gpsraw', 'IMeT1_gpsraw')
        raw_records  = read_raw_comet1(raw_filepath, full_data)

    gprmc_list = create_gps_records([r.gprmc for r in raw_records], 'gprmc')
    gpgga_list = create_gps_records([r.gpgga for r in raw_records], 'gpgga')

    return raw_records, gprmc_list, gpgga_list
