"""
GPS NMEA sentence parsers.
Replaces: read_comet1_ascii_gps.pro, create_gps_struct.pro
Also implements the missing read_gprmc / read_gpgga IDL functions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

from .epoch import to_epoch

ERRVAL = -999.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GprmcRecord:
    """Parsed $GPRMC sentence."""
    time:     float = ERRVAL   # Unix epoch (from GPS date+time)
    time_str: str   = ''       # raw HHMMSS.s from sentence
    date_str: str   = ''       # raw DDMMYY from sentence
    latitude:  float = ERRVAL
    longitude: float = ERRVAL
    spd:   float = ERRVAL      # speed over ground [knots]
    track: float = ERRVAL      # course over ground [°]
    valid: bool  = False


@dataclass
class GpggaRecord:
    """Parsed $GPGGA sentence."""
    time_str:  str   = ''
    latitude:  float = ERRVAL
    longitude: float = ERRVAL
    alt:       float = ERRVAL  # altitude MSL [m]
    num_sats:  int   = 0
    hdop:      float = ERRVAL
    valid:     bool  = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nmea_to_decimal(val_str: str, direction: str) -> float:
    """Convert NMEA lat/lon string (DDDMM.MMMM) to decimal degrees."""
    if not val_str.strip():
        return ERRVAL
    val = float(val_str)
    degrees = int(val / 100)
    minutes = val - degrees * 100.0
    dd = degrees + minutes / 60.0
    if direction.strip().upper() in ('S', 'W'):
        dd = -dd
    return dd


# ---------------------------------------------------------------------------
# GPRMC parser
# ---------------------------------------------------------------------------

def parse_gprmc(sentence: str) -> GprmcRecord:
    """
    Parse a $GPRMC NMEA sentence into a GprmcRecord.

    Standard format:
        $GPRMC,HHMMSS.ss,A,LLLL.LL,N,YYYYY.YY,W,SPD,TRK,DDMMYY,VAR,E*CS
    """
    rec = GprmcRecord()
    try:
        sentence = sentence.split('*')[0].strip()
        parts = sentence.split(',')
        if len(parts) < 10:
            return rec

        rec.time_str = parts[1]
        status       = parts[2]
        lat_str      = parts[3]
        lat_dir      = parts[4]
        lon_str      = parts[5]
        lon_dir      = parts[6]
        spd_str      = parts[7]
        trk_str      = parts[8]
        rec.date_str = parts[9]

        rec.valid = (status.strip().upper() == 'A')

        rec.latitude  = _nmea_to_decimal(lat_str, lat_dir)
        rec.longitude = _nmea_to_decimal(lon_str, lon_dir)

        if spd_str.strip():
            rec.spd = float(spd_str)
        if trk_str.strip():
            rec.track = float(trk_str)

        # Compute Unix epoch from GPS time + date
        ts  = rec.time_str
        ds  = rec.date_str
        if ts and ds and len(ds) == 6 and len(ts) >= 6:
            dd = int(ds[0:2])
            mm = int(ds[2:4])
            yy = int(ds[4:6]) + 2000
            hh = int(ts[0:2])
            mi = int(ts[2:4])
            ss = float(ts[4:]) if len(ts) > 4 else 0.0
            rec.time = to_epoch(yy, mm, dd, hh, mi, ss)

    except (ValueError, IndexError):
        pass

    return rec


# ---------------------------------------------------------------------------
# GPGGA parser
# ---------------------------------------------------------------------------

def parse_gpgga(sentence: str) -> GpggaRecord:
    """
    Parse a $GPGGA NMEA sentence into a GpggaRecord.

    Standard format:
        $GPGGA,HHMMSS.ss,LLLL.LL,N,YYYYY.YY,W,Q,NN,DOP,ALT,M,...
    """
    rec = GpggaRecord()
    try:
        sentence = sentence.split('*')[0].strip()
        parts = sentence.split(',')
        if len(parts) < 10:
            return rec

        rec.time_str  = parts[1]
        lat_str       = parts[2]
        lat_dir       = parts[3]
        lon_str       = parts[4]
        lon_dir       = parts[5]
        fix_q         = int(parts[6]) if parts[6].strip() else 0
        sats          = parts[7]
        hdop          = parts[8]
        alt_str       = parts[9]

        rec.valid     = (fix_q > 0)
        rec.latitude  = _nmea_to_decimal(lat_str, lat_dir)
        rec.longitude = _nmea_to_decimal(lon_str, lon_dir)

        if sats.strip():
            rec.num_sats = int(sats)
        if hdop.strip():
            rec.hdop = float(hdop)
        if alt_str.strip():
            rec.alt = float(alt_str)

    except (ValueError, IndexError):
        pass

    return rec


# ---------------------------------------------------------------------------
# CoMeT-1 GPS file reader (replaces read_comet1_ascii_gps.pro)
# ---------------------------------------------------------------------------

def read_comet1_gps_file(filepath: str) -> Dict[str, List[str]]:
    """
    Read a CoMeT-1 raw GPS file, extracting GPRMC, GPGGA, and PGRME sentences.

    Args:
        filepath: path to the IMeT1_gpsraw_*.txt file.

    Returns:
        dict with keys 'gprmc', 'gpgga', 'pgrme' – each a list of raw sentence strings.
    """
    gprmc_list: List[str] = []
    gpgga_list: List[str] = []
    pgrme_list: List[str] = []

    with open(filepath, 'r', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if '$GPRMC' in line:
                gprmc_list.append(line)
            elif '$GPGGA' in line:
                gpgga_list.append(line)
            elif '$PGRME' in line:
                pgrme_list.append(line)

    return {'gprmc': gprmc_list, 'gpgga': gpgga_list, 'pgrme': pgrme_list}


# ---------------------------------------------------------------------------
# Batch parser (replaces create_gps_struct.pro)
# ---------------------------------------------------------------------------

def create_gps_records(sentences: List[str], gps_type: str):
    """
    Parse a list of NMEA sentences into a list of record dataclasses.
    Replaces create_gps_struct.pro.

    Args:
        sentences: list of raw NMEA sentence strings.
        gps_type:  'gprmc' or 'gpgga'.

    Returns:
        List of GprmcRecord or GpggaRecord objects.
    """
    if gps_type == 'gprmc':
        return [parse_gprmc(s) for s in sentences]
    elif gps_type == 'gpgga':
        return [parse_gpgga(s) for s in sentences]
    else:
        raise ValueError(f"Unknown gps_type: {gps_type!r}")


# ---------------------------------------------------------------------------
# PGRME epoch time extractor (CoMeT-1)
# ---------------------------------------------------------------------------

def pgrme_epoch_times(pgrme_sentences: List[str]) -> np.ndarray:
    """
    Extract computer epoch times from Garmin $PGRME sentences.
    The CoMeT-1 logger appends a computer timestamp as the 8th comma-delimited
    field (index 7, 0-based) of the standard $PGRME sentence.

    Returns:
        numpy array of epoch times (float64); NaN where parsing fails.
    """
    times = np.full(len(pgrme_sentences), np.nan, dtype=float)
    for i, sent in enumerate(pgrme_sentences):
        try:
            parts = sent.split('*')[0].split(',')
            if len(parts) > 7:
                times[i] = float(parts[7])
        except (ValueError, IndexError):
            pass
    return times
