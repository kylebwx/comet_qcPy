"""
Time utilities.
Replaces: epoch.pro, epoch2datetime.pro
"""
from datetime import datetime, timezone
from typing import Tuple, Union
import numpy as np


def to_epoch(yr, mo, dy, hr, mn, ss) -> float:
    """
    Convert datetime components to Unix epoch (seconds since 1970-01-01 00:00:00 UTC).

    Args:
        yr, mo, dy, hr, mn: integer date/time components
        ss: seconds (may be float)

    Returns:
        Float epoch time in seconds.
    """
    base = datetime(int(yr), int(mo), int(dy), int(hr), int(mn), 0,
                    tzinfo=timezone.utc)
    return base.timestamp() + float(ss)


def from_epoch(epoch_time: float) -> Tuple[int, int, int, int, int, float]:
    """
    Convert Unix epoch to (yr, mo, dy, hr, mn, ss).

    Args:
        epoch_time: seconds since 1970-01-01 00:00:00 UTC

    Returns:
        Tuple (year, month, day, hour, minute, seconds)
    """
    epoch_time = float(epoch_time)
    base_int = int(epoch_time)
    frac = epoch_time - base_int
    dt = datetime.fromtimestamp(base_int, tz=timezone.utc)
    return dt.year, dt.month, dt.day, dt.hour, dt.minute, float(dt.second) + frac


def parse_date_time(date_str: str, time_str: str,
                    comet: str, qc: str, version: str) -> float:
    """
    Parse the date/time strings from a CoMeT record into Unix epoch.

    Date format is DDMMYY for most cases.
    Exception: CoMeT-1 QC'd data from 2016/2017 uses YYYYMMDD.

    Args:
        date_str: date string from data record
        time_str: time string HHMMSS.s
        comet: CoMeT designation
        qc: QC level ('orig', 'qcd')
        version: data version string

    Returns:
        Unix epoch float
    """
    date_str = date_str.strip()
    time_str = time_str.strip()

    if (str(comet) == '1' and qc == 'qcd'
            and version in ('2016', '2017')):
        # YYYYMMDD format
        yr = int(date_str[0:4])
        mo = int(date_str[4:6])
        dy = int(date_str[6:8])
    else:
        # DDMMYY format
        dy = int(date_str[0:2])
        mo = int(date_str[2:4])
        yr = int(date_str[4:6]) + 2000

    hr = int(time_str[0:2])
    mn = int(time_str[2:4])
    ss = float(time_str[4:]) if len(time_str) > 4 else 0.0

    return to_epoch(yr, mo, dy, hr, mn, ss)


def epoch_to_date_time_str(epoch_time: float):
    """
    Convert epoch to date/time strings in DDMMYY / HHMMSS.s format.

    Returns:
        (date_str, time_str)
    """
    yr, mo, dy, hr, mn, ss = from_epoch(epoch_time)
    date_str = f"{dy:02d}{mo:02d}{yr - 2000:02d}"
    time_str = f"{hr:02d}{mn:02d}{ss:04.1f}"
    return date_str, time_str
