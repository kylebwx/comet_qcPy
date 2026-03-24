"""
Wind calculation utilities.
Replaces: get_comet_wind_from_raw.pro, comet_find_winddir_offset.pro
"""
from __future__ import annotations
import math
from typing import Tuple
import numpy as np

ERRVAL = -999.0


# ---------------------------------------------------------------------------
# Wind vector from raw anemometer + GPS
# ---------------------------------------------------------------------------

def wind_from_raw(
    veh_track: float,
    veh_spd_kts: float,
    anem_dir: float,
    anem_spd_ms: float,
    anem_dir_off: float,
) -> Tuple[float, float, float, float]:
    """
    Derive earth-relative wind from raw anemometer and GPS data.

    Replicates get_comet_wind_from_raw.pro exactly.

    Args:
        veh_track:    GPS course-over-ground [°] (direction vehicle is moving;
                      90° = eastward).
        veh_spd_kts:  GPS vehicle speed [knots].
        anem_dir:     Raw anemometer direction [°].
        anem_spd_ms:  Raw anemometer speed [m/s].
        anem_dir_off: Wind direction offset (WNDDIROFF) [°].

    Returns:
        Tuple (wind_spd [m/s], wind_dir [°], u [m/s], v [m/s])
        where u = eastward component, v = northward component.
    """
    deg2rad = math.pi / 180.0
    rad2deg = 180.0 / math.pi

    # Vehicle heading in meteorological convention (direction FROM which the
    # vehicle travels) and in radians.
    head = veh_track - 180.0
    if head < 0.0:
        head += 360.0
    head_rad = head * deg2rad

    # Vehicle translation vector (m/s); positive = towards west/south
    veh_ms = veh_spd_kts * 0.514444
    trans = np.array([veh_ms * math.sin(head_rad),
                      veh_ms * math.cos(head_rad)])

    # Apparent wind direction in meteorological coordinates
    winddir_app = anem_dir - anem_dir_off + veh_track - 180.0
    if winddir_app < 0.0:
        winddir_app += 360.0
    winddir_app_rad = winddir_app * deg2rad
    app = anem_spd_ms * np.array([math.sin(winddir_app_rad),
                                   math.cos(winddir_app_rad)])

    diff = trans - app

    # Wind direction from diff vector
    if diff[0] >= 0.0:
        wind_dir = 90.0  - rad2deg * math.atan2(diff[1], diff[0])
    else:
        wind_dir = 450.0 - rad2deg * math.atan2(diff[1], diff[0])
    if wind_dir > 360.0:
        wind_dir -= 360.0

    wind_spd = math.sqrt(diff[0]**2 + diff[1]**2)
    u = -diff[0]
    v = -diff[1]

    return wind_spd, wind_dir, u, v


# ---------------------------------------------------------------------------
# Wind direction offset optimisation
# ---------------------------------------------------------------------------

def find_winddir_offset(
    gprmc_records,
    raw_records,
    off_range: Tuple[float, float, float] = (-20.0, 270.0, 1.0),
    spd_thresh_kts: float = 40.0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Find the wind-direction offset (WNDDIROFF) that minimises mean wind speed.

    Replicates comet_find_winddir_offset.pro.

    Args:
        gprmc_records: List of GprmcRecord objects.
        raw_records:   List of RawRecord objects (same length).
        off_range:     (start, stop, step) for offset search in degrees.
        spd_thresh_kts: Only records where vehicle speed > this threshold
                        are used in the optimisation [knots].

    Returns:
        Tuple (best_offset [°], offsets_array, mean_speeds_array)
    """
    off0, off1, doff = off_range
    offsets = np.arange(off0, off1 + doff, doff)
    noff    = len(offsets)

    # Indices where vehicle is moving fast enough
    spds    = np.array([r.spd for r in gprmc_records])
    movpos  = np.where(spds > spd_thresh_kts)[0]

    if len(movpos) == 0:
        print("  <!!!!> Cannot calculate wind direction offset: "
              "no records with vehicle speed > {:.0f} kts".format(spd_thresh_kts))
        return np.nan, offsets, np.full(noff, np.nan)

    mean_spd = np.zeros(noff)
    for i, off in enumerate(offsets):
        speeds = []
        for j in movpos:
            g   = gprmc_records[j]
            raw = raw_records[j]
            if (g.track == ERRVAL or g.spd == ERRVAL
                    or raw.wnddirraw == ERRVAL or raw.wndspdraw == ERRVAL):
                continue
            ws, *_ = wind_from_raw(g.track, g.spd, raw.wnddirraw,
                                   raw.wndspdraw, off)
            speeds.append(ws)
        mean_spd[i] = np.mean(speeds) if speeds else np.nan

    best_idx = int(np.nanargmin(mean_spd))
    best_off = offsets[best_idx]

    print(f"  <--> Minimum mean speed {mean_spd[best_idx]:.2f} m/s "
          f"at offset {best_off:.1f}°")
    print(f"  <--> Original offset used: {raw_records[0].wnddiroff:.1f}°")

    return best_off, offsets, mean_spd
