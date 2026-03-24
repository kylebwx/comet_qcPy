"""
Pressure correction for flow-speed-dependent bias across the Gill pressure port.
Replaces: comet_pressure_corr.pro

Reference: S. Waugh correction cited in CoMeT README.
"""
import numpy as np


def pressure_correction(p_hPa, ws_raw_ms: float, port_type: str = 'gill') -> float:
    """
    Apply the Waugh flow-speed bias correction to raw pressure.

    p = p0 + a3*V^3 + a2*V^2 + a1*V

    where p0 is the uncorrected pressure [hPa] and V is the vehicle-relative
    anemometer wind speed [m/s].

    Args:
        p_hPa:      Uncorrected pressure [hPa] (scalar or array).
        ws_raw_ms:  Raw anemometer speed (vehicle-relative) [m/s].
        port_type:  'gill' (default) or 'alum'.

    Returns:
        Corrected pressure [hPa].
    """
    p_hPa    = np.asarray(p_hPa,    dtype=float)
    ws_raw_ms = np.asarray(ws_raw_ms, dtype=float)

    if port_type == 'gill':
        a3 =  5.0e-7
        a2 = -1.0e-3
        a1 = -6.0e-5
    else:  # 'alum'
        a3 =  3.5e-5
        a2 = -5.0e-4
        a1 =  4.5e-3

    return p_hPa + a3 * ws_raw_ms**3 + a2 * ws_raw_ms**2 + a1 * ws_raw_ms
