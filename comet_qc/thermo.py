"""
Thermodynamic utility functions.
Replaces: sound_mixr.pro, sound_satvappres.pro, sound_td.pro,
          sound_thta.pro, sound_thte.pro, sound_thtv.pro

NOTE ON 'arm' vs 'amr':
The original IDL code has a typo – the August-Roche-Magnus method key is
spelled 'amr' in the function definition but called as 'arm' everywhere.
Because IDL's CASE falls to the ELSE branch for an unmatched key, all calls
using 'arm' actually execute the default (Tetens) formula.  This is replicated
here for bit-exact compatibility: the key 'arm' triggers the default formula.
Only the key 'amr' (exact IDL match) triggers the true ARM formula.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Saturation vapour pressure
# ---------------------------------------------------------------------------

def sat_vap_pres(T_K, method: str = 'default', phase: str = 'l') -> np.ndarray:
    """
    Saturation vapor pressure in Pascals.

    Args:
        T_K:    Temperature in Kelvin (scalar or array-like)
        method: 'bolton', 'amr' (August-Roche-Magnus), or anything else
                (defaults to Tetens formula).  Passing 'arm' triggers the
                default formula to match IDL behaviour – see module docstring.
        phase:  'l' (liquid, default) or 'i' (ice); only used in default path.

    Returns:
        Saturation vapour pressure [Pa] with same shape as T_K.
    """
    T_K = np.asarray(T_K, dtype=float)
    T_C = T_K - 273.15

    if method == 'bolton':
        a, b, c = 611.2, 17.67, 29.65
    elif method == 'amr':                   # True ARM formula (rarely used)
        a = 610.94
        b = 17.625
        c = 273.15 - 243.04               # ≈ 30.11
    else:                                   # 'arm', 'default', or anything else
        a = 610.78
        if phase == 'i':
            b, c = 21.875, 7.66
        else:
            b, c = 17.269, 35.86

    return a * np.exp(b * T_C / (T_K - c))


# ---------------------------------------------------------------------------
# Mixing ratio
# ---------------------------------------------------------------------------

def mixing_ratio(p_Pa, e_Pa) -> np.ndarray:
    """
    Water vapour mixing ratio in g/kg.

    Args:
        p_Pa: Total pressure [Pa]
        e_Pa: Vapour pressure [Pa]

    Returns:
        Mixing ratio [g/kg]
    """
    eps = 0.622
    p_Pa = np.asarray(p_Pa, dtype=float)
    e_Pa = np.asarray(e_Pa, dtype=float)
    return 1000.0 * eps * e_Pa / (p_Pa - e_Pa)


# ---------------------------------------------------------------------------
# Dew point
# ---------------------------------------------------------------------------

def dewpoint(RH, T_C) -> np.ndarray:
    """
    Dew-point temperature in °C from relative humidity and air temperature.
    Uses the August-Roche-Magnus approximation (A=17.625, B=243.04 °C).

    Args:
        RH:  Relative humidity [%]
        T_C: Air temperature [°C]

    Returns:
        Dew-point temperature [°C]
    """
    A = 17.625
    B = 243.04
    RH = np.asarray(RH, dtype=float)
    T_C = np.asarray(T_C, dtype=float)
    gma = np.log(0.01 * RH) + A * T_C / (B + T_C)
    return B * gma / (A - gma)


# ---------------------------------------------------------------------------
# Potential temperature
# ---------------------------------------------------------------------------

def potential_temp(T_K, p_Pa, qv_kgkg=0.0) -> np.ndarray:
    """
    Potential temperature in Kelvin.

    Args:
        T_K:      Temperature [K]
        p_Pa:     Pressure [Pa]
        qv_kgkg:  Mixing ratio [kg/kg] (default 0)

    Returns:
        Potential temperature [K]
    """
    Cpd = 1004.0
    Cpv = 1870.0
    p0  = 1.0e5
    Rd  = 287.0
    Rv  = Cpd - Rd

    T_K      = np.asarray(T_K,      dtype=float)
    p_Pa     = np.asarray(p_Pa,     dtype=float)
    qv_kgkg  = np.asarray(qv_kgkg,  dtype=float)

    expnt = (Rd + qv_kgkg * Rv) / (Cpd + qv_kgkg * Cpv)
    return T_K * (p0 / p_Pa) ** expnt


# ---------------------------------------------------------------------------
# Virtual potential temperature
# ---------------------------------------------------------------------------

def virtual_potential_temp(theta, qv_gkg) -> np.ndarray:
    """
    Virtual potential temperature in Kelvin.

    Args:
        theta:   Potential temperature [K]
        qv_gkg:  Mixing ratio [g/kg]

    Returns:
        Virtual potential temperature [K]
    """
    return np.asarray(theta, dtype=float) * (1.0 + 0.61 * 0.001 * np.asarray(qv_gkg, dtype=float))


# ---------------------------------------------------------------------------
# Equivalent potential temperature
# ---------------------------------------------------------------------------

def equivalent_potential_temp(T_K, p_Pa, RH) -> np.ndarray:
    """
    Equivalent potential temperature in Kelvin (Bolton 1980).

    Args:
        T_K:  Temperature [K]
        p_Pa: Pressure [Pa]
        RH:   Relative humidity [%]

    Returns:
        Equivalent potential temperature [K]
    """
    Rd = 287.0
    Cp = 1004.0
    kpa = Rd / Cp

    T_K  = np.asarray(T_K,  dtype=float)
    p_Pa = np.asarray(p_Pa, dtype=float)
    RH   = np.asarray(RH,   dtype=float)

    theta = potential_temp(T_K, p_Pa, 0.0)

    # Use 'arm' (→ default Tetens) to match IDL behaviour
    es = sat_vap_pres(T_K, 'arm')     # Pa
    e  = 0.01 * es * RH               # Pa  (es * RH/100)
    qv = 0.622 * e / p_Pa             # kg/kg

    T_lcl = 55.0 + 2840.0 / (3.5 * np.log(T_K) - np.log(0.01 * e) - 4.805)
    Tm    = theta * (T_K / theta) ** (kpa * qv)

    return Tm * np.exp(((3376.0 / T_lcl) - 2.54) * qv * (1.0 + 0.81 * qv))


# ---------------------------------------------------------------------------
# Convenience: recalculate all derived thermo variables
# ---------------------------------------------------------------------------

def recalc_thermo(temp_f_C, temp_s_C, rh_slow, pressure_hPa):
    """
    Recalculate derived thermodynamic quantities from base measurements.

    Returns dict with keys: rh_fast, dewpoint, water_vapor_mixing_ratio,
                             theta, theta_v, theta_e
    All returned values use the same units as the rest of the CoMeT data:
    rh [%], dewpoint [°C], mixing_ratio [g/kg], theta/theta_v/theta_e [K].
    """
    T_K  = temp_f_C + 273.15
    p_Pa = pressure_hPa * 100.0

    # Slow dew point → used for vapour pressure
    T_d_slow = dewpoint(rh_slow, temp_s_C)           # °C
    e_slow   = 0.01 * sat_vap_pres(T_d_slow + 273.15, 'arm')  # hPa
    es_fast  = 0.01 * sat_vap_pres(T_K,               'arm')  # hPa
    rh_fast  = 100.0 * e_slow / es_fast               # %

    td = dewpoint(rh_fast, temp_f_C)                  # °C
    e_f = 0.01 * sat_vap_pres(td + 273.15, 'arm')     # hPa
    qv  = mixing_ratio(pressure_hPa, e_f)             # g/kg

    theta   = potential_temp(T_K, p_Pa, 0.0)
    theta_v = virtual_potential_temp(theta, qv)
    theta_e = equivalent_potential_temp(T_K, p_Pa, rh_fast)

    return dict(
        rh_fast=rh_fast,
        dewpoint=td,
        water_vapor_mixing_ratio=qv,
        theta=theta,
        theta_v=theta_v,
        theta_e=theta_e,
    )
