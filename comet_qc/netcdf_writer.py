"""
NetCDF writer for QC'd CoMeT data.
Replaces: comet_ascii_2_cfnetcdf_torus_v2.pro

Variable layout, units, and transformations match the IDL script exactly,
including the known IDL quirks:
  - dewpoint: written in Celsius (no +273.15), but units attribute says 'Kelvin'
  - mixing_ratio: written in g/g (0.001 × g/kg value), units='1'
  - pressure: converted hPa → Pa (×100)
  - temperatures: Celsius → Kelvin (+273.15)   [fast_temp and slow_temp only]

Requires netCDF4 (pip install netCDF4).
Falls back to scipy NetCDF3 if netCDF4 is unavailable.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

ERRVAL = -999.0

# ---------------------------------------------------------------------------
# Variable table: matches IDL comet_ascii_2_cfnetcdf_torus_v2.pro exactly.
# Each entry: (nc_name, df_column, dtype, units, standard_name, long_name,
#              source, transform_fn)
# transform_fn is applied to the raw column array (masked for errval).
# PLEASE NOTE: You may add or remove variables avaliable in the raw CoMeT data as needed.
# However, it your responsibility to ensure:
# 1) The data are avaliable
# 2) The QC process is applied correctly to your variables.
#
# It is probably better if you calculate any other variables from pre-QC'd data
# before you add more to the _VARS array.
# ---------------------------------------------------------------------------
_VARS = [
    # nc_name        df_col                 dt   units                                  std_name                           long_name                                            source              transform
    ('time',         'epoch_time',          'f8', 'seconds since 1970-01-01 00:00:00', 'time',                            'Seconds since 00:00:00, 01-01-1970',                 'Garmin GPS',        None),
    ('alt',          'altitude',            'f4', 'meters',                             'altitude',                        'Height above mean sea level',                        'Garmin GPS',        None),
    ('lat',          'latitude',            'f4', 'degrees_north',                      'latitude',                        'Latitude',                                           'Garmin GPS',        None),
    ('lon',          'longitude',           'f4', 'degrees_east',                       'longitude',                       'Longitude',                                          'Garmin GPS',        None),
    # IDL: tfast = data.temperature_fast + 273.15
    ('fast_temp',    'temperature_fast',    'f4', 'Kelvin',                             'air_temperature',                 'Air Temperature (Fast Response)',                     'Campbell Scientific 10922-L Thermistor', lambda x: x + 273.15),
    # IDL: tslow = data.temperature_slow + 273.15
    ('slow_temp',    'temperature_slow',    'f4', 'Kelvin',                             'air_temperature',                 'Air Temperature (Slow Response)',                     'Vaisala HMP155A-L-PT',      lambda x: x + 273.15),
    # IDL: pres = 100.*data.pressure   (hPa → Pa)
    ('pressure',     'pressure',            'f4', 'Pascals',                            'air_pressure',                    'Air Pressure',                                       'Vaisala PTB210 Barometer',  lambda x: 100.0 * x),
    ('logger_RH',    'rh_slow',             'f4', 'percent',                            'relative_humidity',               'Logger Relative Humidity',                           'Vaisala HMP155A-L-PT',      None),
    ('calc_corr_RH', 'rh_fast',             'f4', 'percent',                            'relative_humidity',               'Calculated Corrected Relative Humidity (using Fast Temperature)', '', None),
    ('wind_speed',   'wind_speed',          'f4', 'meters per second',                  'wind_speed',                      'Calculated Wind Speed',                              '',                  None),
    ('wind_dir',     'wind_direction',      'f4', 'degrees',                            'wind_from_direction',             'Calculated Wind From Direction',                     '',                  None),
    ('vehicle_dir',  'vehicle_heading',     'f4', 'degrees',                            '',                                'Mesonet Vehicle Heading',                            'Garmin GPS',        None),
    ('vehicle_speed', 'vehicle_speed',      'f4', 'm/s',                                '',                                'Mesonet Vehicle Speed',                              'Garmin GPS', None),
    # IDL: td = data.dewpoint  (NO +273.15 — IDL bug, replicated for compatibility)
    ('dewpoint',     'dewpoint',            'f4', 'Kelvin',                             'dew_point_temperature',           'Calculated Dew Point Temperature',                   '',                  None),
    # IDL: qv = 0.001*data.water_vapor_mixing_ratio  (g/kg → g/g = dimensionless)
    ('mixing_ratio', 'water_vapor_mixing_ratio', 'f4', '1',                            'humidity_mixing_ratio',           'Calculated Mixing Ratio',                            '',                  lambda x: 0.001 * x),
    ('theta',        'theta',               'f4', 'Kelvin',                             'air_potential_temperature',       'Calculated Potential Temperature (Theta)',            '',                  None),
    ('theta_v',      'theta_v',             'f4', 'Kelvin',                             'virtual_potential_temperature',   'Calculated Virtual Potential Temperature (Theta_V)', '',                  None),
    ('theta_e',      'theta_e',             'f4', 'Kelvin',                             'equivalent_potential_temperature','Calculated Equivalent Potential Temperature (Theta_E)', '',               None),
]


def _try_nc4(): # I'm tired boss. So this is a catch-all for those stuck with nc3
    try:
        import netCDF4 as nc4
        return nc4
    except ImportError:
        return None


def write_netcdf(
    df: pd.DataFrame,
    file_out: str,
    global_attrs: Dict[str, str],
    fileformat: dict,
    errval: float = ERRVAL,
) -> None:
    """
    Write a QC'd CoMeT DataFrame to a CF-compliant NetCDF file.

    Matches comet_ascii_2_cfnetcdf_torus_v2.pro exactly.

    Args:
        df:           DataFrame with CoMeT data (one row per record).
        file_out:     Output file path (*.nc).
        global_attrs: dict with keys 'title', 'institution', 'source', 'comment'.
        fileformat:   dict with 'comet', 'version', 'qc'.
        errval:       Fill value for missing data.
    """
    nc4 = _try_nc4()
    Path(file_out).parent.mkdir(parents=True, exist_ok=True)

    if nc4 is not None:
        _write_nc4(df, file_out, global_attrs, fileformat, errval, nc4)
    else:
        _write_scipy(df, file_out, global_attrs, fileformat, errval)


def _apply_transform(raw: np.ndarray, xform, errval: float) -> np.ndarray:
    """Apply transform to non-missing values only."""
    if xform is None:
        return raw.copy()
    out  = raw.copy()
    mask = (raw != errval) & np.isfinite(raw)
    out[mask] = xform(raw[mask])
    return out


def _write_nc4(df, file_out, global_attrs, fileformat, errval, nc4):
    n = len(df)
    with nc4.Dataset(file_out, 'w', format='NETCDF4') as ds:
        # Global attributes
        for k, v in global_attrs.items():
            setattr(ds, k, v)

        ds.createDimension('time_dim', n)

        for (nc_name, df_col, dtype, units, std_name, long_name,
             source, xform) in _VARS:
            if df_col not in df.columns:
                continue
            raw  = pd.to_numeric(df[df_col], errors='coerce').fillna(errval).values.astype(float)
            data = _apply_transform(raw, xform, errval)

            var  = ds.createVariable(nc_name, dtype, ('time_dim',),
                                     fill_value=float(errval))
            var[:] = data.astype(dtype)
            var.units     = units
            var.long_name = long_name
            if std_name:
                var.standard_name = std_name
            if source:
                var.source = source

        # Error flag string variable (present in QC'd files)
        if fileformat.get('qc') == 'qcd' and 'error_string' in df.columns:
            errid = ds.createVariable('error_flag', str, ('time_dim',))
            errid.long_name = 'Error string'
            errid[:] = np.array(df['error_string'].fillna('').values, dtype=object)

    print(f"  <--> NetCDF4 written: {file_out}")


def _write_scipy(df, file_out, global_attrs, fileformat, errval):
    try:
        from scipy.io import netcdf_file
    except ImportError:
        raise ImportError(
            "Neither netCDF4 nor scipy is available.  "
            "Install netCDF4:  pip install netCDF4")

    n = len(df)
    with netcdf_file(file_out, 'w') as ds:
        for k, v in global_attrs.items():
            setattr(ds, k, v)
        ds.createDimension('time_dim', n)

        for (nc_name, df_col, dtype, units, std_name, long_name,
             source, xform) in _VARS:
            if df_col not in df.columns:
                continue
            raw  = pd.to_numeric(df[df_col], errors='coerce').fillna(errval).values.astype(float)
            data = _apply_transform(raw, xform, errval).astype(np.float32)

            var  = ds.createVariable(nc_name, np.float32, ('time_dim',))
            var[:] = data
            var.units     = units
            var.long_name = long_name
            if std_name:
                var.standard_name = std_name

    print(f"  <--> NetCDF3 (scipy) written: {file_out}")
