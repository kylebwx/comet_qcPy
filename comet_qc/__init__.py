"""
comet_qc – Python port of the CoMeT (Combined Mesonet and Tracker) QC pipeline.

Quick-start
-----------
From the command line::

    comet-qc qc \\
        --catalog /data/comet_qc_2024.csv \\
        --format-file /data/CoMeT_data_format.csv \\
        --dir-template "/data/[project]/[YYYYMMDD]/CoMeT-[*]/[type]/" \\
        --project MITTEN-CI --day 20240707 --comet 3

    comet-qc merge \\
        --catalog /data/comet_qc_2024.csv \\
        --format-file /data/CoMeT_data_format.csv \\
        --dir-template "/data/[project]/[YYYYMMDD]/CoMeT-[*]/[type]/" \\
        --project MITTEN-CI

Python API::

    from comet_qc.qc import run_qc
    run_qc(catalog_file=..., format_file=..., dir_template=..., project='MITTEN-CI')

    from comet_qc.read_ascii import read_comet_ascii
    df = read_comet_ascii('CoMeT3_full_20240707_1735.txt',
                          fileformat={'comet': '3', 'version': '2024', 'qc': 'orig'},
                          format_file='CoMeT_data_format.csv')
"""

__version__ = "1.1.2"
__author__  = "Dr. Adam Houston [University of Nebraska-Lincoln] / Kyle Brooks [Central Michigan University]"

from .epoch         import to_epoch, from_epoch, parse_date_time
from .thermo        import (sat_vap_pres, mixing_ratio, dewpoint,
                             potential_temp, virtual_potential_temp,
                             equivalent_potential_temp, recalc_thermo)
from .spike_filter  import spike_filter
from .pressure_corr import pressure_correction
from .data_format   import get_format
from .gps_parser    import (parse_gprmc, parse_gpgga,
                             read_comet1_gps_file, create_gps_records)
from .wind          import wind_from_raw, find_winddir_offset
from .read_raw      import get_raw, RawRecord
from .read_ascii    import read_comet_ascii
from .netcdf_writer import write_netcdf
from .qc            import run_qc
from .merge_netcdf  import run_merge_netcdf

__all__ = [
    "to_epoch", "from_epoch", "parse_date_time",
    "sat_vap_pres", "mixing_ratio", "dewpoint",
    "potential_temp", "virtual_potential_temp", "equivalent_potential_temp",
    "recalc_thermo",
    "spike_filter",
    "pressure_correction",
    "get_format",
    "parse_gprmc", "parse_gpgga", "read_comet1_gps_file", "create_gps_records",
    "wind_from_raw", "find_winddir_offset",
    "get_raw", "RawRecord",
    "read_comet_ascii",
    "write_netcdf",
    "run_qc",
    "run_merge_netcdf",
]
