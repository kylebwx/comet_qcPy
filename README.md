# comet-qc

Python port of the CoMeT (Combined Mesonet and Tracker) meteorological data quality-control pipeline, originally written in IDL by Adam Houston (UNL) and Jason Keeler (CMU).

## Requirements

| Package | Role |
|---------|------|
| `numpy` | Array math |
| `pandas` | Data I/O and tabular processing |
| `netCDF4` *(optional)* | NetCDF4 output (recommended) |
| `scipy` *(optional)* | Fallback NetCDF3 writer if netCDF4 is absent |

```bash
pip install numpy pandas
pip install netCDF4          # recommended for NetCDF output
```

## Installation

```bash
cd comet_qc/
pip install .
```

ONLY if the above is unsuccessful, try:

```bash
cd comet_qc/
pip install -e .
```

After installation the `comet-qc` command is available on your PATH.

## Command-line usage

There is a single command: `process`. Point it at the catalog, the format file, and the directory containing your un-QC'd files. That's it.

```bash
comet-qc process \
    --catalog /data/comet_qc_2024.csv \
    --format  /data/CoMeT_data_format.csv \
    --dir     "/data/MITTEN-CI/20240712/CoMeT-alpha/Original data/"
```

The tool will:
1. Find every `CoMeT*_full_*.txt` (and `IMeT*_full_*.txt`) file in `--dir`.
2. Look each file up in the catalog to get its QC flags.
3. Apply all QC corrections and write QC'd ASCII files.
4. Merge all files for the same date + vehicle into **one combined NetCDF**.

### Output location

QC'd ASCII files and the merged NetCDF are written to a sibling `QC'd data/`
folder next to `--dir`:

```
Original data/
    CoMeTalpha_full_20240712_1703.txt
    CoMeTalpha_raw_20240712_1703.txt
QC'd data/
    CoMeTalpha_full_20240712_1703_L2_P2_RH1_W2.txt
    CoMeTalpha_full_20240712_1703_P2_RH1_W2.nc      ← merged NetCDF
```

Use `--outdir /some/other/path` to write output somewhere else.

### NetCDF filename format

```
CoMeT<id>_full_<YYYYMMDD>_<HHMM>_<flags>.nc
```

For example:
```
CoMeTalpha_full_20240712_1703_P2_RH1_RH2_TF2_TS2_W2_W4.nc
CoMeT3_full_20240707_1735_P2_F1_W4.nc
```

### Optional flags

| Flag | Meaning |
|------|---------|
| `--nodump` | Dry-run: perform QC but don't write any output files |
| `--skipqcd` | Skip files already marked as QC'd in the catalog without prompting |
| `--outdir PATH` | Override the output directory |

### Example: dry run first

```bash
comet-qc process \
    --catalog /data/comet_qc_2024.csv \
    --format  /data/CoMeT_data_format.csv \
    --dir     "/data/MITTEN-CI/20240712/CoMeT-alpha/Original data/" \
    --nodump
```

## Python API

```python
from comet_qc.qc import run_qc_dir

run_qc_dir(
    catalog_file = 'comet_qc_2024.csv',
    format_file  = 'CoMeT_data_format.csv',
    input_dir    = '/data/MITTEN-CI/20240712/CoMeT-alpha/Original data/',
    skipqcd      = True,
)
```

You can also read individual files directly:

```python
from comet_qc.read_ascii import read_comet_ascii

df = read_comet_ascii(
    filepath   = 'CoMeTalpha_full_20240712_1703.txt',
    fileformat = {'comet': 'alpha', 'version': '2024', 'qc': 'orig'},
    format_file= 'CoMeT_data_format.csv',
)
```

## Module overview

| Module | Replaces (IDL) | Purpose |
|--------|---------------|---------|
| `epoch.py` | `epoch.pro`, `epoch2datetime.pro` | Unix epoch ↔ datetime |
| `thermo.py` | `sound_*.pro` | Saturation vapour pressure, mixing ratio, dewpoint, θ, θᵥ, θₑ |
| `spike_filter.py` | `spikefilter_new.pro` | Two-pass median/std spike removal |
| `pressure_corr.py` | `comet_pressure_corr.pro` | Gill/alum port flow-speed bias |
| `data_format.py` | `comet_get_format.pro`, `tag_rename.pro` | Read `CoMeT_data_format.csv` |
| `gps_parser.py` | `create_gps_struct.pro`, `read_comet1_ascii_gps.pro` | NMEA sentence parsers |
| `wind.py` | `get_comet_wind_from_raw.pro`, `comet_find_winddir_offset.pro` | Wind vector from anemometer + GPS |
| `read_raw.py` | `read_comet_raw.pro`, `get_raw.pro` | Raw Campbell logger file reader |
| `read_ascii.py` | `read_comet_ascii.pro` | Full (QC'd or original) ASCII file reader |
| `qc.py` | `comet_qc_2024.pro` | Full QC pipeline (all flags a1–w4) + `run_qc_dir` entry point |
| `netcdf_writer.py` | `comet_ascii_2_cfnetcdf_torus_v2.pro` | CF-compliant NetCDF writer |
| `cli.py` | *(new)* | `comet-qc process` command-line entry point |

## Directory layout expected on disk

```
<Project name>/
  <YYYYMMDD>/
    CoMeT-<label>/
      Original data/          ← point --dir here
        CoMeT<N>_full_<YYYYMMDD>_<HHMM>.txt
        CoMeT<N>_raw_<YYYYMMDD>_<HHMM>.txt
      QC'd data/              ← output goes here automatically
```

`<label>` is a number (`1`, `2`, `3`) or `alpha`.

## Error flag format

Each output record carries a per-instrument error flag string:

```
a##-g##-p##-tf##-ts##-rh##-w##-f##
```

Values accumulate bitwise: level 1 → +1, level 2 → +2, level 3 → +4, level 4 → +8.

## Important notes

### `arm` vs `amr` typo (IDL compatibility)

The IDL `sound_satvappres.pro` contains a typo: the August-Roche-Magnus
method key is defined as `'amr'` but all call sites pass `'arm'`. Because
IDL's `CASE` statement falls to `ELSE` for an unmatched key, every `'arm'`
call actually executes the default Tetens formula. **This behaviour is
intentionally replicated** in `thermo.py` for bit-exact output compatibility.

### NetCDF output

`netCDF4` (NETCDF4 format) is preferred. If it is not installed, the code
falls back to `scipy`'s legacy NetCDF3 writer automatically.

### Catalog update

After successfully QC-ing a file, the `QC TXT` column in the catalog CSV is
set to `1`. The CSV is rewritten in place.
