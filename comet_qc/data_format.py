"""
Data format utilities.
Replaces: comet_get_format.pro, tag_rename.pro
"""
from pathlib import Path
import pandas as pd


def get_format(fileformat: dict, format_file: str) -> pd.DataFrame:
    """
    Read CoMeT_data_format.csv and return the rows matching *fileformat*.

    Args:
        fileformat: dict with keys 'comet' (str or int), 'version' (str), 'qc' (str).
        format_file: Path to CoMeT_data_format.csv.

    Returns:
        DataFrame whose rows correspond to the fields for the matched format
        and whose columns are the spreadsheet headers (lower-cased).

    Raises:
        FileNotFoundError: if format_file does not exist.
        ValueError: if no matching rows are found.
    """
    path = Path(format_file)
    if not path.exists():
        raise FileNotFoundError(f"Data format file not found: {format_file}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Fill NaN in string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('')

    comet_str   = str(fileformat['comet']).strip()
    version_str = str(fileformat['version']).strip()
    qc_str      = str(fileformat['qc']).strip()

    mask = (
        df['comet'].astype(str).str.strip() == comet_str
    ) & (
        df['version'].astype(str).str.strip() == version_str
    ) & (
        df['qclevel'].astype(str).str.strip() == qc_str
    )

    result = df[mask].copy().reset_index(drop=True)

    if result.empty:
        raise ValueError(
            f"No records in data format file matching: "
            f"CoMeT={comet_str!r}, version={version_str!r}, qc={qc_str!r}\n"
            f"File: {format_file}"
        )

    # Ensure numeric columns have correct types
    for col in ('mult_fact', 'add_fact', 'headerlines'):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce').fillna(
                1.0 if col == 'mult_fact' else 0.0)

    return result


def instrument_string(fmt_row: pd.Series) -> str:
    """
    Build a combined instrument string from instrument1..instrumentN columns.
    """
    parts = []
    for col in fmt_row.index:
        if col.startswith('instrument') and not col.startswith('instrument') is False:
            pass
        if col.startswith('instrument'):
            val = str(fmt_row[col]).strip()
            if val and val not in ('', 'nan'):
                parts.append(val)
    return ', '.join(parts)
