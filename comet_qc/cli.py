"""
Command-line interface for the CoMeT QC toolkit.

Usage
-----
QC all un-QC'd files in a directory and write one merged NetCDF per
date + vehicle:

    comet-qc process \
        --catalog /data/comet_qc_2024.csv \
        --format  /data/CoMeT_data_format.csv \
        --dir     "/data/MITTEN-CI/20240712/CoMeT-alpha/Original data/"

The three arguments are all that is required.  The tool will:
  1. Find every CoMeT*_full_*.txt (and IMeT*_full_*.txt) file in --dir.
  2. Look each file up in the catalog to retrieve its QC flags.
  3. Apply QC and write QC'd ASCII files to the output directory.
  4. Merge all files for the same date+vehicle into one NetCDF named
       CoMeT<id>_full_<YYYYMMDD>_<HHMM>_<flags>.nc
     in a "QC'd data" sub-directory next to --dir (or --outdir if given).
"""
import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog='comet-qc',
        description='CoMeT data quality control and NetCDF conversion toolkit.',
    )
    sub = root.add_subparsers(dest='command', required=True)

    # -----------------------------------------------------------------------
    # process sub-command  (QC + merge in one shot)
    # -----------------------------------------------------------------------
    proc = sub.add_parser(
        'process',
        help='QC all files in a directory and write a merged NetCDF.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    proc.add_argument(
        '--catalog', required=True,
        help='Path to the QC catalog CSV (e.g. comet_qc_2024.csv).')
    proc.add_argument(
        '--format', required=True, dest='format_file',
        help='Path to CoMeT_data_format.csv.')
    proc.add_argument(
        '--dir', required=True, dest='input_dir',
        help=(
            "Directory that contains the un-QC'd full (and raw) .txt files. "
            "QC'd ASCII output goes into a sibling 'QC'd data' folder; "
            "the merged NetCDF is written there as well."))
    proc.add_argument(
        '--nodump', action='store_true',
        help='Run QC but do not write output files (dry run).')
    proc.add_argument(
        '--skipqcd', action='store_true',
        help="Skip files already marked as QC'd in the catalog without prompting.")
    proc.add_argument(
        '--outdir',
        help=(
            "Override the output directory for QC'd ASCII and NetCDF files. "
            "Defaults to a sibling 'QC'd data' folder next to --dir."))

    return root


def main(argv=None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    if args.command == 'process':
        from .qc import run_qc_dir
        run_qc_dir(
            catalog_file = args.catalog,
            format_file  = args.format_file,
            input_dir    = args.input_dir,
            output_dir   = args.outdir,
            nodump       = args.nodump,
            skipqcd      = args.skipqcd,
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
