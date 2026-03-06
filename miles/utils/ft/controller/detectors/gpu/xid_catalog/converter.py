# /// script
# requires-python = ">=3.10"
# dependencies = ["openpyxl>=3.1", "typer>=0.9"]
# ///
"""Convert NVIDIA Xid-Catalog.xlsx to CSV files.

Usage:
    uv run converter.py /path/to/Xid-Catalog.xlsx [--output-dir DIR]

Downloads the xlsx from:
    https://docs.nvidia.com/deploy/xid-errors/Xid-Catalog.xlsx
"""
import csv
import logging
from pathlib import Path
from typing import Annotated

import openpyxl
import typer

logger = logging.getLogger(__name__)

SHEET_TO_FILENAME: dict[str, str] = {
    "Xids": "xids.csv",
    "Xid 144-150 Decode": "xid_144_150_decode.csv",
    "Resolution Buckets": "resolution_buckets.csv",
}


def _sanitize_header(value: object) -> str:
    """Normalize multi-line xlsx headers into clean single-line snake_case."""
    s = str(value).strip().replace("\n", " ").replace("\r", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def _sanitize_cell(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s


def _convert_sheet(
    workbook: openpyxl.Workbook,
    sheet_name: str,
    output_path: Path,
) -> int:
    ws = workbook[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        logger.warning("Sheet %r is empty, skipping", sheet_name)
        return 0

    header = [_sanitize_header(h) for h in rows[0]]
    data_rows = rows[1:]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data_rows:
            writer.writerow([_sanitize_cell(cell) for cell in row])

    return len(data_rows)


def main(
    xlsx_path: Annotated[Path, typer.Argument(help="Path to Xid-Catalog.xlsx")],
    output_dir: Annotated[
        Path, typer.Option(help="Output directory for CSV files")
    ] = Path(__file__).parent,
) -> None:
    if not xlsx_path.exists():
        raise typer.BadParameter(f"File not found: {xlsx_path}")

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    available_sheets = set(wb.sheetnames)

    output_dir.mkdir(parents=True, exist_ok=True)

    for sheet_name, csv_filename in SHEET_TO_FILENAME.items():
        if sheet_name not in available_sheets:
            logger.warning("Sheet %r not found in workbook, skipping", sheet_name)
            continue

        output_path = output_dir / csv_filename
        count = _convert_sheet(wb, sheet_name, output_path)
        print(f"{sheet_name} -> {output_path} ({count} rows)")

    wb.close()


if __name__ == "__main__":
    typer.run(main)
