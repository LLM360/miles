# /// script
# requires-python = ">=3.10"
# dependencies = ["openpyxl>=3.1", "typer>=0.9"]
# ///
"""Convert NVIDIA Xid-Catalog.xlsx into info.py with NON_AUTO_RECOVERABLE_XIDS frozenset.

Usage:
    uv run converter.py /path/to/Xid-Catalog.xlsx [--output-dir DIR]

The xlsx can be downloaded from:
    https://docs.nvidia.com/deploy/xid-errors/Xid-Catalog.xlsx
with doc at:
    https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html
"""
import logging
from pathlib import Path
from typing import Annotated

import openpyxl
import typer

logger = logging.getLogger(__name__)

# Every resolution bucket from the xlsx must appear in exactly one of these two
# sets. This makes it easy to audit: if NVIDIA adds a new bucket in a future
# xlsx release, the assert below will fail loudly, forcing a human to classify it.
NON_AUTO_RECOVERABLE_BUCKETS: frozenset[str] = frozenset({
    "CONTACT_SUPPORT",
    "RESET_GPU",
    "RESTART_BM",
    "UPDATE_SWFW",
    "WORKFLOW_XID_48",
    "WORKFLOW_NVLINK_ERR",
    "WORKFLOW_NVLINK5_ERR",
})

AUTO_RECOVERABLE_BUCKETS: frozenset[str | None] = frozenset({
    "CHECK_MECHANICALS",
    "CHECK_UVM",
    "IGNORE",
    "RESTART_APP",
    "RESTART_VM",
    "WORKFLOW_XID_45",
    "XID_154",
    None,  # Not yet classified in xlsx (XIDs 162-172) — review manually
})


def main(
    xlsx_path: Annotated[Path, typer.Argument(help="Path to Xid-Catalog.xlsx")],
    output_dir: Annotated[
        Path, typer.Option(help="Output directory for info.py")
    ] = Path(__file__).parent,
) -> None:
    if not xlsx_path.exists():
        raise typer.BadParameter(f"File not found: {xlsx_path}")

    non_auto_recoverable, unclassified = _extract_xids(xlsx_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "info.py"
    output_path.write_text(_generate_info_py(non_auto_recoverable), encoding="utf-8")

    print(f"Found {len(non_auto_recoverable)} non-auto-recoverable XIDs from {xlsx_path.name}")
    for code, mnemonic, bucket in non_auto_recoverable:
        print(f"  XID {code:3d}  {bucket:<25s}  {mnemonic}")

    if unclassified:
        print(f"\nWARNING: {len(unclassified)} XIDs have no resolution bucket (review manually):")
        for code, mnemonic, _ in unclassified:
            print(f"  XID {code:3d}  {mnemonic}")

    print(f"\nWritten to {output_path}")


def _extract_xids(
    xlsx_path: Path,
) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]:
    """Return (non_auto_recoverable, unclassified), each a sorted list of (code, mnemonic, bucket)."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["Xids"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    header = [str(h).strip() for h in rows[0]]
    code_idx = header.index("Code")
    mnemonic_idx = header.index("Mnemonic")
    bucket_idx = next(i for i, h in enumerate(header) if "Immediate Action" in h)

    all_known_buckets = NON_AUTO_RECOVERABLE_BUCKETS | AUTO_RECOVERABLE_BUCKETS
    non_auto_recoverable: list[tuple[int, str, str]] = []
    unclassified: list[tuple[int, str, str]] = []

    for row in rows[1:]:
        raw_bucket = str(row[bucket_idx]).strip() if row[bucket_idx] else None
        assert raw_bucket in all_known_buckets, (
            f"Unknown resolution bucket {raw_bucket!r} for XID {row[code_idx]}. "
            f"Add it to NON_AUTO_RECOVERABLE_BUCKETS or AUTO_RECOVERABLE_BUCKETS in converter.py."
        )

        code = int(row[code_idx])
        mnemonic = str(row[mnemonic_idx]).strip()

        if raw_bucket in NON_AUTO_RECOVERABLE_BUCKETS:
            non_auto_recoverable.append((code, mnemonic, raw_bucket))
        elif raw_bucket is None:
            unclassified.append((code, mnemonic, ""))

    non_auto_recoverable.sort(key=lambda x: x[0])
    unclassified.sort(key=lambda x: x[0])
    return non_auto_recoverable, unclassified


def _generate_info_py(xids: list[tuple[int, str, str]]) -> str:
    lines: list[str] = [
        '"""NVIDIA XID codes that will NOT auto-recover — requires GPU reset, node reboot, or RMA.',
        "",
        "Auto-generated from Xid-Catalog.xlsx by converter.py.",
        "Source: https://docs.nvidia.com/deploy/xid-errors/Xid-Catalog.xlsx",
        '"""',
        "",
        "NON_AUTO_RECOVERABLE_XIDS: frozenset[int] = frozenset({",
    ]
    for code, mnemonic, bucket in xids:
        lines.append(f"    {code},  # {mnemonic}, {bucket}")
    lines.append("})")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    typer.run(main)
