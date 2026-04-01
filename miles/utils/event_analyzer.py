"""Post-hoc checker for cross-replica weight checksum consistency."""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from miles.backends.megatron_utils.weight_checksum import WeightChecksumEntry
from miles.utils.pydantic_utils import StrictBaseModel

logger = logging.getLogger(__name__)


class ChecksumMismatch(StrictBaseModel):
    step: int
    tensor_category: str
    tensor_name: str
    cell_indices: list[int]
    hashes: list[str]


def _parse_step_and_rank(file_path: Path) -> tuple[int, int]:
    """Extract step and rank from a checksum file path like step_0000042/rank_0003.json."""
    step = int(file_path.parent.name.removeprefix("step_"))
    rank = int(file_path.stem.removeprefix("rank_"))
    return step, rank


def _find_mismatches_in_group(
    step: int,
    category: str,
    entries: list[tuple[int, dict[str, str]]],
) -> list[ChecksumMismatch]:
    """Compare hash dicts across replicas for a single (step, category) group."""
    mismatches: list[ChecksumMismatch] = []

    all_keys: set[str] = set()
    for _, hashes in entries:
        all_keys.update(hashes.keys())

    for key in sorted(all_keys):
        hash_by_rank: dict[str, list[int]] = defaultdict(list)
        for rank, hashes in entries:
            h = hashes.get(key, "<missing>")
            hash_by_rank[h].append(rank)

        if len(hash_by_rank) > 1:
            cell_indices: list[int] = []
            hash_values: list[str] = []
            for h, ranks in sorted(hash_by_rank.items(), key=lambda x: x[1][0]):
                for r in ranks:
                    cell_indices.append(r)
                    hash_values.append(h)

            mismatches.append(
                ChecksumMismatch(
                    step=step,
                    tensor_category=category,
                    tensor_name=key,
                    cell_indices=cell_indices,
                    hashes=hash_values,
                )
            )

    return mismatches


def check_weight_checksums(checksum_dir: Path) -> list[ChecksumMismatch]:
    """Read all checksum dump files and verify cross-replica consistency.

    Args:
        checksum_dir: Path to the weight_checksum output directory
            (the directory containing step_*/rank_*.json files).

    Returns:
        List of mismatches found. Empty list means all replicas match.
    """
    files = sorted(checksum_dir.glob("step_*/rank_*.json"))
    if not files:
        logger.warning("No checksum files found in %s", checksum_dir)
        return []

    # Group entries by step
    entries_by_step: dict[int, list[tuple[int, WeightChecksumEntry]]] = defaultdict(list)
    for file_path in files:
        step, rank = _parse_step_and_rank(file_path)
        raw = file_path.read_text()
        entry = WeightChecksumEntry.model_validate_json(raw)
        entries_by_step[step].append((rank, entry))

    all_mismatches: list[ChecksumMismatch] = []

    for step in sorted(entries_by_step.keys()):
        step_entries = entries_by_step[step]

        categories = [
            ("param", lambda e: e.param_hashes),
            ("buffer", lambda e: e.buffer_hashes),
            ("master_param", lambda e: e.master_param_hashes),
            ("optimizer_state", lambda e: e.optimizer_state_hashes),
        ]

        for category_name, accessor in categories:
            group = [(rank, accessor(entry)) for rank, entry in step_entries]
            mismatches = _find_mismatches_in_group(
                step=step,
                category=category_name,
                entries=group,
            )
            all_mismatches.extend(mismatches)

    return all_mismatches


def main(checksum_dir: Path) -> int:
    """Run the checker and print results. Returns 0 if all match, 1 if mismatches found."""
    mismatches = check_weight_checksums(checksum_dir)

    if not mismatches:
        print("All replicas match across all steps.")
        return 0

    print(f"Found {len(mismatches)} mismatch(es):\n")
    for m in mismatches:
        print(f"  Step {m.step} | {m.tensor_category} | {m.tensor_name}")
        for idx, h in zip(m.cell_indices, m.hashes, strict=True):
            print(f"    rank {idx}: {h}")
        print()

    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <checksum_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(checksum_dir=Path(sys.argv[1])))
