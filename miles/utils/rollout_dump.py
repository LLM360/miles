"""Full rollout dumps in torch or Parquet format.

Parquet keeps homogeneous primitive/list columns queryable. Structured Python
values use per-cell torch serialization, recorded in schema metadata, so empty
dicts, tensors, ndarray dtypes/shapes and extension fields round-trip losslessly.
Like .pt dumps, these files must come from a trusted producer.
"""

import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

_FORMAT_KEY = b"miles.rollout_dump.version"


def find_rollout_dump(path: Path) -> Path:
    """Resolve a template across format changes; newest wins if both exist."""
    candidates = {path, path.with_suffix(".pt"), path.with_suffix(".parquet")}
    existing = [candidate for candidate in candidates if candidate.is_file()]
    return max(existing, key=lambda p: (p.stat().st_mtime_ns, p.suffix)) if existing else path


def save_rollout_dump(path: Path, payload: dict) -> None:
    """Publish a complete dump atomically before retention is allowed to run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        if path.suffix == ".parquet":
            _save_parquet(Path(temporary), payload)
        else:
            torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_rollout_dump(path: Path, *, map_location=None, mmap: bool = False) -> dict:
    path = Path(path)
    if path.suffix != ".parquet":
        return torch.load(path, weights_only=False, map_location=map_location, mmap=mmap)
    # Parquet is optional for the default torch format.
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    if _FORMAT_KEY not in metadata:
        # Original stable dumps contained plain Arrow rows and only rollout_id.
        rows = table.to_pylist()
        for row in rows:
            for field in ("rollout_routed_experts", "rollout_indexer_topk"):
                if row.get(field) is not None:
                    row[field] = np.asarray(row[field], dtype=np.int32)
        return dict(rollout_id=int(metadata[b"rollout_id"]), samples=rows, metadata={})
    if metadata[_FORMAT_KEY] != b"1":
        raise ValueError(f"Unsupported rollout Parquet version: {metadata[_FORMAT_KEY]!r}")
    payload = _decode(metadata[b"miles.payload"], map_location=map_location)
    rows = table.to_pylist() if table.num_columns else [{} for _ in range(int(metadata[b"miles.rows"]))]
    for column in json.loads(metadata[b"miles.encoded"]):
        for row in rows:
            row[column] = _decode(row[column], map_location=map_location)
    for column, indices in json.loads(metadata[b"miles.missing"]).items():
        for index in indices:
            rows[index].pop(column)
    return dict(payload, samples=rows)


def _encode(value) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _decode(value: bytes, *, map_location=None):
    return torch.load(io.BytesIO(value), weights_only=False, map_location=map_location)


def _native_type(values, pa):
    values = [value for value in values if value is not None]
    if not values:
        return pa.null()
    kind = type(values[0])
    if any(type(value) is not kind for value in values):
        raise TypeError("Mixed Python types need lossless encoding")
    primitives = {bool: pa.bool_(), int: pa.int64(), float: pa.float64(), str: pa.string(), bytes: pa.binary()}
    if kind in primitives:
        return primitives[kind]
    if kind is list:
        return pa.list_(_native_type([item for value in values for item in value], pa))
    raise TypeError(f"{kind.__name__} needs lossless encoding")


def _save_parquet(path: Path, payload: dict) -> None:
    # Deferred so .pt users do not need pyarrow installed.
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = payload["samples"]
    columns = dict.fromkeys(key for row in rows for key in row)
    encoded, missing = [], {}
    for key in columns:
        values = [row.get(key) for row in rows]
        try:
            columns[key] = pa.array(values, type=_native_type(values, pa))
        except (TypeError, OverflowError, pa.ArrowException):
            columns[key] = pa.array([_encode(value) for value in values], type=pa.binary())
            encoded.append(key)
        absent = [index for index, row in enumerate(rows) if key not in row]
        if absent:
            missing[key] = absent
    table = pa.table(columns).replace_schema_metadata(
        {
            _FORMAT_KEY: b"1",
            b"rollout_id": str(payload["rollout_id"]).encode(),
            b"miles.payload": _encode({key: value for key, value in payload.items() if key != "samples"}),
            b"miles.encoded": json.dumps(encoded).encode(),
            b"miles.missing": json.dumps(missing).encode(),
            b"miles.rows": str(len(rows)).encode(),
        }
    )
    pq.write_table(table, path, compression="snappy")
