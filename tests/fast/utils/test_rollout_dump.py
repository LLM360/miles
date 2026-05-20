from dataclasses import fields, is_dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from miles.utils import rollout_dump
from miles.utils.rollout_dump import load_rollout_dump, save_rollout_dump
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import AdapterRef, RewardSpec, Sample


def assert_same(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_same(a, b)
    elif is_dataclass(expected):
        for field in fields(expected):
            assert_same(getattr(actual, field.name), getattr(expected, field.name))
    else:
        assert actual == expected


@pytest.mark.parametrize("suffix", ["pt", "parquet"])
def test_full_sample_types_and_metadata_round_trip(tmp_path, suffix):
    sample = Sample(
        index=3,
        tokens=[10, 11, 12],
        response_length=2,
        loss_mask=[1, 0],
        reward=1.5,
        metadata={"empty": {}, "tuple": (1, "x"), "bytes": b"bytes"},
        rollout_routed_experts=np.arange(12, dtype=np.int32).reshape(2, 3, 2),
        rollout_indexer_topk=np.arange(8, dtype=np.int64).reshape(2, 2, 2),
        multimodal_train_inputs={"pixels": torch.arange(8, dtype=torch.float16).reshape(2, 4)},
        rollout_sampling_mask=RolloutSamplingMask.from_mask_list([[11, 9], [12]]),
        adapter=AdapterRef(name="a", slot=2),
        reward_spec=RewardSpec(rm_type="math"),
    )
    sample.extension = {"scalar": np.float32(0.25), "empty_array": np.empty((0, 2), dtype=np.int16)}
    other = Sample(index=4, tokens=[1], metadata={})
    payload = dict(
        rollout_id=7,
        metadata={"prompt_group_sizes": [2], "other": (2, {})},
        samples=[sample.to_dict(), other.to_dict()],
    )
    path = tmp_path / f"7.{suffix}"
    save_rollout_dump(path, payload)
    result = load_rollout_dump(path, map_location="cpu")
    assert_same(result, payload)
    assert_same(Sample.from_dict(result["samples"][0]).to_dict(), sample.to_dict())
    if suffix == "parquet":
        table = pq.read_table(path)
        assert table.num_rows == 2
        assert table["tokens"].to_pylist() == [[10, 11, 12], [1]]
        assert pa.types.is_integer(table["index"].type)


@pytest.mark.parametrize(
    "samples", [[], [{}], [{}, {}], [{"x": 1}, {"x": 1.5}], [{"x": []}, {"x": [None]}], [{"x": 2**90}, {"x": None}]]
)
def test_parquet_sparse_empty_and_mixed_rows(tmp_path, samples):
    payload = dict(rollout_id=1, metadata={}, samples=samples)
    path = tmp_path / "1.parquet"
    save_rollout_dump(path, payload)
    assert_same(load_rollout_dump(path), payload)


def test_load_legacy_stable_parquet(tmp_path):
    # This schema matches the original writer, with no version or batch metadata.
    row = Sample(index=2, tokens=[1, 2], response_length=1, reward=2.0).to_dict()
    row["metadata"] = {"source": "legacy"}
    row["rollout_routed_experts"] = [[[1, 2], [3, 4]]]
    table = pa.Table.from_pylist([row]).replace_schema_metadata({b"rollout_id": b"4"})
    path = tmp_path / "4.parquet"
    pq.write_table(table, path)
    payload = load_rollout_dump(path)
    assert payload["rollout_id"] == 4 and payload["metadata"] == {}
    sample = Sample.from_dict(payload["samples"][0])
    assert sample.index == 2 and sample.tokens == [1, 2] and sample.metadata == {"source": "legacy"}
    np.testing.assert_array_equal(sample.rollout_routed_experts, np.array([[[1, 2], [3, 4]]], dtype=np.int32))


@pytest.mark.parametrize("suffix", ["pt", "parquet"])
def test_failed_write_preserves_existing_file_and_removes_temporary(tmp_path, monkeypatch, suffix):
    path = tmp_path / f"1.{suffix}"
    path.write_bytes(b"previous file")

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(rollout_dump, "_save_parquet", fail)
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        save_rollout_dump(path, dict(rollout_id=1, samples=[]))
    assert path.read_bytes() == b"previous file"
    assert list(tmp_path.iterdir()) == [path]
