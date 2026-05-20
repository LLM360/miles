import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import train_data_conversion as conversion
from miles.ray.rollout.diagnostics import _estimate_payload_bytes
from miles.utils.ray_utils import Box


@pytest.mark.parametrize("scheduled,multi_lora", [(False, False), (True, False), (False, True)])
def test_diagnostics_preserve_partition_payloads_and_store_refs(monkeypatch, caplog, scheduled, multi_lora):
    args = make_args(
        balance_data=False,
        global_batch_size=4,
        use_dynamic_batch_size=False,
        micro_batch_size=1,
        multi_lora_config="unused" if multi_lora else None,
    )
    # Match the current configuration gate without initializing external LoRA state.
    monkeypatch.setattr(conversion, "is_multi_lora_enabled", lambda args: multi_lora)
    data = dict(
        tokens=[[1] * n for n in (3, 5, 7, 9)],
        response_lengths=[1, 2, 3, 4],
        rewards=[1.0, 0.5, -0.5, -1.0],
        loss_masks=[[1] * n for n in (1, 2, 3, 4)],
        rollout_ids=[0, 1, 2, 3],
    )
    if multi_lora:
        data["adapter_slots"] = [1, 0, 0, 1]
    config = dict(dp_size=2)
    if scheduled:
        config.update(
            cp_size=1,
            tp_size=1,
            pp_size=1,
            vpp_size=None,
            micro_batch_size=1,
            use_dynamic_batch_size=False,
            max_tokens_per_gpu=None,
            microbatch_group_size_per_vp_stage=None,
        )
    use_schedule = conversion.can_schedule_on_rollout_side(args, data, config)
    assert use_schedule == scheduled
    expected = (
        conversion.split_train_data_by_dp_scheduled_raw(args, copy.deepcopy(data), train_parallel_config=config)
        if scheduled
        else conversion.split_train_data_by_dp_raw(args, copy.deepcopy(data), dp_size=2)
    )
    stored, references = [], []

    def put(*, value, value_spec):
        assert value_spec == conversion.ROLLOUT_DATA_VALUE_SPEC
        stored.append(value)
        # Mooncake-shaped ref has no hex() method.
        ref = Box({"object_id": f"shard-{len(stored)}"})
        references.append(ref)
        return ref

    monkeypatch.setattr(conversion.object_store, "get_instance", lambda: SimpleNamespace(put=put))
    refs = conversion.split_train_data_by_dp(args, data, config)
    assert stored == expected
    assert all(ref is expected_ref for ref, expected_ref in zip(refs, references, strict=True))
    assert caplog.text.count("ROLLOUT_DP_SHARD") == 2 and "ROLLOUT_DP_IMBALANCE" in caplog.text
    assert "total_tokens=24" in caplog.text


def test_payload_estimate_counts_arrays_tensors_and_handles_cycles():
    array = np.zeros((2, 3, 4), dtype=np.int32)
    tensor = torch.zeros((2, 3), dtype=torch.float16)
    values = [array, tensor]
    values.append(values)
    assert _estimate_payload_bytes(values) == array.nbytes + tensor.numel() * tensor.element_size()
