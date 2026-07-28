import itertools
import json
import logging
import os
import random
import re

import numpy as np
import ray

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from miles.utils.types import MultimodalTypes, Sample

from .rollout_sharding import (
    ROLLOUT_DATA_REF_FORMAT,
    ROUTED_EXPERTS_SHARD_META_KEY,
    rollout_destination_key,
)
from .timer import Timer

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)


def read_file(path):
    path, row_slice = _parse_generalized_path(path)
    reader = None

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):

        def jsonl_reader(p):
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error at line {line_num}: {e}")
                        continue

        reader = jsonl_reader(path)

    elif path.endswith(".parquet"):
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")

        def parquet_reader(p):
            pf = pq.ParquetFile(p)

            for batch in pf.iter_batches():
                yield from batch.to_pylist()

        reader = parquet_reader(path)

    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:

        logger.info("read_file path=%s applying slice row_slice=%s", path, row_slice)
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    yield from reader


def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


def filter_long_prompt(origin_samples: list[Sample], tokenizer, processor, max_length: int | None) -> list[Sample]:
    if max_length is None:
        return False

    if not isinstance(origin_samples[0].prompt, str):
        logger.warning(
            "Skipping max_length check for list prompt. Set apply_chat_template=True to enable length filtering."
        )
        return False

    if processor:
        filtered_samples = []
        for sample in origin_samples:
            from miles.utils.processing_utils import process_vision_info

            multimodal_inputs = process_vision_info(sample.prompt, processor)
            processor_output = processor(text=sample.prompt, **multimodal_inputs)
            input_ids = processor_output["input_ids"][0]
            if len(input_ids) <= max_length:
                filtered_samples.append(sample)
    else:
        prompts = [sample.prompt for sample in origin_samples]
        input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        filtered_samples = [
            sample
            for sample, input_ids in zip(origin_samples, input_ids_list, strict=True)
            if len(input_ids) <= max_length
        ]

    logger.info(f"Filtered {len(origin_samples) - len(filtered_samples)} samples longer than max_length={max_length}.")

    return filtered_samples


def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        # If prompt is a string and we don't apply chat template, return the prompt as is.
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]

    if multimodal_keys:
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodals[mt.placeholder] = (mt, list(data.get(data_key)))

        pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

        for message in prompt:
            if isinstance(message["content"], str):
                content_list = []
                for segment in re.split(pattern, message["content"]):
                    if not segment:
                        continue
                    if segment in multimodals:
                        mt, content = multimodals[segment]
                        content_list.append({"type": mt.name, mt.name: content.pop(0)})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list

            elif isinstance(message["content"], list):
                # TODO: handle more general cases. where message['content'] is a dict and contains multiple types of content.
                # e.g.
                #  "content": [
                #     {
                #         "type": "image",
                #         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                #     },
                #     {"type": "text", "text": "Describe this image."},
                # ],
                logger.warning("message['content'] is a list of dicts, no processing will be done.")
                continue
            else:
                raise ValueError(
                    f"Unsupported content type: {type(message['content'])}, expected str or list of dicts"
                )

    return prompt


class Dataset:
    def __init__(
        self,
        path,
        tokenizer,
        processor,
        max_length,
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        seed=42,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
    ):
        origin_samples = []
        for data in read_file(path):
            # Both chat templates and multimodal inputs require conversation format (list of message dicts)
            as_conversation = apply_chat_template or (multimodal_keys is not None)
            prompt = _build_messages(data, prompt_key, as_conversation, multimodal_keys)

            metadata = data.get(metadata_key) or {}
            tools = None
            if tool_key is not None and tool_key in data:
                tools = data[tool_key]
                if isinstance(tools, str):
                    tools = json.loads(tools)
                elif isinstance(tools, np.ndarray):
                    tools = tools.tolist()
                assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
                metadata["tools"] = tools

            if apply_chat_template:
                output_prompt = tokenizer.apply_chat_template(
                    prompt,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(apply_chat_template_kwargs or {}),
                )
            else:
                output_prompt = prompt

            if processor:
                from miles.utils.processing_utils import process_vision_info

                assert isinstance(
                    prompt, list
                ), f"prompt must be a list when processor is not None, got {type(prompt)} instead"
                multimodal_inputs = process_vision_info(prompt, processor)
            else:
                multimodal_inputs = None

            origin_samples.append(
                Sample(
                    prompt=output_prompt,
                    label=data[label_key] if label_key is not None else None,
                    metadata=metadata,
                    multimodal_inputs=multimodal_inputs,
                )
            )

        if max_length is not None:
            self.origin_samples = filter_long_prompt(origin_samples, tokenizer, processor, max_length)
        else:
            self.origin_samples = origin_samples

        self.epoch_id = -1
        self.seed = seed
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    # use first fit to get the number of micro batches
    batches = []
    for length in total_lengths:
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                break
        else:
            batches.append(length)

    return len(batches)


def get_rollout_data_ref_fingerprint(
    rollout_data_ref,
    dp_rank,
    *,
    pp_rank=0,
    cp_rank=0,
    include_routed_experts=True,
):
    """Identify exactly the local Ray objects backing one actor's preload."""
    if isinstance(rollout_data_ref, dict) and rollout_data_ref.get("format") == ROLLOUT_DATA_REF_FORMAT:
        refs = [rollout_data_ref["base"][dp_rank].inner]
        if include_routed_experts:
            destination = rollout_destination_key(dp_rank, pp_rank, cp_rank)
            routed_experts_ref = rollout_data_ref.get("rollout_routed_experts", {}).get(destination)
            if routed_experts_ref is None:
                raise KeyError(f"missing rollout routing-replay shard for destination {destination}")
            refs.append(routed_experts_ref.inner)
    else:
        refs = [rollout_data_ref[dp_rank].inner]

    return ":".join(ref.hex() for ref in refs)


def process_rollout_data(
    args,
    rollout_data_ref,
    dp_rank,
    dp_size,
    *,
    pp_rank=0,
    pp_size=1,
    cp_rank=0,
    cp_size=1,
    include_routed_experts=True,
):
    if isinstance(rollout_data_ref, dict) and rollout_data_ref.get("format") == ROLLOUT_DATA_REF_FORMAT:
        base_refs = rollout_data_ref["base"]
        assert len(base_refs) == dp_size
        refs = [base_refs[dp_rank].inner]

        routed_experts_ref = None
        if include_routed_experts:
            destination = rollout_destination_key(dp_rank, pp_rank, cp_rank)
            routed_experts_ref = rollout_data_ref.get("rollout_routed_experts", {}).get(destination)
            if routed_experts_ref is None:
                raise KeyError(
                    f"missing rollout routing-replay shard for destination {destination}; "
                    f"available={sorted(rollout_data_ref.get('rollout_routed_experts', {}))}"
                )
            refs.append(routed_experts_ref.inner)

        fetched = ray.get(refs)
        rollout_data = fetched[0]
        if routed_experts_ref is not None:
            routed_experts = fetched[1]
            shard_metadata = routed_experts[ROUTED_EXPERTS_SHARD_META_KEY]
            if shard_metadata.get("version") != 1:
                raise ValueError(
                    f"unsupported routing-replay shard version {shard_metadata.get('version')}"
                )
            expected_destination = (dp_rank, pp_rank, cp_rank)
            actual_destination = tuple(
                shard_metadata[key] for key in ("dp_rank", "pp_rank", "cp_rank")
            )
            if actual_destination != expected_destination:
                raise ValueError(
                    f"routing-replay shard destination mismatch: expected {expected_destination}, "
                    f"got {actual_destination}"
                )
            if shard_metadata["pp_size"] != pp_size or shard_metadata["cp_size"] != cp_size:
                raise ValueError(
                    "routing-replay shard parallel-size mismatch: "
                    f"shard pp/cp=({shard_metadata['pp_size']}, {shard_metadata['cp_size']}), "
                    f"actor pp/cp=({pp_size}, {cp_size})"
                )
            if shard_metadata["qkv_format"] != args.qkv_format:
                raise ValueError(
                    f"routing-replay qkv_format mismatch: shard={shard_metadata['qkv_format']}, "
                    f"actor={args.qkv_format}"
                )
            if len(routed_experts["rollout_routed_experts"]) != len(rollout_data["tokens"]):
                raise ValueError(
                    "routing-replay shard sample count does not match base payload: "
                    f"{len(routed_experts['rollout_routed_experts'])} != {len(rollout_data['tokens'])}"
                )
            num_local_layers = len(shard_metadata["layer_indices"])
            topk = None
            for sample_idx, (sample_routing, tokens) in enumerate(
                zip(
                    routed_experts["rollout_routed_experts"],
                    rollout_data["tokens"],
                    strict=True,
                )
            ):
                num_tokens = len(tokens)
                if shard_metadata["qkv_format"] == "thd":
                    target_length = num_tokens
                else:
                    target_length = shard_metadata["max_seq_len"]
                    if target_length is None or target_length < num_tokens:
                        raise ValueError(
                            f"invalid BSHD routing-replay max_seq_len {target_length} "
                            f"for sample {sample_idx} with {num_tokens} tokens"
                        )
                if cp_size == 1:
                    expected_rows = target_length
                else:
                    chunk_size = (target_length + 2 * cp_size - 1) // (2 * cp_size)
                    expected_rows = 2 * chunk_size
                shape = getattr(sample_routing, "shape", None)
                if shape is None or len(shape) != 3:
                    raise ValueError(
                        f"routing-replay sample {sample_idx} must be rank 3, got shape={shape}"
                    )
                if shape[0] != expected_rows or shape[1] != num_local_layers:
                    raise ValueError(
                        f"routing-replay sample {sample_idx} shape mismatch: got {shape}, "
                        f"expected ({expected_rows}, {num_local_layers}, topk)"
                    )
                if topk is None:
                    topk = shape[2]
                elif shape[2] != topk:
                    raise ValueError(
                        f"routing-replay topk mismatch in sample {sample_idx}: {shape[2]} != {topk}"
                    )
            rollout_data["rollout_routed_experts"] = routed_experts["rollout_routed_experts"]
            rollout_data[ROUTED_EXPERTS_SHARD_META_KEY] = shard_metadata
    else:
        assert len(rollout_data_ref) == dp_size
        rollout_data = ray.get(rollout_data_ref[dp_rank].inner)
        if not include_routed_experts:
            rollout_data.pop("rollout_routed_experts", None)

    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
