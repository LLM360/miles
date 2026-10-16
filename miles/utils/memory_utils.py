import gc
import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def release_cache_for_large_backward(logits: torch.Tensor) -> bool:
    """Release unused forward buffers before large dense logit gradients.

    Long packed vocabulary logits can need another full-sized gradient plus an
    accumulation buffer. Synchronize pending frees before releasing cache; no
    live tensors or autograd state are modified.
    """
    size = logits.numel() * logits.element_size()
    if not logits.is_cuda or not logits.requires_grad or size < 1024**3:
        return False
    device = logits.device
    free, _ = torch.cuda.mem_get_info(device)
    unused = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    if free >= 2 * size + 1024**3 or unused <= size:
        return False
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    logger.info(
        "Released unused cache before large-logit backward: logits_bytes=%d "
        "free_before=%d free_after=%d",
        size, free, torch.cuda.mem_get_info(device)[0],
    )
    return True


def available_memory():
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
