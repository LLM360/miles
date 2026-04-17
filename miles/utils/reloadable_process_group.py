import logging
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

from miles.utils.memory_utils import print_memory

logger = logging.getLogger(__name__)

old_new_group_dict = {}


def _maybe_wrap_alltoall_diag(func):
    """Wrap all_to_all_single to emit per-rank diagnostic info on RuntimeError.

    Activated via MILES_R3_DIAG_ALLTOALL=1. On crash, log: (a) the split sizes
    and tensor shapes (b) the calling frame's local torch.Tensor variables —
    this surfaces Megatron token_dispatcher's num_tokens_per_expert histogram,
    permutation order, probs tensor, etc., which tells us whether the bug is
    in Megatron's derivation of split sizes from (replayed) top_indices or
    in the replay manager's choice of top_indices itself.

    See radixark/miles#1002.
    """
    if os.environ.get("MILES_R3_DIAG_ALLTOALL", "0") != "1":
        return func

    import sys
    import traceback

    def _describe(val):
        if isinstance(val, torch.Tensor):
            info = f"Tensor shape={tuple(val.shape)} dtype={val.dtype}"
            if val.numel() > 0 and val.numel() <= 32:
                try:
                    info += f" values={val.tolist()}"
                except Exception:
                    pass
            elif val.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                try:
                    info += f" min={int(val.min())} max={int(val.max())}"
                    if val.numel() < 1_000_000:
                        flat = val.flatten()
                        unique = torch.unique(flat)
                        info += f" n_unique={unique.numel()}"
                        if unique.numel() <= 16:
                            info += f" unique={unique.tolist()}"
                except Exception:
                    pass
            return info
        if isinstance(val, (list, tuple)) and len(val) <= 32:
            return f"{type(val).__name__}({len(val)})={list(val)}"
        if isinstance(val, (int, float, bool, str)):
            return repr(val)
        return f"<{type(val).__name__}>"

    def _dump_caller_frames(msg_parts, depth=8):
        frame = sys._getframe(0)
        for i in range(depth + 3):
            if frame is None:
                return
            frame = frame.f_back
            if frame is None:
                return
            co = frame.f_code
            fname = co.co_filename
            lineno = frame.f_lineno
            func_name = co.co_name
            # Skip wrapper frames; focus on callers outside reloadable_process_group
            if "reloadable_process_group" in fname:
                continue
            # Only dump frames that look relevant to MoE / dispatch / replay.
            rel = any(kw in fname for kw in ("moe", "token_dispatcher", "mappings", "replay", "model_provider"))
            if not rel and i < 2:
                # Still worth showing the immediate caller once even if not matched.
                pass
            msg_parts.append(f"\n  === frame depth={i} {fname}:{lineno} in {func_name} ===\n")
            for name, val in frame.f_locals.items():
                if name.startswith("__"):
                    continue
                msg_parts.append(f"    {name}: {_describe(val)}\n")
            if rel and i >= 1:
                # One relevant frame is enough — bail after dumping it.
                break

    def diag(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            rank = dist.get_rank() if dist.is_initialized() else -1
            output_t = args[0] if len(args) >= 1 else kwargs.get("output")
            input_t = args[1] if len(args) >= 2 else kwargs.get("input")
            output_ss = args[2] if len(args) >= 3 else kwargs.get("output_split_sizes")
            input_ss = args[3] if len(args) >= 4 else kwargs.get("input_split_sizes")
            group = args[4] if len(args) >= 5 else kwargs.get("group")

            def _shape(t):
                return tuple(t.shape) if isinstance(t, torch.Tensor) else None

            def _dtype(t):
                return str(t.dtype) if isinstance(t, torch.Tensor) else None

            input_sum = sum(input_ss) if input_ss is not None else None
            output_sum = sum(output_ss) if output_ss is not None else None
            input_dim0 = _shape(input_t)[0] if _shape(input_t) else None
            output_dim0 = _shape(output_t)[0] if _shape(output_t) else None

            group_ranks = None
            try:
                group_ranks = dist.get_process_group_ranks(group) if group is not None else None
            except Exception:
                group_ranks = "<unavailable>"

            msg_parts = [
                f"[R3_DIAG rank={rank}] all_to_all_single RuntimeError: {e}\n",
                f"  input: shape={_shape(input_t)} dtype={_dtype(input_t)}\n",
                f"  output: shape={_shape(output_t)} dtype={_dtype(output_t)}\n",
                f"  input_split_sizes={input_ss} sum={input_sum} input.dim0={input_dim0}\n",
                f"  output_split_sizes={output_ss} sum={output_sum} output.dim0={output_dim0}\n",
                f"  group_ranks={group_ranks}\n",
                f"  input_sum_matches_dim0={input_sum == input_dim0 if input_sum is not None else 'n/a'}\n",
                f"  output_sum_matches_dim0={output_sum == output_dim0 if output_sum is not None else 'n/a'}\n",
                f"  call stack (top 6):\n",
                *["    " + line for line in traceback.format_stack()[-8:-2]],
            ]
            try:
                _dump_caller_frames(msg_parts, depth=6)
            except Exception as dump_err:  # diagnostic must not crash the crash
                msg_parts.append(f"  (frame dump failed: {dump_err!r})\n")
            logger.error("".join(msg_parts))
            raise

    return diag


def monkey_patch_torch_dist():
    pid = os.getpid()
    if pid in old_new_group_dict:
        assert dist.old_new_group == old_new_group_dict[pid]
        return

    logger.info("Applying monkey patch to torch.distributed")

    old_new_group = dist.new_group
    old_new_group_dict[pid] = old_new_group
    dist.old_new_group = old_new_group

    def new_group(*args, **kwargs):
        group = old_new_group(*args, **kwargs)
        # skip none nccl group.
        if len(args) >= 3 and args[2] == "gloo" or "backend" in kwargs and kwargs["backend"] == "gloo":
            return group

        # Get ranks from arguments
        if len(args) >= 1 and args[0] is not None:
            ranks = args[0]
        elif "ranks" in kwargs and kwargs["ranks"] is not None:
            ranks = kwargs["ranks"]
        else:
            # If no ranks specified, use all ranks in world
            ranks = list(range(dist.get_world_size()))

        if len(ranks) == 1:
            return group

        group = ReloadableProcessGroup(group, ranks)
        return group

    dist.new_group = new_group

    def get_new_function(func):
        def new_function(*args, **kwargs):
            args = tuple([arg.group if isinstance(arg, ReloadableProcessGroup) else arg for arg in args])
            kwargs = {k: (v.group if isinstance(v, ReloadableProcessGroup) else v) for k, v in kwargs.items()}
            with _wrap_low_level_call():
                return func(*args, **kwargs)

        return new_function

    dist.get_rank = get_new_function(dist.get_rank)
    dist.get_world_size = get_new_function(dist.get_world_size)
    dist.get_backend = get_new_function(dist.get_backend)
    dist.get_global_rank = get_new_function(dist.get_global_rank)
    dist.get_group_rank = get_new_function(dist.get_group_rank)
    dist.get_process_group_ranks = get_new_function(dist.get_process_group_ranks)

    dist.all_reduce = get_new_function(dist.all_reduce)
    dist.all_gather = get_new_function(dist.all_gather)
    dist.all_gather_into_tensor = get_new_function(dist.all_gather_into_tensor)
    dist.all_gather_object = get_new_function(dist.all_gather_object)
    dist.all_to_all = get_new_function(dist.all_to_all)
    dist.all_to_all_single = _maybe_wrap_alltoall_diag(get_new_function(dist.all_to_all_single))
    dist.broadcast = get_new_function(dist.broadcast)
    dist.reduce = get_new_function(dist.reduce)
    dist.reduce_scatter = get_new_function(dist.reduce_scatter)
    dist.reduce_scatter_tensor = get_new_function(dist.reduce_scatter_tensor)
    dist.scatter = get_new_function(dist.scatter)
    dist.gather = get_new_function(dist.gather)
    dist.barrier = get_new_function(dist.barrier)
    dist.send = get_new_function(dist.send)
    dist.recv = get_new_function(dist.recv)
    dist._coalescing_manager = get_new_function(dist._coalescing_manager)

    # p2p
    old_isend = dist.isend
    old_irecv = dist.irecv

    dist.isend = get_new_function(dist.isend)
    dist.irecv = get_new_function(dist.irecv)

    def get_new_p2pop_function(func):
        def new_function(*args, **kwargs):
            def convert(arg):
                if isinstance(arg, ReloadableProcessGroup):
                    return arg.group
                elif arg == dist.isend:
                    arg = old_isend
                elif arg == dist.irecv:
                    arg = old_irecv
                return arg

            args = (convert(arg) for arg in args)
            kwargs = {k: convert(v) for k, v in kwargs.items()}
            return func(*args, **kwargs)

        return new_function

    dist.P2POp.__new__ = get_new_p2pop_function(dist.P2POp.__new__)
    dist.P2POp.__init__ = get_new_p2pop_function(dist.P2POp.__init__)


class ReloadableProcessGroup(torch.distributed.ProcessGroup):
    GROUPS = {}

    def __init__(self, group, ranks):
        super().__init__(
            rank=dist.get_rank(group),
            size=dist.get_world_size(group),
        )
        self.group = group
        self.group_info = {
            "ranks": ranks,
        }
        pid = os.getpid()
        if pid not in ReloadableProcessGroup.GROUPS:
            ReloadableProcessGroup.GROUPS[pid] = []
        ReloadableProcessGroup.GROUPS[pid].append(self)

    def __getattr__(self, name):
        return getattr(self.group, name)

    @staticmethod
    def destroy_process_groups():
        pid = os.getpid()
        for reloadable_group in ReloadableProcessGroup.GROUPS.get(pid, []):
            if reloadable_group.group is None:
                continue
            try:
                dist.destroy_process_group(reloadable_group.group)
            except ValueError as e:
                logger.warning(
                    f"Process group already invalid/destroyed; skipping cleanup. Exception: {e}",
                    exc_info=True,
                )

            del reloadable_group.group
            reloadable_group.group = None

    @staticmethod
    def reload_process_groups():
        pid = os.getpid()
        reloadable_groups = ReloadableProcessGroup.GROUPS.get(pid, [])
        logger.info(f"Reloading {len(reloadable_groups)} process groups in pid {pid}")
        old_new_group = old_new_group_dict.get(pid)
        for reloadable_group in reloadable_groups:
            if reloadable_group.group is not None:
                continue
            group = old_new_group(ranks=reloadable_group.group_info["ranks"], backend="nccl")
            reloadable_group.group = group

    def rank(self) -> int:
        return self.group.rank()

    def size(self) -> int:
        return self.group.size()

    def name(self) -> str:
        return self.group.name()

    def shutdown(self) -> None:
        if self.group is not None:
            self.group.shutdown()

    def abort(self) -> None:
        if self.group is not None:
            self.group.abort()

    def _fwd(self, method, *args, **kwargs):
        inner = self.group
        if inner is None:
            raise RuntimeError("ReloadableProcessGroup: inner PG is None, call reload() first.")
        with _wrap_low_level_call():
            return getattr(inner, method)(*args, **kwargs)

    def barrier(self, *a, **kw):
        return self._fwd("barrier", *a, **kw)

    def broadcast(self, *a, **kw):
        return self._fwd("broadcast", *a, **kw)

    def allreduce(self, *a, **kw):
        return self._fwd("allreduce", *a, **kw)

    def allreduce_coalesced(self, *a, **kw):
        return self._fwd("allreduce_coalesced", *a, **kw)

    def reduce(self, *a, **kw):
        return self._fwd("reduce", *a, **kw)

    def allgather(self, *a, **kw):
        return self._fwd("allgather", *a, **kw)

    def _allgather_base(self, *a, **kw):
        return self._fwd("_allgather_base", *a, **kw)

    def allgather_coalesced(self, *a, **kw):
        return self._fwd("allgather_coalesced", *a, **kw)

    def allgather_into_tensor_coalesced(self, *a, **kw):
        return self._fwd("allgather_into_tensor_coalesced", *a, **kw)

    def gather(self, *a, **kw):
        return self._fwd("gather", *a, **kw)

    def scatter(self, *a, **kw):
        return self._fwd("scatter", *a, **kw)

    def reduce_scatter(self, *a, **kw):
        return self._fwd("reduce_scatter", *a, **kw)

    def _reduce_scatter_base(self, *a, **kw):
        return self._fwd("_reduce_scatter_base", *a, **kw)

    def reduce_scatter_tensor_coalesced(self, *a, **kw):
        return self._fwd("reduce_scatter_tensor_coalesced", *a, **kw)

    def alltoall_base(self, *a, **kw):
        return self._fwd("alltoall_base", *a, **kw)

    def alltoall(self, *a, **kw):
        return self._fwd("alltoall", *a, **kw)

    def send(self, *a, **kw):
        return self._fwd("send", *a, **kw)

    def recv(self, *a, **kw):
        return self._fwd("recv", *a, **kw)

    def recv_anysource(self, *a, **kw):
        return self._fwd("recv_anysource", *a, **kw)

    def _start_coalescing(self, *a, **kw):
        return self._fwd("_start_coalescing", *a, **kw)

    def _end_coalescing(self, *a, **kw):
        return self._fwd("_end_coalescing", *a, **kw)

    def _get_backend_name(self):
        return self._fwd("_get_backend_name")

    def _get_backend(self, *a, **kw):
        return self._fwd("_get_backend", *a, **kw)

    def _set_default_backend(self, *a, **kw):
        return self._fwd("_set_default_backend", *a, **kw)

    @property
    def bound_device_id(self):
        return self.group.bound_device_id

    @bound_device_id.setter
    def bound_device_id(self, dev):
        self.group.bound_device_id = dev


def destroy_process_groups():
    """Destroy all reloadable process groups."""
    ReloadableProcessGroup.destroy_process_groups()


def reload_process_groups():
    """Reload all reloadable process groups."""
    ReloadableProcessGroup.reload_process_groups()


@contextmanager
def _wrap_low_level_call():
    try:
        yield
    except Exception as e:
        mem_info = print_memory("after torch distributed error")
        e.add_note(f"{mem_info=}")
        raise
