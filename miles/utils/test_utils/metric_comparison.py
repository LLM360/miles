import logging
import subprocess
import sys
from pathlib import Path

from miles.utils.event_logger.logger import read_events
from miles.utils.event_logger.models import MetricEvent

logger = logging.getLogger(__name__)


def compare_dumps(
    baseline_dir: str,
    target_dir: str,
    *,
    diff_threshold: float = 0.0085,
    allow_skipped_pattern: str = "input_ids|positions|cu_seqlens_q|cu_seqlens_kv|qkv_format",
    extra_args: list[str] | None = None,
) -> None:
    """Compare tensor dumps between baseline and target using the sglang comparator.

    Raises AssertionError if the comparator reports mismatches or exits non-zero.
    """
    baseline_path = Path(baseline_dir) / "dumps"
    target_path = Path(target_dir) / "dumps"

    if not baseline_path.exists():
        logger.warning("Baseline dump dir %s does not exist, skipping dump comparison", baseline_path)
        return
    if not target_path.exists():
        logger.warning("Target dump dir %s does not exist, skipping dump comparison", target_path)
        return

    cmd: list[str] = [
        sys.executable,
        "-m",
        "sglang.srt.debug_utils.comparator",
        "--baseline-path",
        str(baseline_path),
        "--target-path",
        str(target_path),
        "--output-format",
        "json",
        "--preset",
        "sglang_megatron",
        "--diff-threshold",
        str(diff_threshold),
        "--allow-skipped-pattern",
        allow_skipped_pattern,
    ]
    if extra_args:
        cmd.extend(extra_args)

    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.stdout.strip():
        print(f"[comparator stdout]\n{result.stdout}")
    if result.stderr.strip():
        print(f"[comparator stderr]\n{result.stderr}")

    assert result.returncode == 0, (
        f"Dump comparator failed (rc={result.returncode})\nstderr: {result.stderr[-2000:]}"
    )
    print(f"Dump comparison passed: {baseline_path} vs {target_path}")


def compare_metrics(
    baseline_dir: str,
    target_dir: str,
    *,
    rtol: float = 1e-3,
    key_prefixes: list[str] | None = None,
) -> None:
    """Compare MetricEvents between baseline and target runs.

    Focuses on train/* metrics by default (grad_norm, loss).
    """
    if key_prefixes is None:
        key_prefixes = ["train/"]

    baseline_metrics = _read_metric_events(Path(baseline_dir))
    target_metrics = _read_metric_events(Path(target_dir))
    print(f"Metrics: {baseline_metrics=} {target_metrics=}")

    assert len(baseline_metrics) > 0, f"No MetricEvents found in baseline dir: {baseline_dir}"
    assert len(target_metrics) > 0, f"No MetricEvents found in target dir: {target_dir}"
    assert len(baseline_metrics) == len(target_metrics), (
        f"MetricEvent count mismatch: baseline={len(baseline_metrics)}, target={len(target_metrics)}"
    )

    for step_idx, (b_event, t_event) in enumerate(zip(baseline_metrics, target_metrics, strict=True)):
        for key in b_event.metrics:
            if not any(key.startswith(prefix) for prefix in key_prefixes):
                continue
            assert key in t_event.metrics, (
                f"Step {step_idx}, metric '{key}' present in baseline but missing in target"
            )

            b_val = b_event.metrics[key]
            t_val = t_event.metrics[key]
            if not isinstance(b_val, (int, float)) or not isinstance(t_val, (int, float)):
                continue
            if b_val == 0.0 and t_val == 0.0:
                continue

            rel_diff = abs(b_val - t_val) / max(abs(b_val), abs(t_val), 1e-12)
            assert rel_diff <= rtol, (
                f"Step {step_idx}, metric '{key}': baseline={b_val}, target={t_val}, "
                f"rel_diff={rel_diff:.6f} > rtol={rtol}"
            )

    print(f"MetricEvent comparison passed: {len(baseline_metrics)} steps compared")


def _read_metric_events(dump_dir: Path) -> list[MetricEvent]:
    """Read MetricEvents from the event logger output directory."""
    events_dir: Path = dump_dir / "events"
    if not events_dir.exists():
        return []
    all_events = read_events(events_dir)
    return [e for e in all_events if isinstance(e, MetricEvent)]


