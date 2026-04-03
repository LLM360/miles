import logging
import subprocess
import sys
from pathlib import Path

from miles.utils.event_logger.logger import read_events
from miles.utils.event_logger.models import MetricEvent

logger = logging.getLogger(__name__)

_REQUIRED_METRIC_KEYS: list[str] = ["train/grad_norm", "train/loss"]


def compare_dumps(
    baseline_dir: str,
    target_dir: str,
    *,
    diff_threshold: float = 0.0085,
    allow_skipped_pattern: str = "input_ids|positions|cu_seqlens_q|cu_seqlens_kv|qkv_format",
    extra_args: list[str] | None = None,
) -> None:
    baseline_path = Path(baseline_dir) / "dumps"
    target_path = Path(target_dir) / "dumps"

    assert baseline_path.exists(), f"Baseline dump dir does not exist: {baseline_path}"
    assert target_path.exists(), f"Target dump dir does not exist: {target_path}"

    result = _run_comparator(
        baseline_path=baseline_path,
        target_path=target_path,
        diff_threshold=diff_threshold,
        allow_skipped_pattern=allow_skipped_pattern,
        extra_args=extra_args,
    )

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
    if key_prefixes is None:
        key_prefixes = ["train/"]

    baseline_events = _read_metric_events(Path(baseline_dir))
    target_events = _read_metric_events(Path(target_dir))
    print(f"Metrics: {baseline_events=} {target_events=}")

    issues: list[str] = []
    issues += _check_event_counts(baseline_events, target_events, baseline_dir, target_dir)

    if not issues:
        for step_idx, (b_event, t_event) in enumerate(zip(baseline_events, target_events, strict=True)):
            issues += _check_step_metrics(step_idx, b_event, t_event, key_prefixes, rtol)

    issues += _check_required_keys_exist(baseline_events)

    assert not issues, (
        f"MetricEvent comparison found {len(issues)} issue(s):\n" + "\n".join(f"  - {i}" for i in issues)
    )
    print(f"MetricEvent comparison passed: {len(baseline_events)} steps compared")


def _check_event_counts(
    baseline: list[MetricEvent],
    target: list[MetricEvent],
    baseline_dir: str,
    target_dir: str,
) -> list[str]:
    issues: list[str] = []
    if len(baseline) == 0:
        issues.append(f"No MetricEvents found in baseline dir: {baseline_dir}")
    if len(target) == 0:
        issues.append(f"No MetricEvents found in target dir: {target_dir}")
    if len(baseline) > 0 and len(target) > 0 and len(baseline) != len(target):
        issues.append(
            f"MetricEvent count mismatch: baseline={len(baseline)}, target={len(target)}"
        )
    return issues


def _check_step_metrics(
    step_idx: int,
    baseline_event: MetricEvent,
    target_event: MetricEvent,
    key_prefixes: list[str],
    rtol: float,
) -> list[str]:
    issues: list[str] = []
    for key in baseline_event.metrics:
        if not any(key.startswith(prefix) for prefix in key_prefixes):
            continue

        if key not in target_event.metrics:
            issues.append(f"Step {step_idx}: metric '{key}' present in baseline but missing in target")
            continue

        issues += _check_single_metric(step_idx, key, baseline_event.metrics[key], target_event.metrics[key], rtol)
    return issues


def _check_single_metric(
    step_idx: int,
    key: str,
    baseline_val: object,
    target_val: object,
    rtol: float,
) -> list[str]:
    if not isinstance(baseline_val, (int, float)) or not isinstance(target_val, (int, float)):
        return []
    if baseline_val == 0.0 and target_val == 0.0:
        return []

    rel_diff = abs(baseline_val - target_val) / max(abs(baseline_val), abs(target_val), 1e-12)
    if rel_diff > rtol:
        return [
            f"Step {step_idx}, metric '{key}': baseline={baseline_val}, target={target_val}, "
            f"rel_diff={rel_diff:.6f} > rtol={rtol}"
        ]
    return []


def _check_required_keys_exist(events: list[MetricEvent]) -> list[str]:
    all_keys: set[str] = set()
    for event in events:
        all_keys.update(event.metrics.keys())

    issues: list[str] = []
    for required in _REQUIRED_METRIC_KEYS:
        if required not in all_keys:
            issues.append(
                f"Required metric '{required}' not found in any baseline MetricEvent. "
                f"Available keys: {sorted(all_keys)}"
            )
    return issues


def _run_comparator(
    *,
    baseline_path: Path,
    target_path: Path,
    diff_threshold: float,
    allow_skipped_pattern: str,
    extra_args: list[str] | None,
) -> subprocess.CompletedProcess[str]:
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

    return result


def _read_metric_events(dump_dir: Path) -> list[MetricEvent]:
    events_dir: Path = dump_dir / "events"
    if not events_dir.exists():
        return []
    all_events = read_events(events_dir)
    return [e for e in all_events if isinstance(e, MetricEvent)]
