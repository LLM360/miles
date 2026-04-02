from pathlib import Path

from miles.utils.event_logger.logger import read_events
from miles.utils.event_logger.models import MetricEvent


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


def assert_events_dir_exists(dump_dir: str) -> None:
    """Assert that the events directory exists and contains jsonl files."""
    events_dir: Path = Path(dump_dir) / "events"
    assert events_dir.exists(), f"Events directory not found: {events_dir}"
    jsonl_files = list(events_dir.glob("**/*.jsonl"))
    assert len(jsonl_files) > 0, f"No event files found in {events_dir}"


def _read_metric_events(dump_dir: Path) -> list[MetricEvent]:
    """Read MetricEvents from the event logger output directory."""
    events_dir: Path = dump_dir / "events"
    if not events_dir.exists():
        return []
    all_events = read_events(events_dir)
    return [e for e in all_events if isinstance(e, MetricEvent)]


