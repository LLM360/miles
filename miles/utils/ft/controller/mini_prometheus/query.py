from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterator

import polars as pl

from miles.utils.ft.controller.mini_prometheus.promql import (
    CompareExpr,
    CompareOp,
    MetricSelector,
    PromQLExpr,
    RangeFunction,
    RangeFunctionCompare,
    compare_col,
    match_labels,
    parse_promql,
)

_SeriesKey = tuple[str, frozenset[tuple[str, str]]]

_EMPTY_INSTANT = pl.DataFrame({"__name__": [], "value": []})
_EMPTY_RANGE = pl.DataFrame({"__name__": [], "timestamp": [], "value": []})


@dataclass
class TimeSeriesSample:
    timestamp: datetime
    value: float


@dataclass
class SeriesStore:
    series: dict[_SeriesKey, deque[TimeSeriesSample]] = field(default_factory=dict)
    label_maps: dict[_SeriesKey, dict[str, str]] = field(default_factory=dict)
    name_index: dict[str, set[_SeriesKey]] = field(default_factory=dict)

    def iter_matching(
        self, selector: MetricSelector,
    ) -> Iterator[tuple[dict[str, str], deque[TimeSeriesSample]]]:
        for key in self.name_index.get(selector.name, []):
            samples = self.series.get(key)
            if not samples:
                continue

            labels = self.label_maps[key]
            if not match_labels(labels, selector.matchers):
                continue

            yield labels, samples


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------


def instant_query(store: SeriesStore, query: str) -> pl.DataFrame:
    expr = parse_promql(query)
    return _evaluate_instant(store, expr)


def range_query(
    store: SeriesStore,
    query: str,
    start: datetime,
    end: datetime,
    step: timedelta,
) -> pl.DataFrame:
    expr = parse_promql(query)
    return _evaluate_range(store, expr, start=start, end=end, step=step)


# ---------------------------------------------------------------------------
# Internal: shared helpers
# ---------------------------------------------------------------------------


def _filter_by_compare(
    df: pl.DataFrame, op: CompareOp, threshold: float,
) -> pl.DataFrame:
    if df.is_empty():
        return df
    return df.filter(compare_col(pl.col("value"), op, threshold))


def _to_dataframe(
    rows: list[dict[str, object]], empty: pl.DataFrame,
) -> pl.DataFrame:
    if not rows:
        return empty
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal: instant evaluation
# ---------------------------------------------------------------------------


def _evaluate_instant(store: SeriesStore, expr: PromQLExpr) -> pl.DataFrame:
    if isinstance(expr, MetricSelector):
        return _instant_selector(store, expr)

    if isinstance(expr, CompareExpr):
        df = _instant_selector(store, expr.selector)
        return _filter_by_compare(df, expr.op, expr.threshold)

    if isinstance(expr, RangeFunction):
        return _instant_range_function(store, expr)

    if isinstance(expr, RangeFunctionCompare):
        df = _instant_range_function(store, expr.func)
        return _filter_by_compare(df, expr.op, expr.threshold)

    raise ValueError(f"Unsupported expression type: {type(expr)}")


def _instant_selector(store: SeriesStore, selector: MetricSelector) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for labels, samples in store.iter_matching(selector):
        latest = samples[-1]
        row: dict[str, object] = {"__name__": selector.name, "value": latest.value}
        row.update(labels)
        rows.append(row)

    return _to_dataframe(rows, empty=_EMPTY_INSTANT)


def _instant_range_function(store: SeriesStore, func: RangeFunction) -> pl.DataFrame:
    now = datetime.now(timezone.utc)
    window_start = now - func.duration
    rows: list[dict[str, object]] = []

    for labels, samples in store.iter_matching(func.selector):
        window_samples = [s for s in samples if s.timestamp >= window_start]
        if not window_samples:
            continue

        value = _apply_range_function(func.func_name, window_samples)
        row: dict[str, object] = {"__name__": func.selector.name, "value": value}
        row.update(labels)
        rows.append(row)

    return _to_dataframe(rows, empty=_EMPTY_INSTANT)


# ---------------------------------------------------------------------------
# Internal: range evaluation
# ---------------------------------------------------------------------------


def _evaluate_range(
    store: SeriesStore,
    expr: PromQLExpr,
    start: datetime,
    end: datetime,
    step: timedelta,
) -> pl.DataFrame:
    if isinstance(expr, MetricSelector):
        return _range_selector(store, expr, start=start, end=end, step=step)

    if isinstance(expr, CompareExpr):
        df = _range_selector(store, expr.selector, start=start, end=end, step=step)
        return _filter_by_compare(df, expr.op, expr.threshold)

    raise ValueError(
        f"range_query not yet supported for expression type: {type(expr)}"
    )


def _range_selector(
    store: SeriesStore,
    selector: MetricSelector,
    start: datetime,
    end: datetime,
    step: timedelta,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for labels, samples in store.iter_matching(selector):
        for sample in samples:
            if sample.timestamp > end:
                break
            if sample.timestamp >= start:
                row: dict[str, object] = {
                    "__name__": selector.name,
                    "timestamp": sample.timestamp,
                    "value": sample.value,
                }
                row.update(labels)
                rows.append(row)

    return _to_dataframe(rows, empty=_EMPTY_RANGE)


# ---------------------------------------------------------------------------
# Range function evaluation
# ---------------------------------------------------------------------------


def _apply_range_function(
    func_name: str,
    samples: list[TimeSeriesSample],
) -> float:
    if func_name == "count_over_time":
        return float(len(samples))

    if func_name == "changes":
        if len(samples) < 2:
            return 0.0
        changes = sum(
            1
            for i in range(1, len(samples))
            if samples[i].value != samples[i - 1].value
        )
        return float(changes)

    if func_name == "min_over_time":
        return min(s.value for s in samples)

    if func_name == "max_over_time":
        return max(s.value for s in samples)

    if func_name == "avg_over_time":
        return sum(s.value for s in samples) / len(samples)

    raise ValueError(f"Unknown range function: {func_name}")
