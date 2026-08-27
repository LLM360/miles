from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from miles.utils.types import Sample


class DynamicSamplingReplacementBudgetExceeded(RuntimeError):
    """Dynamic sampling could not make accepted progress within its budget."""


class EvaluatorInvalidReplacementBudgetExceeded(DynamicSamplingReplacementBudgetExceeded):
    """Evaluator-invalid groups prevented dynamic sampling from making progress."""


def _flatten_samples(value) -> list[Sample]:
    if isinstance(value, Sample):
        return [value]
    if isinstance(value, (list, tuple)):
        return [sample for item in value for sample in _flatten_samples(item)]
    return []


def group_has_evaluator_failure(group: list[Sample]) -> bool:
    """Return whether a scorer explicitly marked any sample as invalid.

    ``evaluation_failed`` is the backend-independent contract. The older
    ``llm_judge_failed`` marker remains recognized so rollout liveness is also
    bounded for existing custom scorers.
    """

    for sample in _flatten_samples(group):
        metadata = getattr(sample, "metadata", None) or {}
        if metadata.get("evaluation_failed") is True or metadata.get("llm_judge_failed") is True:
            return True
    return False


@dataclass
class DynamicSamplingReplacementTracker:
    """Bound rejected replacement groups while preserving normal sampling.

    The optional limit counts rejected prompt groups since the last group that
    survived the dynamic filter. Any accepted group is real progress and resets
    the window. A disabled limit (``None``) retains unbounded dynamic sampling.
    """

    max_rejected_groups_without_progress: int | None = None
    rejected_groups_without_progress: int = 0
    evaluator_invalid_groups_without_progress: int = 0
    rejection_reasons: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        limit = self.max_rejected_groups_without_progress
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise ValueError("max rejected groups without progress must be a positive integer or None")

    def record_filter_result(
        self,
        *,
        keep: bool,
        group: list[Sample],
        reason: str | None,
    ) -> None:
        if keep:
            self.reset_progress_window()
            return

        self.rejected_groups_without_progress += 1
        self.rejection_reasons[str(reason) if reason else "unspecified"] += 1
        if group_has_evaluator_failure(group):
            self.evaluator_invalid_groups_without_progress += 1

    def record_filter_batch(
        self,
        results: Iterable[tuple[bool, list[Sample], str | None]],
    ) -> None:
        """Record an unordered completion batch deterministically.

        ``asyncio.wait`` returns a set. If any group in that set is accepted,
        the batch made progress regardless of set iteration order, so it resets
        the no-progress window as a unit.
        """

        completed = list(results)
        if any(keep for keep, _group, _reason in completed):
            self.reset_progress_window()
            return
        for keep, group, reason in completed:
            self.record_filter_result(keep=keep, group=group, reason=reason)

    def raise_if_exhausted(self) -> None:
        limit = self.max_rejected_groups_without_progress
        if not self.exhausted:
            return

        details = (
            f"rejected_groups_without_progress={self.rejected_groups_without_progress}, "
            f"evaluator_invalid_groups={self.evaluator_invalid_groups_without_progress}, "
            f"max_rejected_groups_without_progress={limit}, "
            f"rejection_reasons={dict(sorted(self.rejection_reasons.items()))}"
        )
        if (
            self.evaluator_invalid_groups_without_progress > 0
            and self.evaluator_invalid_groups_without_progress == self.rejected_groups_without_progress
        ):
            raise EvaluatorInvalidReplacementBudgetExceeded(
                "dynamic sampling replacement budget exhausted because evaluator-invalid groups "
                f"prevented accepted progress: {details}"
            )
        raise DynamicSamplingReplacementBudgetExceeded(
            f"dynamic sampling replacement budget exhausted without accepted progress: {details}"
        )

    @property
    def exhausted(self) -> bool:
        limit = self.max_rejected_groups_without_progress
        return limit is not None and self.rejected_groups_without_progress >= limit

    def reset_progress_window(self) -> None:
        self.rejected_groups_without_progress = 0
        self.evaluator_invalid_groups_without_progress = 0
        self.rejection_reasons.clear()
