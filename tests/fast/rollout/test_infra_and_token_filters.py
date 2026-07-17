from argparse import Namespace

import pytest

from miles.rollout._agentic_outcomes import _TOKEN_TRUNCATION_EXIT_STATUSES as TRUNCATION_EXIT_STATUSES
from miles.rollout.filter_hub.dynamic_sampling_filters import check_no_infra_failures
from miles.rollout.filter_hub.rollout_filters import mask_token_truncated
from miles.utils.types import Sample

# Preserve coverage of the historical infrastructure statuses.
INFRA_FAILURE_EXIT_STATUSES = frozenset(
    {
        "AgentTimeout",
        "AgentTimeoutError",
        "HealthcheckError",
        "_K8sInternalInfraError",
        "Cancelled",
        "RewardFileNotFoundError",
        "AgentSetupTimeout",
        "AgentSetupTimeoutError",
        "SqsConsumerError",
        "VerifierTimeout",
        "VerifierTimeoutError",
        "EnvStartTimeout",
        "EnvironmentStartTimeoutError",
        "TimeoutError",
        "AddTestsDirError",
    }
)


@pytest.mark.parametrize("exit_status", sorted(INFRA_FAILURE_EXIT_STATUSES))
def test_infrastructure_filter_rejects_nested_failure_before_reading_reward(exit_status):
    failed = Sample(status=Sample.Status.COMPLETED, metadata={"exit_status": exit_status}, reward=None)
    other = Sample(status=Sample.Status.COMPLETED, reward=1)
    output = check_no_infra_failures(Namespace(reward_key=None), [[failed], [other]])
    assert not output.keep and output.reason == f"group_has_{exit_status}"


@pytest.mark.parametrize("exit_status", sorted(TRUNCATION_EXIT_STATUSES))
def test_token_filter_keeps_reward_and_mask_values_for_later_converter(exit_status):
    sample = Sample(
        status=Sample.Status.COMPLETED,
        metadata={"exit_status": exit_status},
        reward=0.5,
        tokens=[1, 2, 3],
        response_length=2,
        loss_mask=[1, 1],
    )
    other = Sample(status=Sample.Status.COMPLETED, metadata=None, reward=1.0)
    mask_token_truncated(None, [[sample], [other]])
    assert sample.remove_sample is True
    assert sample.reward == 0.5 and sample.loss_mask == [1, 1] and sample.tokens == [1, 2, 3]
    assert not other.remove_sample
