from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from miles.backends.training_utils import ci_utils


def _engine(version):
    remote = MagicMock(return_value=f"version-ref-{version}")
    return SimpleNamespace(get_weight_version=SimpleNamespace(remote=remote))


def test_all_rollout_engine_versions_are_queried(monkeypatch):
    engines = [_engine("7"), _engine("7"), _engine("7")]
    ray_get = MagicMock(return_value=["7", 7, "7"])
    monkeypatch.setattr("ray.get", ray_get)

    ci_utils.assert_rollout_engine_weight_versions(engines, expected_version=7)

    assert [engine.get_weight_version.remote.call_count for engine in engines] == [1, 1, 1]
    ray_get.assert_called_once_with(
        ["version-ref-7", "version-ref-7", "version-ref-7"]
    )


def test_all_rollout_engine_versions_report_every_mismatch(monkeypatch):
    engines = [_engine("7"), _engine("6"), _engine("stale")]
    monkeypatch.setattr("ray.get", MagicMock(return_value=["7", "6", "stale"]))

    with pytest.raises(RuntimeError, match=r"expected 7.*\(1, '6'\).*\(2, 'stale'\)"):
        ci_utils.assert_rollout_engine_weight_versions(engines, expected_version=7)


def test_all_rollout_engine_versions_accepts_empty_engine_set(monkeypatch):
    ray_get = MagicMock()
    monkeypatch.setattr("ray.get", ray_get)

    ci_utils.assert_rollout_engine_weight_versions([], expected_version=7)

    ray_get.assert_not_called()
