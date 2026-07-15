import runpy
from pathlib import Path

import pytest

from miles.rollout.generate_utils import openai_endpoint_utils as endpoint


CONFIG = {
    "MILES_SESSION_REQUEST_TIMEOUT_SECONDS": ("_SESSION_REQUEST_TIMEOUT", 86400.0),
    "MILES_SESSION_HTTP_CONNECT_TIMEOUT_SECONDS": ("_HTTP_CONNECT_TIMEOUT", 600.0),
    "MILES_SESSION_HTTP_READ_TIMEOUT_SECONDS": ("_HTTP_READ_TIMEOUT", 86400.0),
    "MILES_SESSION_HTTP_WRITE_TIMEOUT_SECONDS": ("_HTTP_WRITE_TIMEOUT", 600.0),
    "MILES_SESSION_HTTP_POOL_TIMEOUT_SECONDS": ("_HTTP_POOL_TIMEOUT", 600.0),
    "MILES_SESSION_COLLECT_CONCURRENCY": ("_COLLECT_RECORDS_CONCURRENCY", 8192),
}


def _load_client(monkeypatch, overrides):
    for key in CONFIG:
        monkeypatch.delenv(key, raising=False)
    for key, value in overrides.items():
        monkeypatch.setenv(key, str(value))
    # Execute the real module in an isolated namespace, leaving active clients alone.
    return runpy.run_path(str(Path(endpoint.__file__)))


def test_defaults_and_live_timeout_and_semaphore(monkeypatch):
    config = _load_client(monkeypatch, {})
    for name, default in CONFIG.values():
        assert config[name] == default
    timeout = config["OpenAIEndpointTracer"]._timeout()
    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (600.0, 86400.0, 600.0, 600.0)
    assert config["_COLLECT_RECORDS_SEMAPHORE"]._value == 8192


@pytest.mark.parametrize("key", CONFIG)
def test_each_environment_limit_is_honored(monkeypatch, key):
    config = _load_client(monkeypatch, {key: 7})
    assert config[CONFIG[key][0]] == 7
    for other, (name, default) in CONFIG.items():
        if other != key:
            assert config[name] == default
    if key == "MILES_SESSION_COLLECT_CONCURRENCY":
        assert config["_COLLECT_RECORDS_SEMAPHORE"]._value == 7


@pytest.mark.parametrize("key", CONFIG)
@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_invalid_limits_fail_at_client_initialization(monkeypatch, key, value):
    with pytest.raises(ValueError):
        _load_client(monkeypatch, {key: value})
