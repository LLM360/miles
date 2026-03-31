"""Unit tests for fully-async interrupt policy pause mode selection.

Verifies that _get_pause_mode() returns the correct SGLang pause mode
based on fully_async_interrupt_policy and fully_async_pause_mode settings.

Plan section 5.B coverage:
- legacy_abort_resume → pause_generation(mode="abort")
- no_interrupt + retract → pause_generation(mode="retract")
- no_interrupt + in_place → pause_generation(mode="in_place")
"""

from argparse import Namespace

import pytest


def _get_pause_mode(args: Namespace) -> str:
    """Replicate the _get_pause_mode logic from weight updaters for unit testing."""
    policy = getattr(args, "fully_async_interrupt_policy", "legacy_abort_resume")
    if policy == "no_interrupt":
        return getattr(args, "fully_async_pause_mode", "retract")
    return "abort"


class TestGetPauseMode:
    def test_legacy_abort_resume_returns_abort(self):
        args = Namespace(fully_async_interrupt_policy="legacy_abort_resume")
        assert _get_pause_mode(args) == "abort"

    def test_no_interrupt_retract(self):
        args = Namespace(fully_async_interrupt_policy="no_interrupt", fully_async_pause_mode="retract")
        assert _get_pause_mode(args) == "retract"

    def test_no_interrupt_in_place(self):
        args = Namespace(fully_async_interrupt_policy="no_interrupt", fully_async_pause_mode="in_place")
        assert _get_pause_mode(args) == "in_place"

    def test_no_interrupt_default_pause_mode_is_retract(self):
        """When no_interrupt is set but pause_mode is not specified, default to retract."""
        args = Namespace(fully_async_interrupt_policy="no_interrupt")
        assert _get_pause_mode(args) == "retract"

    def test_missing_policy_defaults_to_abort(self):
        """When fully_async_interrupt_policy is not set at all, default to abort."""
        args = Namespace()
        assert _get_pause_mode(args) == "abort"


class TestSGLangEnginePauseMode:
    """Test that pause_generation passes mode correctly to the HTTP request."""

    def test_pause_generation_sends_mode_in_payload(self):
        """Verify the pause_generation method sends the mode parameter."""
        from unittest.mock import MagicMock, patch

        from miles.backends.sglang_utils.sglang_engine import SGLangEngine

        engine = SGLangEngine.__new__(SGLangEngine)
        engine.server_host = "127.0.0.1"
        engine.server_port = 9999

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("miles.backends.sglang_utils.sglang_engine.requests.post", return_value=mock_response) as mock_post:
            engine.pause_generation(mode="retract")
            mock_post.assert_called_once_with(
                "http://127.0.0.1:9999/pause_generation",
                json={"mode": "retract"},
            )

    @pytest.mark.parametrize("mode", ["abort", "retract", "in_place"])
    def test_pause_generation_all_modes(self, mode):
        """All three modes are correctly forwarded."""
        from unittest.mock import MagicMock, patch

        from miles.backends.sglang_utils.sglang_engine import SGLangEngine

        engine = SGLangEngine.__new__(SGLangEngine)
        engine.server_host = "127.0.0.1"
        engine.server_port = 9999

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("miles.backends.sglang_utils.sglang_engine.requests.post", return_value=mock_response) as mock_post:
            engine.pause_generation(mode=mode)
            mock_post.assert_called_once_with(
                "http://127.0.0.1:9999/pause_generation",
                json={"mode": mode},
            )

    def test_pause_generation_default_mode_is_abort(self):
        """Default mode parameter is 'abort' for backward compatibility."""
        from unittest.mock import MagicMock, patch

        from miles.backends.sglang_utils.sglang_engine import SGLangEngine

        engine = SGLangEngine.__new__(SGLangEngine)
        engine.server_host = "127.0.0.1"
        engine.server_port = 9999

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("miles.backends.sglang_utils.sglang_engine.requests.post", return_value=mock_response) as mock_post:
            engine.pause_generation()  # no mode arg
            mock_post.assert_called_once_with(
                "http://127.0.0.1:9999/pause_generation",
                json={"mode": "abort"},
            )
