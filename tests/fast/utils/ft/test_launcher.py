from unittest.mock import patch

from typer.testing import CliRunner

from miles.utils.ft.platform.feishu_notifier import FeishuWebhookNotifier
from miles.utils.ft.platform.launcher import _build_notifier, app
from miles.utils.ft.platform.stubs import StubNotifier

runner = CliRunner()


class TestLauncherCli:
    def test_help_output(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "tick-interval" in result.output
        assert "FT Controller" in result.output

    def test_help_includes_platform_option(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--platform" in result.output
        assert "--ray-address" in result.output
        assert "--entrypoint" in result.output

    def test_invalid_platform_raises_error(self) -> None:
        result = runner.invoke(app, ["--platform", "unknown"])
        assert result.exit_code != 0
        assert "Unknown platform" in result.output


class TestBuildNotifier:
    def test_webhook_url_returns_feishu_notifier(self) -> None:
        with patch.dict("os.environ", {"FT_FEISHU_WEBHOOK_URL": "https://hook.example.com"}):
            notifier = _build_notifier(platform_mode="stub")
        assert isinstance(notifier, FeishuWebhookNotifier)

    def test_stub_mode_without_webhook_returns_stub(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            notifier = _build_notifier(platform_mode="stub")
        assert isinstance(notifier, StubNotifier)

    def test_k8s_ray_mode_without_webhook_returns_none(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            notifier = _build_notifier(platform_mode="k8s-ray")
        assert notifier is None

    def test_empty_webhook_url_treated_as_unset(self) -> None:
        with patch.dict("os.environ", {"FT_FEISHU_WEBHOOK_URL": "  "}):
            notifier = _build_notifier(platform_mode="stub")
        assert isinstance(notifier, StubNotifier)
