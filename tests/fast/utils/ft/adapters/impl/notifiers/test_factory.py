from __future__ import annotations

import logging

import pytest

from miles.utils.ft.adapters.impl.notifiers.discord_notifier import DiscordWebhookNotifier
from miles.utils.ft.adapters.impl.notifiers.factory import build_notifier
from miles.utils.ft.adapters.impl.notifiers.lark_notifier import LarkWebhookNotifier
from miles.utils.ft.adapters.impl.notifiers.slack_notifier import SlackWebhookNotifier
from miles.utils.ft.adapters.stubs import StubNotifier


class TestBuildNotifier:
    def test_webhook_url_defaults_to_lark_notifier(self) -> None:
        notifier = build_notifier(platform="ray", notify_webhook_url="https://hook.example.com")
        assert isinstance(notifier, LarkWebhookNotifier)

    def test_explicit_slack_platform(self) -> None:
        notifier = build_notifier(
            platform="ray",
            notify_webhook_url="https://hook.example.com",
            notify_platform="slack",
        )
        assert isinstance(notifier, SlackWebhookNotifier)

    def test_explicit_discord_platform(self) -> None:
        notifier = build_notifier(
            platform="ray",
            notify_webhook_url="https://hook.example.com",
            notify_platform="discord",
        )
        assert isinstance(notifier, DiscordWebhookNotifier)

    def test_strips_whitespace_from_inputs(self) -> None:
        notifier = build_notifier(
            platform="ray",
            notify_webhook_url="  https://hook.example.com  ",
            notify_platform="  slack  ",
        )
        assert isinstance(notifier, SlackWebhookNotifier)

    def test_whitespace_only_url_treated_as_empty(self) -> None:
        notifier = build_notifier(platform="stub", notify_webhook_url="   ")
        assert isinstance(notifier, StubNotifier)

    def test_stub_platform_without_webhook_returns_stub(self) -> None:
        notifier = build_notifier(platform="stub")
        assert isinstance(notifier, StubNotifier)

    def test_non_stub_platform_without_webhook_returns_none_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            notifier = build_notifier(platform="ray")

        assert notifier is None
        assert "No notifier configured" in caplog.text

    def test_unknown_platform_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown notify platform.*'teams'"):
            build_notifier(
                platform="ray",
                notify_webhook_url="https://hook.example.com",
                notify_platform="teams",
            )
