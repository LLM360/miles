"""Notifier factory: resolves webhook configuration from environment and
builds the appropriate WebhookNotifier subclass.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from miles.utils.ft.platform.stubs import StubNotifier
from miles.utils.ft.utils.env import get_lark_webhook_url, get_notify_platform, get_notify_webhook_url

if TYPE_CHECKING:
    from miles.utils.ft.platform.notifiers.webhook_notifier import WebhookNotifier

logger = logging.getLogger(__name__)


def build_notifier(platform: str) -> WebhookNotifier | StubNotifier | None:
    notify_platform, webhook_url = _resolve_notify_config()
    if webhook_url:
        cls = _get_notifier_class(notify_platform)
        return cls(webhook_url=webhook_url)

    if platform == "stub":
        return StubNotifier()

    logger.warning(
        "No notifier configured for platform=%s "
        "(MILES_FT_NOTIFY_WEBHOOK_URL / MILES_FT_LARK_WEBHOOK_URL not set). "
        "Recovery alerts will not be delivered.",
        platform,
    )
    return None


def _resolve_notify_config() -> tuple[str, str]:
    """Return (platform, webhook_url) from environment variables.

    Priority: MILES_FT_NOTIFY_PLATFORM + MILES_FT_NOTIFY_WEBHOOK_URL.
    Fallback: MILES_FT_LARK_WEBHOOK_URL implies platform=lark.
    """
    webhook_url = get_notify_webhook_url()
    notify_platform = get_notify_platform()

    if webhook_url and notify_platform:
        return notify_platform, webhook_url

    if webhook_url and not notify_platform:
        return "lark", webhook_url

    legacy_url = get_lark_webhook_url()
    if legacy_url:
        return "lark", legacy_url

    return notify_platform or "lark", ""


def _get_notifier_class(notify_platform: str) -> type[WebhookNotifier]:
    from miles.utils.ft.platform.notifiers.discord_notifier import DiscordWebhookNotifier
    from miles.utils.ft.platform.notifiers.lark_notifier import LarkWebhookNotifier
    from miles.utils.ft.platform.notifiers.slack_notifier import SlackWebhookNotifier

    registry: dict[str, type[WebhookNotifier]] = {
        "lark": LarkWebhookNotifier,
        "slack": SlackWebhookNotifier,
        "discord": DiscordWebhookNotifier,
    }
    cls = registry.get(notify_platform)
    if cls is None:
        raise ValueError(
            f"Unknown notify platform: {notify_platform!r}. "
            f"Supported: {sorted(registry)}"
        )
    return cls
