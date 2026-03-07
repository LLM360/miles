"""Centralized access to MILES_FT_* environment variables.

All env-var reads in miles.utils.ft go through the functions here so that
the mapping from env-var name to Python value is defined in one place.
Functions (not module-level constants) are used because some env vars may be
set after import time.
"""
from __future__ import annotations

import os
from pathlib import Path


def get_ft_id() -> str:
    return os.environ.get("MILES_FT_ID", "")


def get_training_run_id() -> str:
    return os.environ.get("MILES_FT_TRAINING_RUN_ID", "")


def get_exception_inject_path() -> Path | None:
    raw = os.environ.get("MILES_FT_EXCEPTION_INJECT_PATH", "")
    return Path(raw) if raw else None


def get_notify_webhook_url() -> str:
    return (os.environ.get("MILES_FT_NOTIFY_WEBHOOK_URL") or "").strip()


def get_notify_platform() -> str:
    return (os.environ.get("MILES_FT_NOTIFY_PLATFORM") or "").strip().lower()


def get_lark_webhook_url() -> str:
    return (os.environ.get("MILES_FT_LARK_WEBHOOK_URL") or "").strip()


def get_k8s_label_prefix() -> str:
    return os.environ.get("MILES_FT_K8S_LABEL_PREFIX", "")
