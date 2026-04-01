import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Self, TextIO

from miles.utils.event_logger.models import EventBase

_event_logger: "EventLogger | None" = None


def set_event_logger(logger: "EventLogger") -> None:
    global _event_logger
    _event_logger = logger


def get_event_logger() -> "EventLogger":
    if _event_logger is None:
        raise RuntimeError("EventLogger not initialized. Call set_event_logger() first.")
    return _event_logger


def is_event_logger_initialized() -> bool:
    return _event_logger is not None


class EventLogger:
    def __init__(self, *, log_dir: Path | str, file_name: str = "events.jsonl") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file: TextIO = open(self._log_dir / file_name, "a", encoding="utf-8")

    def log(self, event: EventBase) -> None:
        event.timestamp = datetime.now(timezone.utc)
        line = event.model_dump_json() + "\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
