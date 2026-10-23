"""Opt-in synchronized wall timers; nested totals must not be added twice."""
from collections import defaultdict
from contextlib import contextmanager
import time
import torch


class StageTimes:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.seconds = defaultdict(float)

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.seconds[name] += time.perf_counter()-started

    def publish(self):
        if self.enabled:
            from miles.utils.timer import Timer
            for key, value in self.seconds.items():
                Timer().add(key, value)
