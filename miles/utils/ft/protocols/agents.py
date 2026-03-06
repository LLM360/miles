from __future__ import annotations

from typing import Protocol

from miles.utils.ft.models.diagnostics import DiagnosticResult


class NodeAgentProtocol(Protocol):
    async def run_diagnostic(
        self, diagnostic_type: str, timeout_seconds: int = 120,
        **kwargs: object,
    ) -> DiagnosticResult: ...


class DiagnosticProtocol(Protocol):
    diagnostic_type: str

    async def run(
        self, node_id: str, timeout_seconds: int = 120,
    ) -> DiagnosticResult: ...
