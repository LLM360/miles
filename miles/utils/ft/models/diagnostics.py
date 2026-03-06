from miles.utils.ft.models.base import FtBaseModel


class DiagnosticResult(FtBaseModel):
    diagnostic_type: str
    node_id: str
    passed: bool
    details: str

    @classmethod
    def pass_result(
        cls, *, diagnostic_type: str, node_id: str, details: str,
    ) -> "DiagnosticResult":
        return cls(diagnostic_type=diagnostic_type, node_id=node_id, passed=True, details=details)

    @classmethod
    def fail_result(
        cls, *, diagnostic_type: str, node_id: str, details: str,
    ) -> "DiagnosticResult":
        return cls(diagnostic_type=diagnostic_type, node_id=node_id, passed=False, details=details)


class UnknownDiagnosticError(Exception):
    """Raised when a node agent is asked to run a diagnostic type it does not have."""
