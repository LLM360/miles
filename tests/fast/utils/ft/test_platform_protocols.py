import pytest

from miles.utils.ft.platform.protocols import (
    JobStatus,
    NodeManagerProtocol,
    NotificationProtocol,
    TrainingJobProtocol,
)


class TestJobStatus:
    def test_enum_values(self) -> None:
        assert JobStatus.RUNNING == "running"
        assert JobStatus.STOPPED == "stopped"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.PENDING == "pending"

    def test_all_members(self) -> None:
        assert set(JobStatus) == {
            JobStatus.RUNNING,
            JobStatus.STOPPED,
            JobStatus.FAILED,
            JobStatus.PENDING,
        }

    def test_string_conversion(self) -> None:
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus("failed") == JobStatus.FAILED


class TestNodeManagerProtocolStructuralSubtyping:
    def test_conforming_class_is_accepted(self) -> None:
        class _ConformingNodeManager:
            async def mark_node_bad(self, node_id: str, reason: str) -> None:
                pass

            async def unmark_node_bad(self, node_id: str) -> None:
                pass

            async def get_bad_nodes(self) -> list[str]:
                return []

        instance: NodeManagerProtocol = _ConformingNodeManager()
        assert isinstance(instance, _ConformingNodeManager)


class TestTrainingJobProtocolStructuralSubtyping:
    def test_conforming_class_is_accepted(self) -> None:
        class _ConformingTrainingJob:
            async def stop_training(self, timeout_seconds: int = 300) -> None:
                pass

            async def submit_training(self) -> str:
                return "run-123"

            async def get_training_status(self) -> JobStatus:
                return JobStatus.RUNNING

        instance: TrainingJobProtocol = _ConformingTrainingJob()
        assert isinstance(instance, _ConformingTrainingJob)


class TestNotificationProtocolStructuralSubtyping:
    def test_conforming_class_is_accepted(self) -> None:
        class _ConformingNotification:
            async def send(self, title: str, content: str, severity: str) -> None:
                pass

            async def aclose(self) -> None:
                pass

        instance: NotificationProtocol = _ConformingNotification()
        assert isinstance(instance, _ConformingNotification)
