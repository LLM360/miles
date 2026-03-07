from __future__ import annotations

from miles.utils.ft.agents.diagnostics.stack_trace import PySpyThread


class StackTraceAggregator:
    """Aggregate stack traces from multiple nodes and identify suspect nodes.

    Uses majority/minority fingerprint grouping: nodes whose stack trace
    fingerprint differs from the majority group are considered suspects.
    """

    def aggregate(self, traces: dict[str, list[PySpyThread]]) -> list[str]:
        if len(traces) <= 1:
            return []

        fp_to_nodes: dict[str, list[str]] = {}
        for node_id, threads in traces.items():
            fp = self._extract_fingerprint(threads)
            fp_to_nodes.setdefault(fp, []).append(node_id)

        if len(fp_to_nodes) <= 1:
            return []

        majority_size = max(len(nodes) for nodes in fp_to_nodes.values())
        return sorted(nid for nodes in fp_to_nodes.values() if len(nodes) < majority_size for nid in nodes)

    @staticmethod
    def _extract_fingerprint(threads: list[PySpyThread]) -> str:
        """Build a stable fingerprint from the innermost frame of each thread."""
        top_frames: list[str] = []
        for thread in threads:
            if thread.frames:
                frame = thread.frames[0]
                top_frames.append(f"{frame.name} ({frame.filename})")

        top_frames.sort()
        return "|".join(top_frames)
