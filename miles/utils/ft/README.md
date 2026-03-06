# Fault Tolerance (`miles.utils.ft`)

Fault tolerance for Megatron distributed training on Ray + K8s. Detects faulty nodes, evicts them via K8s node labels, and auto-restarts training from the latest checkpoint.

## Architecture

- **Platform layer**: Platform-specific details, such as Kubernetes node-label adapter and Ray job adapter.
- **Controller**: Central control logic.
  - **Detectors**: Detect faults based on metrics.
  - **Recovery Orchestrator**: Multi-phase recovery state machine.
  - **Diagnostics**: On-demand diagnostics tools.
- **Agents**: Per-node and per-rank objects to collect metric and do actions.
  - **Collectors**: Collect various metrics.

## Data Flow

Three independent monitoring channels feed the controller:

1. **Hardware metrics (pull)** -- Collectors on each node produce `MetricSample`s,
   exposed via a Prometheus HTTP exporter on `FtNodeAgent`. MiniPrometheus periodically
   scrapes these exporters; the controller queries the store for anomalies.

2. **Training heartbeat (pull)** -- Each `FtTrainingRankAgent` (one per Megatron rank)
   exposes iteration counter and training phase as Prometheus gauges, scraped by
   MiniPrometheus. The controller detects stalls by checking iteration progress.

3. **Per-step training metrics (push)** -- Each training rank calls `log_step()` to
   push loss, grad norm, MFU, and iteration time to MiniWandb inside the controller.
   The controller queries recent steps to detect NaN loss or MFU decline.

## Fault Handling

The detector chain runs every controller tick. When a fault is detected:

- **High-confidence hardware fault** (critical XID, majority NIC down): immediately
  evict the bad node (K8s label) and restart training.
- **Other faults** (crash, hang, NaN loss, MFU decline): enter the recovery
  orchestrator, which first reattempts training, then runs on-demand diagnostics
  (GPU health, NCCL bandwidth, stack traces) to identify the culprit node before
  evicting.

## Tests

Tests live in `tests/fast/utils/ft/` and mirror the source directory structure.
Run with `pytest tests/fast/utils/ft/`.
