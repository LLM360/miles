# Fault Tolerance

To ensure long-term, stable RL training, miles enables a certain level of fault tolerance by default. This section introduces the design philosophy behind fault tolerance in miles.

To enable the fault tolerance function in miles, please set `--use-fault-tolerance`.

## Rollout Fault Tolerance

During the rollout process, miles periodically sends heartbeat requests (`/health_generate`) to all SGLang servers. If a heartbeat times out, that SGLang server will be stopped. After the current rollout round is complete, the server will be restarted and its parameters will be correctly updated.

- `--rollout-health-check-first-wait`: Since some large MoE models require compilation on their first run, miles will wait for `rollout_health_check_first_wait` seconds before the first rollout to start sending heartbeats. Defaults to 300s.
- `--rollout-health-check-interval`: The interval between heartbeat checks. Defaults to 10s.
- `--rollout-health-check-timeout`: The timeout limit for a heartbeat request. Defaults to 5s.

## Trainer Fault Tolerance

When using independent data parallelism (`indep_dp`), miles automatically monitors trainer actors via heartbeat checks. A background thread periodically calls each actor's `heartbeat()` method and compares the returned last-active timestamp against the current time. If the timestamp is stale (exceeding `--trainer-heartbeat-staleness`) or the RPC times out, the cell is marked as errored and excluded from subsequent training rounds.

This mechanism is independent of `--use-fault-tolerance` and activates automatically when multiple training cells are present.

- `--trainer-heartbeat-first-wait`: Initial grace period before starting heartbeat checks, allowing time for model initialization and compilation. Defaults to 300s.
- `--trainer-heartbeat-interval`: The interval between heartbeat checks. Defaults to 30s.
- `--trainer-heartbeat-timeout`: The timeout for a single heartbeat RPC. Defaults to 10s.
- `--trainer-heartbeat-staleness`: Maximum allowed staleness of the last-active timestamp. Defaults to 90s.
