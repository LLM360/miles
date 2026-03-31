# tito/5.2-abort-resume：Fully-Async 无显式中断改造（低侵入版实施计划）

## 1. 设计原则（先约束）

本方案按“主干最小侵入”执行，明确约束：

- **不修改 `train_async.py` 主循环语义**（主干保持不动）。
- 新能力必须是**可选项**，默认配置下行为与当前主干一致。
- 优先通过现有扩展点接入（`--rollout-function-path`、`MILES_EXPERIMENTAL_ROLLOUT_REFACTOR`），避免改动训练主流程。

## 2. Summary

目标是在 fully-async 场景下提供一条“无显式 abort 打断用户请求”的可选路径：

- rollout 侧不再在每轮末尾强制 `abort_request`（仅在可选策略下）。
- weight update 窗口使用 `pause_generation(mode=<user_selected>) -> update -> continue_generation`。
- 对外默认行为不变；仅当显式开启策略时启用新行为。

## 3. 最小改动方案

### A. 策略开关（可选，默认关闭）

在 rollout/weight-update 相关模块引入一个统一策略开关（建议名）：

- `fully_async_interrupt_policy`:
  - `legacy_abort_resume`（默认）
  - `no_interrupt`

并新增 pause mode 可选项（仅对 `no_interrupt` 生效）：

- `fully_async_pause_mode`:
  - `retract`（默认）
  - `in_place`

实现方式优先级（按侵入性从低到高）：

1. 先用环境变量（例如 `MILES_FULLY_ASYNC_INTERRUPT_POLICY`）读取，避免改大范围参数链路。
2. 稳定后再补 CLI 参数透传。

### B. **不改 `train_async.py`**（核心约束）

- 训练循环维持现状，包括 update interval 的同步点。
- 新策略只在 rollout 函数与权重更新实现内部生效。
- 如需实验更激进调度，走独立实验入口（新脚本），不污染主干 `train_async.py`。

### C. Rollout 侧（仅 fully-async 策略分支）

修改：

- `miles/rollout/inference_rollout/inference_rollout_common.py`
- `miles/rollout/inference_rollout/inference_rollout_train.py`

行为：

- 当 `continuous=True` 且策略为 `no_interrupt`：
  - 不在每个 rollout 结束时调用 `abort()`。
  - 维持持久 in-flight / ready 队列，按目标 batch 消费返回。
- 当策略为 `legacy_abort_resume`：
  - 完全保持当前逻辑（含 rollout 末尾 abort）。

边界：

- 继续复用现有 `max_buffer_staleness` 做有界 off-policy。
- `start_rollout_id` 元数据在提交阶段补齐，保证 staleness 判定可用。

### D. Weight Update 侧（仅改 endpoint 调用模式）

修改：

- `miles/backends/sglang_utils/sglang_engine.py`
- `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
- `miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py`

行为：

- 给 `pause_generation` 增加可选参数 `mode`（默认仍是 `abort`，保持兼容）。
- `no_interrupt` 策略下调用：
  - `pause_generation(mode=<fully_async_pause_mode>)`，其中 `<fully_async_pause_mode> ∈ {retract, in_place}`，默认 `retract`。
  - `flush_cache`
  - `update_weights_*`
  - `continue_generation()`
- `legacy_abort_resume` 下保持当前 `abort` 模式。

说明：

- 不改 SGLang server API；仅使用现有 `PauseGenerationReqInput.mode` 能力。

### E. Session server 行为（默认不动）

- 本阶段不改 `session_server.py/sessions.py` 主行为。
- `legacy_abort_resume` 继续沿用现有 gate。
- `no_interrupt` 下不在 session server 内处理 abort/resume；中断控制仅由 SGLang `pause_generation/continue_generation` 处理。

## 4. Public Interface / 兼容性

- 默认策略 `legacy_abort_resume`：现有作业无感知、无行为变化。
- 可选开启 `no_interrupt` 后才启用新路径。
- `no_interrupt` 下用户可选 `fully_async_pause_mode`：`retract` 或 `in_place`。
- `train_async.py` 无改动，主干调用关系不变。

## 5. Test Plan（按最小改动对齐）

### A. Rollout integration

`tests/fast/rollout/inference_rollout/integration/test_fully_async.py` 新增：

- `test_no_interrupt_strategy_no_rollout_end_abort`
- `test_no_interrupt_strategy_preserves_batch_size`
- `test_no_interrupt_strategy_with_staleness_bound`

### B. Weight update unit

新增/补充 megatron update 相关测试：

- `legacy_abort_resume` 下 `pause_generation(mode="abort")`
- `no_interrupt` + `retract` 下 `pause_generation(mode="retract")`
- `no_interrupt` + `in_place` 下 `pause_generation(mode="in_place")`
- 上述策略都应完成 `continue_generation` 收尾

### C. Regression

- 全量保留现有 fully-async 与 buffer-staleness 测试。
- 增加策略参数化，确保 legacy 路径 100% 兼容。

### D. E2E 必测样例：Mock Wait Agent 持续查天气闭环

必须新增一个“能稳定复现长链 tool-call + wait 行为”的 mock e2e，用于验证 `no_interrupt` 策略闭环不是一次性 lucky pass。

新增文件（建议）：

- `tests/e2e/sglang/test_no_interrupt_mock_wait_weather_loop.py`
- `tests/e2e/sglang/utils/mock_wait_weather_agent.py`

#### D1. Mock Wait Agent 行为约定

`mock_wait_weather_agent.py` 的 agent 逻辑固定为“持续查天气”：

- 每轮都发 `/v1/chat/completions`（通过 session server）。
- 强制要求 assistant 返回 `tool_calls`（天气工具）。
- agent 执行 mock weather tool 后写回 `tool` message。
- 每轮插入 `await asyncio.sleep(wait_s)` 模拟 wait agent（例如 `wait_s=0.2`）。
- 循环 `max_turns`（建议 12~20），不依赖 final_answer 退出。
- 返回 metadata：`turns_completed`、`total_tool_calls`、`wait_calls`、`aborted_seen`。

建议退出条件：

- 达到 `max_turns` 正常退出（主路径）。
- 任一轮非 200 或无 tool_call 立即 fail（防止“假闭环”）。

#### D2. Mock 服务能力要求

在现有 `MockSGLangServer` 基础上扩展（仅测试用）：

- 支持 `/pause_generation` 与 `/continue_generation`。
- 支持内部 paused 状态（Event）：
- paused 时，新 chat/completions 请求阻塞等待；
- continue 后恢复处理。
- 记录控制面日志：`pause_calls`、`continue_calls`、最后一次 `pause_mode`。

说明：这里不追求精确模拟 SGLang 内部调度，只要求能稳定注入“update 期间暂停/恢复”扰动，验证 agent 闭环可持续。

#### D3. 用例矩阵（至少 3 条）

1. `test_mock_wait_weather_loop_no_interrupt_retract`
- 策略：`fully_async_interrupt_policy=no_interrupt`，`fully_async_pause_mode=retract`。
- 扰动：后台定时触发多次 `pause_generation -> continue_generation`。
- 断言：
- agent 完成 `max_turns`。
- `total_tool_calls >= max_turns`。
- 全程无 HTTP 非 200。
- session 记录中无 `finish_reason=abort` 暴露到 agent 侧。

2. `test_mock_wait_weather_loop_no_interrupt_in_place`
- 同上，仅 pause mode 改为 `in_place`。
- 断言同上，并额外校验 mock 记录到的 `pause_mode == "in_place"`。

3. `test_mock_wait_weather_loop_legacy_abort_resume_control`
- 策略：`legacy_abort_resume`。
- 扰动：通过 session server 的 `/abort_sessions` + `/resume_sessions` 注入。
- 断言：
- agent 能最终跑完（允许慢一些）。
- 与 no_interrupt 组对比，重试/rollback 指标显著更高（作为对照组，不要求绝对阈值）。

#### D4. 通过标准（闭环判定）

该 e2e 组的“闭环通过”标准不是单个请求成功，而是：

- 连续多轮（>=12）tool-call + wait 能稳定进行；
- 中途多次 pause/continue 扰动后仍能继续下一轮；
- 对 no_interrupt 组，agent 侧不感知 abort 中断（无显式失败、无 abort 响应暴露）。

### E. E2E 必测样例：`mode=abort` vs `mode=retract` deterministic 全等验证

目标：在 fully-async 端到端路径下，分别使用 `pause_generation(mode="abort")` 与 `pause_generation(mode="retract")` 跑同一组样本，验证两条路径结果完全一致；并验证 `retract` 路径内部的 re-prefill logprob 与 decode logprob 一致。

优先复用现有用例与工具：

- `tests/e2e/sglang/test_tito_logprob_equivalence.py`
- `tests/e2e/sglang/utils/logprob_verify_generate.py`

#### E1. 测试配置（必须 deterministic）

- 打开 `--sglang-enable-deterministic-inference`
- `temperature=0.0`
- 固定随机种子（沿用 rollout deterministic seed 机制）
- 关闭会引入非确定性的额外扰动（例如与本用例无关的并发随机流）

#### E2. 对比口径

- 对比对象（分两层）：
  - 层 1（模式内一致性，`retract`）：session incremental decode 路径的 `output_token_logprobs` vs full re-prefill 的 `input_token_logprobs`
  - 层 2（跨模式一致性，`abort` vs `retract`）：两次 e2e rollout 的最终 token 序列、逐 token logprob、以及（若开启）`routed_experts`
- Token 对齐规则：
  - 先做 token_id 精确匹配（不允许偏移/替换）
  - 仅对匹配 token 做 logprob 对比

#### E3. 通过标准（严格）

- token_id：逐 token **完全一致**
- logprob：逐 token **完全一致**（工程上允许 `abs diff <= 1e-8`，超出即 fail）
- 若启用 routing replay（MoE）：
  - `routed_experts` 逐项一致
- 跨模式（`abort` vs `retract`）：
  - 同一 prompt/seed 下，最终 rollout 样本内容与可比 metadata（排除时间戳、request_id 等非确定字段）必须一致

#### E4. 与策略联动的覆盖要求

- 必须覆盖两组：
  - `pause_generation(mode="abort")`
  - `pause_generation(mode="retract")`
- 两组使用同一份 prompt、同一 deterministic 配置、同一 seed，执行后做逐项 diff
- `abort` 组可通过 `legacy_abort_resume` 路径实现，或通过测试专用 mode override 显式触发 `pause_generation(mode="abort")`
- 同时保留一条“无 pause 干预”的基线跑法，确保失败时可区分是 pause mode 引入还是通用逻辑问题
- 测试输出需打印 mismatch 定位信息（turn index、token index、token_id、两侧 logprob）

## 6. 分阶段落地

- **Phase 1（本次）**：Rollout + weight update 最小改动，train_async 不动。
- **Phase 2（可选）**：若需要进一步减少 update 阻塞，再评估独立实验入口（非主干）做更激进 fully-async 调度。

## 7. Assumptions

- 当前主目标后端：Megatron 路径。
- `use_fault_tolerance=false` 前提不变。
- `use_session_server` 先保持现状，避免一次性扩大改动面。

## 8. Phase 1 实施记录

### 已完成项

| 模块 | 文件 | 改动 |
|------|------|------|
| 配置 | `miles/utils/arguments.py` | `--fully-async-interrupt-policy` (legacy_abort_resume/no_interrupt) 和 `--fully-async-pause-mode` (retract/in_place)，支持 env var 回退 |
| SGLang Engine | `miles/backends/sglang_utils/sglang_engine.py` | `pause_generation(mode="abort")` 透传 mode 到 SGLang `/pause_generation` endpoint |
| Rollout | `miles/rollout/inference_rollout/inference_rollout_train.py` | `no_interrupt + continuous` 时跳过 `abort()`；报告 `rollout/no_interrupt/pending_at_end` 指标 |
| Weight Update (distributed) | `miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py` | `_get_pause_mode()` + 按策略选择 mode |
| Weight Update (tensor) | `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | 同上 |
| Actor | `miles/backends/megatron_utils/actor.py` | `no_interrupt` 时跳过 session server pause/resume |
| Mock Server | `miles/utils/test_utils/mock_sglang_server.py` | `/pause_generation`、`/continue_generation`、`/flush_cache` endpoint + abort/pause/continue 调用计数 |

### 测试覆盖

| 类别 | 文件 | 用例数 |
|------|------|--------|
| 5.A Rollout 集成 | `tests/fast/rollout/inference_rollout/integration/test_fully_async.py` | 3 new |
| 5.B Weight update 单元 | `tests/fast/backends/megatron_utils/test_pause_mode.py` | 10 |
| 5.C 回归 | 现有 fully-async + buffer staleness 测试 | 全量通过 |
| 5.D E2E Mock Wait Agent | `tests/e2e/sglang/test_no_interrupt_mock_wait_weather_loop.py` | 3 |
| 5.E Deterministic 全等 | 需真实 GPU 推理，Phase 1 暂缓 | 0（待 GPU 环境验证） |

### Phase 2 注意事项

- **持久化 in-flight 队列**：Phase 1 跳过 abort 后，pending asyncio Task 结果被丢弃。Phase 2 应实现跨 rollout 调用的持久队列，复用 in-flight 结果。
- **start_rollout_id 缺失**：当前 `start_rollout_id` 仅在 `abort()` 的 partial rollout 收集路径设置。Phase 2 持久队列需要在提交时（而非回收时）就标注 `start_rollout_id`，否则 staleness filter 无法判定 off-policy age。
- **retract vs in_place 语义**：`retract` 回退已生成 token 后以新权重 re-prefill（结果更干净），`in_place` 直接暂停/恢复（更快但中间存在 old/new 权重混合 token）。用户应根据对 off-policy tolerance 的需求选择。
