# P2P Weight Transfer

This is an example of P2P (point-to-point) weight transfer between training and rollout engines. By using `--update-weight-transfer-mode p2p`, miles bypasses the default NCCL broadcast and instead transfers weights directly from training ranks to rollout engine ranks, which can reduce weight update latency in multi-node setups. More details on the design and implementation can be found in [this issue](https://github.com/radixark/miles/issues/755).

## Files

* `test_qwen3_4b_p2p.py`: single-node CI test with Qwen3-4B.
* `prepare-qwen3-30B-A3B.sh`: download model/datasets and convert checkpoint for the multi-node example.
* `run-qwen3-30B-A3B-4node-profile.sh`: 4-node launch script for Qwen3-30B-A3B with P2P weight transfer.

## Quick Start

* [Single Node](README-single-node.md) - Qwen3-4B on a single node.

## Quick Explanation

The default weight transfer mode in miles is `broadcast`: after training, updated weights are broadcast via NCCL to all rollout engine ranks. This works well on a single node but becomes a bottleneck as the number of nodes grows.

P2P mode (`--update-weight-transfer-mode p2p`) changes this by having each training rank directly write its shard of weights to the corresponding rollout engine rank(s). The key steps are:

1. **Initialization**: Training ranks establish point-to-point connections (via RDMA or other transports) to their target rollout engine ranks.

2. **Weight gather**: Megatron TP/EP shards are all-gathered and converted to HF format, same as the broadcast path.

3. **P2P transfer**: Instead of a collective broadcast, each source rank writes bucketed weight tensors directly to the destination rollout rank's memory.

4. **Synchronization**: After all transfers complete, rollout engines are resumed for the next generation step.
