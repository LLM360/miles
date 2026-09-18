# QAE with adaptive clipping

This recipe combines Quantile Advantage Estimation (QAE) with the adaptive
upper clipping controller from MAI-Thinking-1.

```bash
python train.py \
  ... \
  --advantage-estimator grpo \
  --qae-quantile 0.4 \
  --eps-clip 0.6 \
  --use-adaptive-clip \
  --adaptive-clip-target-entropy 0.3 \
  --adaptive-clip-step-size 0.25 \
  --adaptive-clip-max-relaxation 2.5 \
  --entropy-coef 0
```

`--qae-quantile 0.4` changes only the group reward baseline. The implementation
uses the paper's right-continuous empirical quantile. For binary rewards this
produces the two QAE regimes exactly: hard groups update successes and easy
groups update failures.

With adaptive clipping, the PPO ratio interval at step `t` is

```text
[1 - eps_clip, 1 / (1 - eps_clip) + k_t]
```

and the controller applies

```text
k_{t+1} = clip(k_t + delta * sign(target_entropy - estimated_entropy), 0, k_max).
```

The entropy estimate is the importance-weighted all-token average from the MAI
report. The controller updates once after each optimizer step and its state is
saved beside checkpoints. Monitor these training metrics:

- `adaptive_clip_entropy`: controller entropy estimate
- `adaptive_clip_relaxation`: current `k`
- `adaptive_clip_high`: MILES upper clip offset (the upper ratio is `1 + value`)

The MAI values above are a starting point, not universal defaults for every
model and rollout temperature. A sign controller commonly oscillates in a
narrow band around the target; it does not guarantee a smooth sine wave.
