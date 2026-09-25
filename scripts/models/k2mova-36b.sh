MODEL_ARGS=(
    --bf16
    --num-layers 48
    --hidden-size 2560
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --ffn-hidden-size 6144

    # FFN MoE
    --num-experts 100
    --moe-ffn-hidden-size 768
    --moe-router-topk 8
    --moe-router-score-function sigmoid
    --moe-router-topk-scaling-factor 2.5
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0.001
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1.1111111111111112e-6
    --moe-router-dtype fp32
    --moe-shared-expert-intermediate-size 768
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    --moe-shared-expert-overlap

    # routed value experts
    --mova-num-value-experts 64
    --mova-router-topk 4
    --mova-router-score-function sigmoid
    --mova-router-topk-scaling-factor 2.5
    --mova-router-enable-expert-bias
    --mova-router-bias-update-rate 0.001
    --mova-router-load-balancing-type aux_loss
    --mova-router-aux-loss-coeff 1.1111111111111112e-6
    --mova-num-dense-layers 3
    --mova-norm-num-groups 2
    --mova-attention-gate-function softplus
    --mova-value-backend grouped_gemm
    --xllm-router-compatibility
    --xllm-router-gemm-partitions 2

    --attention-output-gate
    --disable-bias-linear
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --apply-layernorm-1p
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 10000000
    --rotary-interleaved
    # The pinned fused packed-THD rope path drops rotary_interleaved, so MoVA
    # keeps the unfused path to stay in parity with SGLang.
    --no-rope-fusion
    --max-position-embeddings 524288
    --untie-embeddings-and-output-weights
    --init-method-std 0.02
    --vocab-size 250624
    --make-vocab-size-divisible-by 1
)
