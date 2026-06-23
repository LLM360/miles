# xLLM 8B dense GQA model arguments.
MODEL_ARGS=(
    --swiglu
    --num-layers 36
    --hidden-size 4096
    --ffn-hidden-size 12288
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --disable-bias-linear
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --layernorm-num-groups 4
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 10000000
    --untie-embeddings-and-output-weights
    --vocab-size 250624
)
