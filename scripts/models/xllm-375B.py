from miles.utils.external_utils.model_args_utils import moe_layer_freq


N_DENSE_LAYERS = 3
N_MOE_LAYERS = 58


def model_args() -> str:
    # xLLM K2MoE 375B GQA.
    return (
        "--disable-bias-linear "
        "--group-query-attention "
        "--num-attention-heads 48 "
        "--num-query-groups 8 "
        "--kv-channels 128 "
        f"--num-layers {N_DENSE_LAYERS + N_MOE_LAYERS} "
        "--hidden-size 6144 "
        "--ffn-hidden-size 16384 "
        "--norm-epsilon 1e-6 "
        "--normalization RMSNorm "
        "--position-embedding-type rope "
        "--rotary-percent 0.5 "
        "--rotary-base 500000 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 250624 "
        # MoE
        "--moe-ffn-hidden-size 1792 "
        "--moe-shared-expert-intermediate-size 1792 "
        "--moe-router-pre-softmax "
        "--moe-router-score-function sigmoid "
        "--moe-router-enable-expert-bias "
        "--moe-router-bias-update-rate 0 "
        "--moe-router-load-balancing-type seq_aux_loss "
        "--moe-token-dispatcher-type alltoall "
        "--moe-router-topk 8 "
        "--moe-router-topk-scaling-factor 2.5 "
        f"--moe-layer-freq {moe_layer_freq(nlayers=N_DENSE_LAYERS + N_MOE_LAYERS, first_k_dense_replace=N_DENSE_LAYERS)} "
        "--num-experts 192 "
        "--moe-grouped-gemm "
        "--moe-router-dtype fp32 "
        "--moe-permute-fusion "
        "--moe-aux-loss-coeff 0 "
    )
