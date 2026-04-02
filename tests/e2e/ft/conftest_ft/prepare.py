def prepare(mode: FTTestMode) -> None:
    """Download trimmed model, convert checkpoint, download debug rollout data."""
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download {MODEL_HF_REPO} --local-dir /root/models/{mode.model_name}")
    U.convert_checkpoint(
        model_name=mode.model_name,
        megatron_model_type=mode.megatron_model_type,
        num_gpus_per_node=mode.num_gpus_total,
    )
    U.hf_download_dataset(DEBUG_ROLLOUT_DATA_HF_REPO)

