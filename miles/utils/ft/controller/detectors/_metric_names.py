# Re-export metric name constants used by detectors.
# Canonical definitions live in miles.utils.ft.metric_names.
from miles.utils.ft.metric_names import (
    DCGM_FI_DEV_GPU_TEMP as NODE_GPU_TEMPERATURE,
    GPU_AVAILABLE as NODE_GPU_AVAILABLE,
    NODE_FILESYSTEM_AVAIL_BYTES as NODE_DISK_AVAILABLE_BYTES,
    NODE_NETWORK_UP as NODE_NIC_UP,
    TRAINING_ITERATION,
    TRAINING_JOB_STATUS,
    TRAINING_PHASE,
    XID_CODE_RECENT as NODE_XID_CODE_RECENT,
)
