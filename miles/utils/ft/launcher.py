from __future__ import annotations

import json
import logging
import shlex
from typing import Annotated
from uuid import uuid4

import ray
import typer

from miles.utils.ft.platform.controller_actor import FtControllerActor
from miles.utils.ft.platform.controller_factory import FtControllerConfig
from miles.utils.ft.protocols.platform import ft_controller_actor_name

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command(
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)
def main(
    ctx: typer.Context,
    ft_id: Annotated[
        str, typer.Option(help="FT instance ID for multi-instance isolation (auto-generated if empty)")
    ] = "",
    k8s_label_suffix: Annotated[
        str, typer.Option(help="K8s label suffix for label isolation (empty = no suffix, used in testing)")
    ] = "",
    tick_interval: Annotated[
        float, typer.Option(help="Controller main loop interval (seconds)")
    ] = 30.0,
    platform: Annotated[
        str, typer.Option(help="Platform mode: 'stub' or 'k8s-ray'")
    ] = "k8s-ray",
    ray_address: Annotated[
        str, typer.Option(help="Ray dashboard address (k8s-ray mode)")
    ] = "http://127.0.0.1:8265",
    metric_store_backend: Annotated[
        str, typer.Option(help="Metric store backend: 'mini' or 'prometheus'")
    ] = "mini",
    prometheus_url: Annotated[
        str, typer.Option(help="Prometheus server URL (prometheus mode)")
    ] = "http://prometheus:9090",
    controller_exporter_port: Annotated[
        int, typer.Option(help="Controller Prometheus exporter HTTP port (0 = auto)")
    ] = 0,
    runtime_env_json: Annotated[
        str, typer.Option(help="Runtime env JSON for the training Ray job")
    ] = "{}",
) -> None:
    """FT Controller entry point.

    Creates an FtController as a named Ray Actor, submits the training
    command (passed after --) as a Ray job, and blocks until the
    controller loop finishes.

    Usage: python -m miles.utils.ft.launcher [OPTIONS] -- COMMAND...
    """
    entrypoint = shlex.join(ctx.args)
    runtime_env = json.loads(runtime_env_json) if runtime_env_json else {}

    ft_id = ft_id or uuid4().hex[:8]
    config = FtControllerConfig(
        ft_id=ft_id,
        k8s_label_suffix=k8s_label_suffix,
        platform=platform,
        ray_address=ray_address,
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        metric_store_backend=metric_store_backend,
        prometheus_url=prometheus_url,
        controller_exporter_port=controller_exporter_port,
        tick_interval=tick_interval,
    )

    actor_name = ft_controller_actor_name(ft_id)
    logger.info(
        "launcher_started actor_name=%s platform=%s backend=%s exporter_port=%d entrypoint=%s",
        actor_name,
        config.platform,
        config.metric_store_backend,
        config.controller_exporter_port,
        entrypoint,
    )

    actor = FtControllerActor.options(name=actor_name).remote(config=config)
    ray.get(actor.submit_and_run.remote())


if __name__ == "__main__":
    app()
