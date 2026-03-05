"""Training job launcher and FT Controller entry point.

Two entry points live here:

1. **Wrapper mode** (``__main__``): sets environment variables from
   ``--runtime-env-json`` and execs the trailing command.  Sits between
   ``ray job submit`` and the real training script so that future
   concerns (FT controller start, health checks, …) can be added here
   without touching command_utils.py.

   Usage::

       python -m miles.utils.ft.launcher \\
           --runtime-env-json '{"env_vars": {"K": "V"}}' \\
           -- python3 train.py --lr 0.001

2. **FT Controller mode** (``app``): typer CLI that launches the
   FT Controller either as a detached Ray Actor or inline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Annotated

import typer

from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.models import FT_CONTROLLER_ACTOR_NAME
from miles.utils.ft.platform.controller_actor import FtControllerActor
from miles.utils.ft.platform.controller_factory import (
    FtControllerConfig,
    build_ft_controller,
)

_ = FtController, ControllerExporter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Wrapper mode — called via ``python -m miles.utils.ft.launcher``
# ---------------------------------------------------------------------------


def _apply_env_vars(runtime_env_json: str) -> None:
    parsed = json.loads(runtime_env_json)
    env_vars: dict[str, str] = parsed.get("env_vars", {})
    for key, value in env_vars.items():
        os.environ[key] = value
    if env_vars:
        logger.info("launcher set %d env vars: %s", len(env_vars), list(env_vars.keys()))


def _wrapper_main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]

    runtime_env_json: str | None = None
    command: list[str] = []

    i = 0
    while i < len(args):
        if args[i] == "--runtime-env-json" and i + 1 < len(args):
            runtime_env_json = args[i + 1]
            i += 2
        elif args[i].startswith("--runtime-env-json="):
            runtime_env_json = args[i].split("=", 1)[1]
            i += 1
        elif args[i] == "--":
            command = args[i + 1 :]
            break
        else:
            raise SystemExit(f"launcher: unexpected argument: {args[i]}")

    if runtime_env_json is not None:
        _apply_env_vars(runtime_env_json)

    if not command:
        raise SystemExit("launcher: no command given after '--'")

    os.execvp(command[0], command)


# ---------------------------------------------------------------------------
# 2. FT Controller mode — importable typer app
# ---------------------------------------------------------------------------

app = typer.Typer()


@app.command()
def main(
    tick_interval: Annotated[
        float, typer.Option(help="Controller main loop interval (seconds)")
    ] = 30.0,
    platform: Annotated[
        str, typer.Option(help="Platform mode: 'stub' or 'k8s-ray'")
    ] = "stub",
    ray_address: Annotated[
        str, typer.Option(help="Ray dashboard address (k8s-ray mode)")
    ] = "http://127.0.0.1:8265",
    entrypoint: Annotated[
        str, typer.Option(help="Training job entrypoint command (k8s-ray mode)")
    ] = "",
    metric_store_backend: Annotated[
        str, typer.Option(help="Metric store backend: 'mini' or 'prometheus'")
    ] = "mini",
    prometheus_url: Annotated[
        str, typer.Option(help="Prometheus server URL (prometheus mode)")
    ] = "http://prometheus:9090",
    controller_exporter_port: Annotated[
        int, typer.Option(help="Controller Prometheus exporter HTTP port")
    ] = 9400,
    as_ray_actor: Annotated[
        bool, typer.Option(help="Create a detached Ray Actor instead of running inline")
    ] = False,
) -> None:
    """FT Controller entry point.

    When --as-ray-actor is set (production mode), creates a detached named
    Ray Actor and returns immediately. The actor runs the controller loop
    in the background. FtMegatronAgent finds it via ray.get_actor("ft_controller").

    When --as-ray-actor is not set (dev/test mode), builds and runs the
    controller inline with asyncio.run().
    """
    config = FtControllerConfig(
        platform=platform,
        ray_address=ray_address,
        entrypoint=entrypoint,
        metric_store_backend=metric_store_backend,
        prometheus_url=prometheus_url,
        controller_exporter_port=controller_exporter_port,
        tick_interval=tick_interval,
    )

    if as_ray_actor:
        actor = FtControllerActor.options(
            name=FT_CONTROLLER_ACTOR_NAME,
            lifetime="detached",
        ).remote(config=config)
        actor.run.remote()
        logger.info(
            "ft_controller actor created and started "
            "platform=%s backend=%s exporter_port=%d",
            config.platform, config.metric_store_backend, config.controller_exporter_port,
        )
        return

    controller = build_ft_controller(config=config)
    logger.info(
        "launcher_started_inline platform=%s backend=%s exporter_port=%d",
        config.platform, config.metric_store_backend, config.controller_exporter_port,
    )
    asyncio.run(controller.run())


if __name__ == "__main__":
    _wrapper_main()
