import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import timedelta

import pytest
from tests.fast.utils.ft.conftest import TestCollector

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.core.node_agent import FtNodeAgent
from miles.utils.ft.agents.types import GaugeSample
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig


@asynccontextmanager
async def _running_agent_with_prom(
    node_id: str,
    collectors: list[BaseCollector],
) -> AsyncGenerator[tuple[FtNodeAgent, MiniPrometheus], None]:
    agent = FtNodeAgent(node_id=node_id, collectors=collectors)
    try:
        await agent.start()
        await asyncio.sleep(0.3)

        prom = MiniPrometheus(config=MiniPrometheusConfig())
        prom.add_scrape_target(target_id=node_id, address=agent.get_exporter_address())
        await prom.scrape_once()
        yield agent, prom
    finally:
        await agent.stop()


class TestNodeAgentMiniPrometheusIntegration:
    @pytest.mark.anyio
    async def test_scrape_and_query_latest(self) -> None:
        test_collector = TestCollector(
            metrics=[
                GaugeSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=72.5,
                ),
            ],
            collect_interval=0.1,
        )

        async with _running_agent_with_prom("integ-node-0", [test_collector]) as (_agent, prom):
            df = prom.query_latest("gpu_temperature_celsius")
            assert not df.is_empty()
            values = df["value"].to_list()
            assert 72.5 in values

    @pytest.mark.anyio
    async def test_updated_values_visible_after_rescrape(self) -> None:
        test_collector = TestCollector(
            metrics=[
                GaugeSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=60.0,
                ),
            ],
            collect_interval=0.1,
        )

        async with _running_agent_with_prom("integ-node-1", [test_collector]) as (_agent, prom):
            df1 = prom.query_latest("gpu_temperature_celsius")
            assert 60.0 in df1["value"].to_list()

            test_collector.set_metrics(
                [
                    GaugeSample(
                        name="gpu_temperature_celsius",
                        labels={"gpu": "0"},
                        value=85.0,
                    ),
                ]
            )
            await asyncio.sleep(0.3)
            await prom.scrape_once()

            df2 = prom.query_latest("gpu_temperature_celsius")
            assert 85.0 in df2["value"].to_list()

            df_range = prom.query_range(
                "gpu_temperature_celsius",
                window=timedelta(minutes=5),
            )
            range_values = df_range["value"].to_list()
            assert 60.0 in range_values
            assert 85.0 in range_values

    @pytest.mark.anyio
    async def test_multiple_metrics_all_queryable(self) -> None:
        test_collector = TestCollector(
            metrics=[
                GaugeSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=70.0,
                ),
                GaugeSample(
                    name="gpu_memory_used_bytes",
                    labels={"gpu": "0"},
                    value=8192.0,
                ),
                GaugeSample(
                    name="gpu_power_watts",
                    labels={"gpu": "0"},
                    value=250.0,
                ),
            ],
            collect_interval=0.1,
        )

        async with _running_agent_with_prom("integ-node-2", [test_collector]) as (_agent, prom):
            df_temp = prom.query_latest("gpu_temperature_celsius")
            assert 70.0 in df_temp["value"].to_list()

            df_mem = prom.query_latest("gpu_memory_used_bytes")
            assert 8192.0 in df_mem["value"].to_list()

            df_power = prom.query_latest("gpu_power_watts")
            assert 250.0 in df_power["value"].to_list()

    @pytest.mark.anyio
    async def test_label_filter_query(self) -> None:
        test_collector = TestCollector(
            metrics=[
                GaugeSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=65.0,
                ),
                GaugeSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "1"},
                    value=78.0,
                ),
            ],
            collect_interval=0.1,
        )

        async with _running_agent_with_prom("integ-node-3", [test_collector]) as (_agent, prom):
            df = prom.query_latest("gpu_temperature_celsius", label_filters={"gpu": "1"})
            assert not df.is_empty()
            assert 78.0 in df["value"].to_list()
            assert len(df) == 1
