import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from tests.e2e.ft.conftest_ft.execution import prepare, run_training
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode


def create_comparison_app(
    *,
    build_baseline_args: Callable[[FTTestMode, str], str],
    build_target_args: Callable[[FTTestMode, str], str],
    compare_fn: Callable[[str, FTTestMode], None],
    phases: list[str] | None = None,
) -> typer.Typer:
    """Generate a typer app with baseline/target/compare/run commands.

    For simple (no-phase) tests, leave phases empty.
    For multi-phase tests (e.g. with_failure), provide phase names like ["phase_a", "phase_b"].
    """
    app: typer.Typer = typer.Typer()
    effective_phases: list[str] = phases or [""]

    def _get_dump_subdir(side: str, phase: str) -> str:
        if phase:
            return f"{side}/{phase}"
        return side

    def _run_side(
        side: str,
        build_fn: Callable[[FTTestMode, str], str],
        mode: str,
        dump_dir: str | None,
        phase: str,
    ) -> None:
        ft_mode = resolve_mode(mode)
        if dump_dir is None:
            dump_dir = str(Path(tempfile.mkdtemp(prefix="ft_test_")) / "dumps")
        sub = _get_dump_subdir(side, phase)
        full_dump_dir = f"{dump_dir}/{sub}"
        args = build_fn(ft_mode, full_dump_dir)
        prepare(ft_mode)
        run_training(train_args=args, mode=ft_mode)

    @app.command()
    def baseline(
        mode: Annotated[str, typer.Option(help="Test mode variant")],
        dump_dir: Annotated[str | None, typer.Option(help="Dump base directory")] = None,
        phase: Annotated[str, typer.Option(help="Phase name (multi-phase tests)")] = "",
    ) -> None:
        """Run baseline (normal DP) training."""
        _run_side("baseline", build_baseline_args, mode, dump_dir, phase)

    @app.command()
    def target(
        mode: Annotated[str, typer.Option(help="Test mode variant")],
        dump_dir: Annotated[str | None, typer.Option(help="Dump base directory")] = None,
        phase: Annotated[str, typer.Option(help="Phase name (multi-phase tests)")] = "",
    ) -> None:
        """Run target (indep_dp) training."""
        _run_side("target", build_target_args, mode, dump_dir, phase)

    @app.command()
    def compare(
        mode: Annotated[str, typer.Option(help="Test mode variant")],
        dump_dir: Annotated[str | None, typer.Option(help="Dump base directory")] = None,
    ) -> None:
        """Compare baseline and target dumps."""
        ft_mode = resolve_mode(mode)
        compare_fn(dump_dir, ft_mode)

    @app.command()
    def run(
        mode: Annotated[str, typer.Option(help="Test mode variant")],
    ) -> None:
        """Full pipeline: prepare + all phases + compare."""
        ft_mode = resolve_mode(mode)
        dump_dir: str = str(Path(tempfile.mkdtemp(prefix="ft_test_")) / "dumps")
        print(f"Dump directory: {dump_dir}")

        prepare(ft_mode)

        for phase in effective_phases:
            sub_baseline = _get_dump_subdir("baseline", phase)
            sub_target = _get_dump_subdir("target", phase)

            baseline_args = build_baseline_args(ft_mode, f"{dump_dir}/{sub_baseline}")
            run_training(train_args=baseline_args, mode=ft_mode)

            target_args = build_target_args(ft_mode, f"{dump_dir}/{sub_target}")
            run_training(train_args=target_args, mode=ft_mode)

        compare_fn(dump_dir, ft_mode)

    return app


def create_non_comparison_app(
    *,
    build_args: Callable[[FTTestMode, str], str],
    verify_fn: Callable[[str, FTTestMode], None] | None = None,
) -> typer.Typer:
    """Generate a typer app with a single 'run' command for non-comparison tests."""
    app: typer.Typer = typer.Typer()

    @app.command()
    def run(
        mode: Annotated[str, typer.Option(help="Test mode variant")],
    ) -> None:
        """Full pipeline: prepare + execute + verify."""
        ft_mode = resolve_mode(mode)
        dump_dir: str = str(Path(tempfile.mkdtemp(prefix="ft_test_")) / "dumps")
        print(f"Dump directory: {dump_dir}")

        prepare(ft_mode)
        args = build_args(ft_mode, dump_dir)
        run_training(train_args=args, mode=ft_mode)

        if verify_fn is not None:
            verify_fn(dump_dir, ft_mode)

    return app
