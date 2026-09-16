from argparse import Namespace

import pytest

from miles.utils.entropy_control import (
    adaptive_clip_high,
    initialize_adaptive_clip,
    load_adaptive_clip_state,
    save_adaptive_clip_state,
    update_adaptive_clip_relaxation,
)


def test_adaptive_clip_high_matches_mai_parameterization() -> None:
    assert adaptive_clip_high(eps_clip=0.6, relaxation=0.0) == pytest.approx(1.5)
    assert adaptive_clip_high(eps_clip=0.6, relaxation=0.25) == pytest.approx(1.75)


@pytest.mark.parametrize(
    ("entropy", "expected"),
    [(0.2, 0.75), (0.3, 0.5), (0.4, 0.25)],
)
def test_adaptive_clip_controller_moves_toward_target(entropy: float, expected: float) -> None:
    actual = update_adaptive_clip_relaxation(
        relaxation=0.5,
        estimated_entropy=entropy,
        target_entropy=0.3,
        step_size=0.25,
        max_relaxation=1.0,
    )
    assert actual == pytest.approx(expected)


def test_adaptive_clip_controller_clamps_relaxation() -> None:
    assert update_adaptive_clip_relaxation(0.0, 1.0, 0.3, 0.25, 1.0) == 0.0
    assert update_adaptive_clip_relaxation(1.0, 0.0, 0.3, 0.25, 1.0) == 1.0


def test_adaptive_clip_state_round_trip(tmp_path) -> None:
    args = Namespace(
        use_adaptive_clip=True,
        eps_clip=0.6,
        adaptive_clip_initial_relaxation=0.0,
        adaptive_clip_max_relaxation=2.5,
    )
    initialize_adaptive_clip(args)
    args.adaptive_clip_relaxation = 0.75
    args.eps_clip_high = adaptive_clip_high(args.eps_clip, args.adaptive_clip_relaxation)

    save_adaptive_clip_state(args, str(tmp_path), 3)
    args.adaptive_clip_relaxation = 0.0
    args.eps_clip_high = 0.0
    load_adaptive_clip_state(args, str(tmp_path), 3)

    assert args.adaptive_clip_relaxation == pytest.approx(0.75)
    assert args.eps_clip_high == pytest.approx(2.25)
