"""Route each rollout sample to a different generate function by a metadata field.

Lets one job mix task types that need different rollout strategies (e.g. agentic
search/SWE vs. single-step reasoning), since miles binds only one generate fn per job.

Invoke:
    --custom-generate-function-path miles.rollout.generate_hub.dispatch.generate
    --generate-route-key task_type
    --generate-routes "reasoning=miles.rollout.generate_hub.single_turn.generate,search=agent360.harbor.miles.tito_generate.generate"
    --generate-default-route miles.rollout.generate_hub.single_turn.generate

Per sample, sample.metadata[<route-key>] selects the route; the matching dotted path
is loaded with load_generate_function (cached) and awaited. Missing/unmatched -> default
(error if --generate-default-route unset). Flags a routed target needs must be on the
launch command; this module does not register them.
"""

from __future__ import annotations

import argparse

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput

_CACHE: dict = {}


def _resolve(path: str):
    if path not in _CACHE:
        from miles.rollout.inference_rollout.compatibility import load_generate_function

        fn = load_generate_function(path)
        if fn is None:
            raise ValueError(f"dispatch: could not load generate function {path!r}")
        _CACHE[path] = fn
    return _CACHE[path]


def _routes(spec: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in (spec or "").split(","):
        if not item.strip():
            continue
        key, _, path = item.partition("=")
        if not path.strip():
            raise ValueError(f"dispatch: bad route {item!r}, expected key=module.path")
        out[key.strip()] = path.strip()
    return out


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    routes = _routes(getattr(args, "generate_routes", None))
    key = getattr(args, "generate_route_key", "task_type")
    kind = (input.sample.metadata or {}).get(key)
    path = routes.get(kind, getattr(args, "generate_default_route", None))
    if path is None:
        raise ValueError(
            f"dispatch: no route for {key}={kind!r} and no --generate-default-route "
            f"(routes={sorted(routes)})"
        )
    return await _resolve(path)(input)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-route-key", type=str, default="task_type")
    parser.add_argument("--generate-routes", type=str, default=None)
    parser.add_argument("--generate-default-route", type=str, default=None)
    return parser


generate.add_arguments = _add_arguments
