"""Shared fixtures.

Every test in this suite takes an ``engine`` fixture and is run once per
engine in ``conform.toml``. That is the whole trick: a test is written once
and becomes a row in the matrix automatically.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from conform.adapters.base import Engine
from conform.registry import EngineSpec, build, load_specs
from conform.report import DIVERGENCES
from conform.types import Capability, CompletionRequest, EngineError


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires(capability): skip unless the engine under test declares this capability",
    )


def _specs() -> list[EngineSpec]:
    try:
        return load_specs()
    except FileNotFoundError:
        return []


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrise anything asking for ``engine`` across configured engines."""
    if "engine" not in metafunc.fixturenames:
        return

    specs = _specs()
    if not specs:
        pytest.skip("no conform.toml; copy conform.example.toml to get started")

    metafunc.parametrize("engine_spec", specs, ids=[s.name for s in specs], indirect=True)


@pytest.fixture(scope="session")
def engine_spec(request: pytest.FixtureRequest) -> EngineSpec:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def engine(engine_spec: EngineSpec) -> Iterator[Engine]:
    """A live engine.

    Session-scoped because loading a model or opening a pool is expensive and
    nothing in the suite mutates engine state.
    """
    try:
        built = build(engine_spec)
    except Exception as exc:
        pytest.skip(f"could not start {engine_spec.name}: {exc}")

    # Probe once before running anything. A server that is not up is an
    # environment problem, and reporting it as a column of failures would
    # train people to ignore red.
    try:
        built.complete(CompletionRequest(prompt="ping", max_tokens=1, temperature=0.0))
    except EngineError as exc:
        built.close()
        pytest.skip(f"{engine_spec.name} is not reachable: {exc}")
    except Exception:
        pass

    try:
        yield built
    finally:
        built.close()


@pytest.fixture(autouse=True)
def _honour_requires(request: pytest.FixtureRequest) -> None:
    """Turn ``@pytest.mark.requires(...)`` into a skip.

    Lacking a feature is not a conformance failure. Only implementing one
    incorrectly is.
    """
    marker = request.node.get_closest_marker("requires")
    if marker is None:
        return
    if "engine" not in request.fixturenames:
        return

    engine: Engine = request.getfixturevalue("engine")
    for capability in marker.args:
        if not engine.supports(Capability(capability)):
            pytest.skip(f"{engine.name} does not support {Capability(capability).value}")


@pytest.fixture
def record_divergence(request: pytest.FixtureRequest) -> Callable[[str], None]:
    """Record what an engine did, without calling it right or wrong.

    For tier-3 behaviour (see docs/spec.md), where the API genuinely does not
    specify an answer. The row reports ``note`` and carries the observed
    behaviour, so the matrix shows the disagreement without the project
    pretending to arbitrate it.
    """

    def _record(observed: str) -> None:
        DIVERGENCES[request.node.nodeid] = observed

    return _record
