"""Builds engines from ``conform.toml``.

Contributors should never have to edit test code to point the suite at a new
server. They edit config. A config file also means a CI run and a local run
are describing the same thing in the same way.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from conform.adapters.base import Engine
from conform.types import Capability

DEFAULT_CONFIG = Path("conform.toml")
CONFIG_ENV_VAR = "CONFORM_CONFIG"


@dataclass(frozen=True)
class EngineSpec:
    """One ``[[engines]]`` table, before anything is loaded or connected."""

    name: str
    kind: str
    model: str
    base_url: str | None = None
    api_key: str = "not-needed"
    capabilities: frozenset[Capability] | None = None
    enabled: bool = True


def config_path() -> Path:
    """Where to read config from. ``CONFORM_CONFIG`` wins if set."""
    override = os.environ.get(CONFIG_ENV_VAR)
    return Path(override) if override else DEFAULT_CONFIG


def load_specs(path: Path | None = None) -> list[EngineSpec]:
    """Parse ``conform.toml`` into specs, skipping disabled entries."""
    path = path or config_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No config at {path}. Copy conform.example.toml to conform.toml to get started."
        )

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    specs: list[EngineSpec] = []
    for index, entry in enumerate(raw.get("engines", [])):
        missing = {"name", "kind", "model"} - entry.keys()
        if missing:
            raise ValueError(f"engines[{index}] in {path} is missing {sorted(missing)}")

        declared = entry.get("capabilities")
        specs.append(
            EngineSpec(
                name=entry["name"],
                kind=entry["kind"],
                model=entry["model"],
                base_url=entry.get("base_url"),
                api_key=entry.get("api_key", "not-needed"),
                capabilities=(
                    frozenset(Capability(c) for c in declared) if declared is not None else None
                ),
                enabled=entry.get("enabled", True),
            )
        )

    return [spec for spec in specs if spec.enabled]


def build(spec: EngineSpec) -> Engine:
    """Turn a spec into a live engine.

    Imports are local to the branch so that running against HTTP engines does
    not require torch, and running against the reference does not require a
    server to be up.
    """
    if spec.kind == "reference":
        from conform.adapters.reference import ReferenceEngine

        engine = ReferenceEngine(model=spec.model)
        engine.name = spec.name
        return engine

    if spec.kind == "openai":
        from conform.adapters.openai_compat import OpenAICompatEngine

        if not spec.base_url:
            raise ValueError(f"engine {spec.name!r} is kind 'openai' but has no base_url")
        return OpenAICompatEngine(
            name=spec.name,
            base_url=spec.base_url,
            model=spec.model,
            api_key=spec.api_key,
            capabilities=spec.capabilities,
        )

    raise ValueError(f"engine {spec.name!r} has unknown kind {spec.kind!r}")
