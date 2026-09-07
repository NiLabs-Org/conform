"""A pytest plugin that turns a run into the conformance matrix.

Enabled with ``-p conform.report``. It listens to test results, groups them by
engine, and writes both a machine-readable JSON file and a Markdown table.

The matrix is the product. A pass/fail count tells you nothing useful; a grid
showing that three engines agree and one does not is the whole point.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from conform.types import EngineError

SYMBOLS = {
    "pass": "pass",
    "fail": "FAIL",
    "skip": "skip",
    "error": "ERR",
    "diverge": "note",
}

#: nodeid -> observed behaviour, filled by the ``record_divergence`` fixture.
#: Tier-3 behaviour (see docs/spec.md) is recorded, never judged, so these
#: rows report what an engine did instead of whether it was right.
DIVERGENCES: dict[str, str] = {}


@dataclass
class Outcome:
    status: str
    detail: str = ""


@dataclass
class Matrix:
    """Results indexed by test, then by engine."""

    rows: dict[str, dict[str, Outcome]] = field(default_factory=lambda: defaultdict(dict))
    engines: list[str] = field(default_factory=list)

    def record(self, test: str, engine: str, outcome: Outcome) -> None:
        if engine not in self.engines:
            self.engines.append(engine)
        # setup skips arrive before call results; never let one overwrite a
        # real outcome that already landed.
        existing = self.rows[test].get(engine)
        if existing is None or existing.status == "skip":
            self.rows[test][engine] = outcome

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "engines": self.engines,
            "results": {
                test: {
                    engine: {"status": o.status, "detail": o.detail} for engine, o in row.items()
                }
                for test, row in sorted(self.rows.items())
            },
        }

    def to_markdown(self) -> str:
        if not self.engines:
            return "_No results._\n"

        labels = {test: _elide(test) for test in self.rows}
        width = max((len(v) for v in labels.values()), default=4)
        width = max(width, len("test"))
        columns = [max(len(e), 6) for e in self.engines]

        head = "| " + "test".ljust(width) + " | "
        head += " | ".join(e.ljust(w) for e, w in zip(self.engines, columns, strict=True)) + " |"
        rule = "|" + "-" * (width + 2) + "|"
        rule += "|".join("-" * (w + 2) for w in columns) + "|"

        lines = [head, rule]
        for test in sorted(self.rows):
            cells = []
            for engine, w in zip(self.engines, columns, strict=True):
                outcome = self.rows[test].get(engine)
                cells.append((SYMBOLS[outcome.status] if outcome else "-").ljust(w))
            lines.append("| " + labels[test].ljust(width) + " | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"


class MatrixPlugin:
    def __init__(self) -> None:
        self.matrix = Matrix()

    # -- collection --------------------------------------------------------

    @staticmethod
    def _identify(item: pytest.Item) -> tuple[str, str] | None:
        """Split an item into (test key, engine name).

        The engine comes from the callspec rather than the node id, because
        parameter order in the id is not something to rely on.
        """
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            return None
        spec = callspec.params.get("engine_spec")
        if spec is None:
            return None

        # Use pytest's own id string, not the raw parameter values: a
        # parametrised 500-character string must not become a 500-character
        # column header.
        name = getattr(item, "originalname", None) or item.name.split("[")[0]
        full = getattr(callspec, "id", "") or ""
        rest = _strip_engine_id(full, spec.name)
        key = f"{name}[{rest}]" if rest else name
        return key, spec.name

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        identity = getattr(report, "_conform_identity", None)
        if identity is None:
            return
        test, engine = identity

        if report.when == "setup" and report.skipped:
            self.matrix.record(test, engine, Outcome("skip", _reason(report)))
        elif report.when == "call":
            observed = DIVERGENCES.get(report.nodeid)
            if observed is not None:
                self.matrix.record(test, engine, Outcome("diverge", observed))
            elif report.passed:
                self.matrix.record(test, engine, Outcome("pass"))
            elif report.skipped:
                self.matrix.record(test, engine, Outcome("skip", _reason(report)))
            elif getattr(report, "_conform_transport_error", False):
                # The server was unreachable or errored. That is an environment
                # problem, not a divergence, and must never read as a finding.
                self.matrix.record(test, engine, Outcome("error", _reason(report)))
            else:
                self.matrix.record(test, engine, Outcome("fail", _reason(report)))
        elif report.when in {"setup", "teardown"} and report.failed:
            self.matrix.record(test, engine, Outcome("error", _reason(report)))

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo[None]):  # type: ignore[no-untyped-def]
        report = yield
        result = report.get_result()
        identity = self._identify(item)
        if identity is not None:
            result._conform_identity = identity  # noqa: SLF001
            excinfo = getattr(call, "excinfo", None)
            if excinfo is not None and excinfo.errisinstance(EngineError):
                result._conform_transport_error = True  # noqa: SLF001

    # -- output ------------------------------------------------------------

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        out_dir = Path(session.config.getoption("--conform-out", default="results"))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "matrix.json").write_text(
            json.dumps(self.matrix.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "matrix.md").write_text(self.matrix.to_markdown(), encoding="utf-8")

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        terminalreporter.write_sep("=", "conformance matrix")
        for line in self.matrix.to_markdown().splitlines():
            terminalreporter.write_line(line)


def _elide(text: str, limit: int = 64) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _strip_engine_id(full: str, engine: str) -> str:
    """Remove the engine component from a combined pytest parameter id."""
    if full == engine:
        return ""
    if full.startswith(f"{engine}-"):
        return full[len(engine) + 1 :]
    if full.endswith(f"-{engine}"):
        return full[: -len(engine) - 1]
    return full


def _reason(report: pytest.TestReport) -> str:
    if report.longrepr is None:
        return ""
    return str(report.longrepr).strip().splitlines()[-1][:300]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--conform-out",
        default="results",
        help="Directory for matrix.json and matrix.md (default: results)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(MatrixPlugin(), "conform-matrix")
