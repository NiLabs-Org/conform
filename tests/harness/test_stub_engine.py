"""Self-tests for the harness itself.

Everything else in `tests/` checks inference engines. This file checks the
thing doing the checking, against a stub server whose behaviour we control
exactly.

The important case is the last one. A conformance suite that cannot
demonstrate it detects a known-wrong engine is worth nothing, and "all tests
pass" is indistinguishable from "no engines were reachable" unless something
proves otherwise. These tests are that proof, and they need no GPU, no model
and no network.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from conform.adapters.openai_compat import OpenAICompatEngine
from conform.types import Capability, CompletionRequest, EngineError

# Behaviour toggles read by the handler. Set per test.
BEHAVIOUR: dict[str, Any] = {}


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")

        if BEHAVIOUR.get("status", 200) != 200:
            self._send(BEHAVIOUR["status"], {"error": "nope"})
            return

        text = BEHAVIOUR.get("text", "hello world")
        stop = request.get("stop") or []
        # A correct server truncates before the stop string. The buggy one
        # returns it, which is exactly the bug the suite exists to catch.
        if stop and not BEHAVIOUR.get("buggy_stop", False):
            for token in stop:
                index = text.find(token)
                if index >= 0:
                    text = text[:index]
                    break

        self._send(
            200,
            {
                "choices": [
                    {
                        "text": text,
                        "finish_reason": BEHAVIOUR.get("finish_reason", "stop"),
                        "logprobs": BEHAVIOUR.get("logprobs"),
                    }
                ]
            },
        )

    def _send(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:
        """Silence the default stderr access log."""


@pytest.fixture
def stub_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        BEHAVIOUR.clear()


def _engine(url: str) -> OpenAICompatEngine:
    return OpenAICompatEngine(
        name="stub",
        base_url=url,
        model="stub-model",
        capabilities=frozenset({Capability.STOP_SEQUENCES}),
    )


def test_adapter_parses_a_normal_response(stub_url: str) -> None:
    BEHAVIOUR.update(text="Paris is the capital.", finish_reason="stop")
    with _engine(stub_url) as engine:
        result = engine.complete(CompletionRequest(prompt="x", max_tokens=8))
    assert result.text == "Paris is the capital."
    assert result.finish_reason == "stop"


def test_http_error_becomes_engine_error(stub_url: str) -> None:
    """A 500 must raise EngineError, which the suite reports as an environment
    problem rather than as a conformance failure."""
    BEHAVIOUR.update(status=500)
    with _engine(stub_url) as engine, pytest.raises(EngineError):
        engine.complete(CompletionRequest(prompt="x", max_tokens=8))


def test_unreachable_server_becomes_engine_error() -> None:
    engine = OpenAICompatEngine(name="stub", base_url="http://127.0.0.1:1", model="m")
    with engine, pytest.raises(EngineError):
        engine.complete(CompletionRequest(prompt="x", max_tokens=8))


def test_capabilities_gate_the_tokenizer(stub_url: str) -> None:
    with _engine(stub_url) as engine:
        assert not engine.supports(Capability.TOKENIZE)
        with pytest.raises(NotImplementedError):
            engine.encode("hello")


def test_suite_detects_a_correct_stop_implementation(stub_url: str) -> None:
    BEHAVIOUR.update(text="a b c d e", buggy_stop=False)
    with _engine(stub_url) as engine:
        result = engine.complete(CompletionRequest(prompt="x", max_tokens=16, stop=("c",)))
    assert "c" not in result.text


def test_suite_detects_a_buggy_stop_implementation(stub_url: str) -> None:
    """The proof that any of this works.

    Given a server that deliberately includes the stop string, the assertion
    used by the real conformance test must fail. If this ever passes, the
    suite has stopped detecting the bug class it was built for.
    """
    BEHAVIOUR.update(text="a b c d e", buggy_stop=True)
    with _engine(stub_url) as engine:
        result = engine.complete(CompletionRequest(prompt="x", max_tokens=16, stop=("c",)))
    assert "c" in result.text, "stub was supposed to return the stop string"
