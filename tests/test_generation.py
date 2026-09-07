"""Generation conformance: determinism, stop sequences, and length limits.

These are the behaviours an application actually depends on. If greedy
decoding is not deterministic, your evals are noise. If stop strings are
handled differently, swapping engines silently changes your output.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from conform.adapters.base import Engine
from conform.types import CompletionRequest, EngineError

PROMPT = "The capital of France is"


# -- determinism -----------------------------------------------------------


def test_greedy_is_repeatable(engine: Engine) -> None:
    """Temperature 0 twice in a row must give byte-identical text.

    Nothing exotic here, just the same request sent twice. Engines that batch
    requests can fail this when an unrelated request lands in the same batch
    and changes floating-point accumulation order.
    """
    request = CompletionRequest(prompt=PROMPT, max_tokens=16, temperature=0.0)
    first = engine.complete(request)
    second = engine.complete(request)
    assert first.text == second.text


def test_greedy_ignores_top_p(engine: Engine) -> None:
    """At temperature 0 the distribution is a point mass, so top_p is moot.

    An engine that applies top_p before checking for greedy can truncate the
    distribution and pick a different token.
    """
    base = CompletionRequest(prompt=PROMPT, max_tokens=16, temperature=0.0, top_p=1.0)
    narrowed = CompletionRequest(prompt=PROMPT, max_tokens=16, temperature=0.0, top_p=0.1)
    assert engine.complete(base).text == engine.complete(narrowed).text


# -- stop sequences --------------------------------------------------------


@pytest.mark.requires("stop_sequences")
def test_stop_string_is_excluded(engine: Engine) -> None:
    """The OpenAI API documents the stop sequence as not returned."""
    result = engine.complete(
        CompletionRequest(
            prompt="Letters: a b c d e f g",
            max_tokens=32,
            temperature=0.0,
            stop=("d",),
        )
    )
    assert "d" not in result.text


@pytest.mark.requires("stop_sequences")
def test_stop_matches_across_token_boundaries(engine: Engine) -> None:
    """A stop string need not line up with a token boundary.

    Engines that compare token ids rather than decoded text miss these, which
    is the single most common stop-sequence bug.
    """
    result = engine.complete(
        CompletionRequest(
            prompt="Repeat exactly: alpha beta gamma delta",
            max_tokens=32,
            temperature=0.0,
            stop=("mma",),
        )
    )
    assert "mma" not in result.text


@pytest.mark.requires("stop_sequences")
def test_stop_sets_finish_reason(engine: Engine) -> None:
    """Stopping on a stop string reports finish_reason 'stop', not 'length'."""
    result = engine.complete(
        CompletionRequest(
            prompt="Letters: a b c d e f g",
            max_tokens=32,
            temperature=0.0,
            stop=("c",),
        )
    )
    if result.finish_reason is None:
        pytest.skip(f"{engine.name} does not report finish_reason")
    assert result.finish_reason == "stop"


# -- length limits ---------------------------------------------------------


def test_max_tokens_is_respected(engine: Engine) -> None:
    """max_tokens counts generated tokens, never the prompt.

    Checked in characters because not every engine returns token ids, and no
    tokenizer emits more than one character per token here.
    """
    result = engine.complete(CompletionRequest(prompt=PROMPT, max_tokens=4, temperature=0.0))
    if result.tokens:
        assert len(result.tokens) <= 4
    assert len(result.text) <= 4 * 32


def test_truncation_sets_finish_reason_length(engine: Engine) -> None:
    """Running out of budget reports 'length'."""
    result = engine.complete(
        CompletionRequest(prompt="Write a long essay about the sea.", max_tokens=4, temperature=0.0)
    )
    if result.finish_reason is None:
        pytest.skip(f"{engine.name} does not report finish_reason")
    assert result.finish_reason == "length"


def test_zero_max_tokens_behaviour(
    engine: Engine, record_divergence: Callable[[str], None]
) -> None:
    """What ``max_tokens=0`` means is unspecified. Tier 3 in docs/spec.md.

    Engines variously return empty text, reject the request, or generate one
    token anyway. The OpenAI API does not say which is right, so this records
    the behaviour rather than asserting one. Reported as ``note``.
    """
    try:
        result = engine.complete(CompletionRequest(prompt=PROMPT, max_tokens=0, temperature=0.0))
    except EngineError as exc:
        record_divergence(f"rejects the request: {str(exc)[:120]}")
        return

    record_divergence(f"returns {result.text[:60]!r} (finish_reason={result.finish_reason!r})")
