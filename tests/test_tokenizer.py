"""Tokenizer conformance.

Tokenizers look like the boring part and are a reliable source of silent
wrongness. Engines reimplement them: llama.cpp reads its tokenizer out of the
GGUF file rather than using HuggingFace's, so the two implementations have to
be kept in agreement by hand, and periodically they are not.

A tokenizer bug does not raise. It shifts your text by one space, drops a
special token, or mangles a code point, and everything downstream quietly
degrades.
"""

from __future__ import annotations

import pytest

from conform.adapters.base import Engine

pytestmark = pytest.mark.requires("tokenize")


ROUND_TRIP_CASES = [
    pytest.param("hello", id="plain"),
    pytest.param(" hello", id="leading-space"),
    pytest.param("hello ", id="trailing-space"),
    pytest.param("  hello", id="double-leading-space"),
    pytest.param("hello\n", id="trailing-newline"),
    pytest.param("hello\n\nworld", id="blank-line"),
    pytest.param("\t indented", id="tab"),
    pytest.param("héllo", id="accented"),
    pytest.param("日本語", id="cjk"),
    pytest.param("🙂", id="emoji"),
    pytest.param("👨‍👩‍👧‍👦", id="zwj-emoji"),
    pytest.param("a" * 500, id="long-run"),
    pytest.param("def f(x):\n    return x + 1\n", id="code"),
    pytest.param("<|im_start|>", id="special-token-literal"),
]


@pytest.mark.parametrize("text", ROUND_TRIP_CASES)
def test_roundtrip_is_lossless(engine: Engine, text: str) -> None:
    """decode(encode(x)) should give back exactly x.

    Whitespace at the edges is where this usually breaks, because many
    tokenizers normalise a leading space onto the following token and then
    fail to put it back.
    """
    assert engine.decode(engine.encode(text)) == text


def test_empty_string_encodes_to_nothing(engine: Engine) -> None:
    """The empty string should produce no tokens, not a lone special token."""
    assert engine.encode("") == []


def test_encoding_is_deterministic(engine: Engine) -> None:
    """The same text must tokenize identically every time."""
    text = "The quick brown fox jumps over the lazy dog."
    assert engine.encode(text) == engine.encode(text)


def test_concatenation_does_not_lose_characters(engine: Engine) -> None:
    """Tokenizing halves and joining should preserve the characters.

    Token boundaries legitimately differ from tokenizing the whole string at
    once, so this checks the decoded text rather than the ids.
    """
    left, right = "The quick brown ", "fox jumps."
    joined = engine.decode(engine.encode(left)) + engine.decode(engine.encode(right))
    assert joined == left + right
