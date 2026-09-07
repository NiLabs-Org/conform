"""Core types shared by every adapter and every test.

The whole suite speaks in these types. An adapter's job is to translate them
to and from whatever its engine actually wants, so that a test never has to
know which engine it is running against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Capability(StrEnum):
    """Optional engine features.

    Not every engine exposes every feature. A test that needs one it does not
    have is reported as ``skip``, never ``fail``. Missing a feature is not a
    conformance bug; getting it wrong is.
    """

    TOKENIZE = "tokenize"
    """Exposes the tokenizer directly, so round-trips can be checked."""

    STOP_SEQUENCES = "stop_sequences"
    """Accepts stop strings and honours them."""

    LOGPROBS = "logprobs"
    """Returns per-token log probabilities."""

    SEEDED_SAMPLING = "seeded_sampling"
    """Accepts a seed and makes sampling reproducible."""

    TOP_K = "top_k"
    """Accepts a top_k sampling parameter."""

    CHAT_TEMPLATE = "chat_template"
    """Applies the model's chat template to a message list."""

    JSON_SCHEMA = "json_schema"
    """Constrains output to a JSON schema."""


@dataclass(frozen=True)
class Message:
    """One turn in a chat conversation."""

    role: str
    content: str


@dataclass(frozen=True)
class CompletionRequest:
    """A generation request, in the suite's own vocabulary.

    Defaults are deliberately the most deterministic setting available:
    greedy decoding, no penalties, no truncation of the distribution. A test
    that cares about sampling opts in explicitly.
    """

    prompt: str
    max_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None
    logprobs: int | None = None


@dataclass(frozen=True)
class TokenLogprob:
    """One token and the log probability the engine assigned to it."""

    token: str
    logprob: float


@dataclass(frozen=True)
class CompletionResult:
    """What an engine gave back, normalised.

    ``raw`` keeps the untouched engine response so a failing test can show
    what actually came over the wire. It is excluded from equality so two
    results compare on their normalised content alone.
    """

    text: str
    tokens: tuple[int, ...] = ()
    finish_reason: str | None = None
    logprobs: tuple[TokenLogprob, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


class EngineError(RuntimeError):
    """An engine failed to answer at all.

    Distinct from a wrong answer. This means the request errored, timed out,
    or the server was unreachable, and the test cannot draw a conclusion.
    """
