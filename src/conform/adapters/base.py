"""The interface every engine adapter implements.

Adding support for a new engine means writing one subclass of :class:`Engine`
and registering it. Nothing else in the suite changes.
"""

from __future__ import annotations

import abc

from conform.types import Capability, CompletionRequest, CompletionResult, Message


class Engine(abc.ABC):
    """One inference engine, as the suite sees it."""

    #: Short identifier used as the column header in the report matrix.
    name: str = "unnamed"

    #: Which optional features this engine supports. Tests needing anything
    #: not listed here are skipped rather than failed.
    capabilities: frozenset[Capability] = frozenset()

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    # -- generation --------------------------------------------------------

    @abc.abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResult:
        """Generate from a raw prompt, with no chat template applied."""

    # -- optional surface --------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Text to token ids. Requires :attr:`Capability.TOKENIZE`."""
        raise NotImplementedError(f"{self.name} does not expose a tokenizer")

    def decode(self, tokens: list[int]) -> str:
        """Token ids back to text. Requires :attr:`Capability.TOKENIZE`."""
        raise NotImplementedError(f"{self.name} does not expose a tokenizer")

    def render_chat(self, messages: list[Message]) -> str:
        """Apply the model's chat template, returning the prompt string.

        Requires :attr:`Capability.CHAT_TEMPLATE`. Kept separate from
        generation so the template itself can be compared across engines
        without generating a single token.
        """
        raise NotImplementedError(f"{self.name} does not expose chat templating")

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:  # noqa: B027 - optional by design
        """Release sockets, unload weights. Safe to call more than once.

        Concrete and empty on purpose: an adapter holding no resources
        should not be forced to implement a stub.
        """

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
