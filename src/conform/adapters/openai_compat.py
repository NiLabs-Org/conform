"""Adapter for any engine exposing an OpenAI-compatible HTTP API.

vLLM, SGLang, TGI, llama.cpp's server and Ollama all speak some version of
this API, which is precisely why the suite is tractable: one adapter reaches
most of the ecosystem, and only the base URL changes.

"Some version of" is doing real work in that sentence. The differences are
what the suite exists to find, so this adapter stays deliberately literal. It
sends what the request says and reports what came back. It does not paper
over quirks, because papering over quirks is how you end up not measuring
them.
"""

from __future__ import annotations

from typing import Any

import httpx

from conform.adapters.base import Engine
from conform.types import (
    Capability,
    CompletionRequest,
    CompletionResult,
    EngineError,
    TokenLogprob,
)

DEFAULT_TIMEOUT = 120.0


class OpenAICompatEngine(Engine):
    """Talks to a `/v1/completions` endpoint over HTTP."""

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        *,
        api_key: str = "not-needed",
        capabilities: frozenset[Capability] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.capabilities = (
            capabilities
            if capabilities is not None
            else frozenset({Capability.STOP_SEQUENCES, Capability.LOGPROBS, Capability.TOP_K})
        )
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    # -- generation --------------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        # Only send optional fields when the test actually set them. Sending
        # nulls makes some servers 400 and others silently pick a default,
        # which would be a difference the suite created rather than found.
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.top_k is not None:
            payload["top_k"] = request.top_k
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.logprobs is not None:
            payload["logprobs"] = request.logprobs

        data = self._post("/v1/completions", payload)

        try:
            choice = data["choices"][0]
        except (KeyError, IndexError) as exc:
            raise EngineError(f"{self.name}: response had no choices: {data!r}") from exc

        return CompletionResult(
            text=choice.get("text", ""),
            finish_reason=choice.get("finish_reason"),
            logprobs=_parse_logprobs(choice.get("logprobs")),
            raw=data,
        )

    # -- tokenizer ---------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        if not self.supports(Capability.TOKENIZE):
            raise NotImplementedError(f"{self.name} does not expose a tokenizer")
        data = self._post("/tokenize", {"model": self.model, "prompt": text})
        return list(data["tokens"])

    def decode(self, tokens: list[int]) -> str:
        if not self.supports(Capability.TOKENIZE):
            raise NotImplementedError(f"{self.name} does not expose a tokenizer")
        data = self._post("/detokenize", {"model": self.model, "tokens": tokens})
        return str(data["prompt"])

    # -- plumbing ----------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise EngineError(f"{self.name}: {path} unreachable: {exc}") from exc

        if response.status_code >= 400:
            raise EngineError(
                f"{self.name}: {path} returned {response.status_code}: {response.text[:400]}"
            )
        body = response.json()
        if not isinstance(body, dict):
            raise EngineError(f"{self.name}: {path} returned a non-object body: {body!r}")
        return body

    def close(self) -> None:
        self._client.close()


def _parse_logprobs(block: dict[str, Any] | None) -> tuple[TokenLogprob, ...]:
    """Pull the flat token/logprob pairs out of an OpenAI logprobs block."""
    if not block:
        return ()
    tokens = block.get("tokens") or []
    values = block.get("token_logprobs") or []
    return tuple(
        TokenLogprob(token=token, logprob=value)
        for token, value in zip(tokens, values, strict=False)
        if value is not None
    )
