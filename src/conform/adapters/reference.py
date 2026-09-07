"""The reference oracle: HuggingFace transformers on CPU, in float32.

This is the slowest possible way to run a model and that is the point. No
custom kernels, no batching tricks, no fused attention, no quantisation. Every
optimisation a real engine makes is a chance to diverge, and this adapter
makes none of them.

A word on what "reference" means here. Where the OpenAI API documents a
behaviour, this adapter implements the documented behaviour and a mismatch is
a bug in the other engine. Where the API is silent or ambiguous, this adapter
is only *a* reasonable reading, and the suite records the disagreement as a
divergence without claiming anyone is wrong. See docs/spec.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from conform.adapters.base import Engine
from conform.types import Capability, CompletionRequest, CompletionResult, Message

if TYPE_CHECKING:  # pragma: no cover
    pass

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


class ReferenceEngine(Engine):
    """Ground truth, at the cost of speed."""

    name = "reference"
    capabilities = frozenset(
        {
            Capability.TOKENIZE,
            Capability.STOP_SEQUENCES,
            Capability.CHAT_TEMPLATE,
            Capability.TOP_K,
        }
    )

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        # Imported lazily so that `import conform` stays cheap and the HTTP
        # adapters remain usable without torch installed at all.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_id = model
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(
            model,
            dtype=torch.float32,
            device_map="cpu",
        )
        self._model.eval()

    # -- generation --------------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResult:
        torch = self._torch
        inputs = self._tokenizer(request.prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[-1]

        kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_tokens,
            "do_sample": request.temperature > 0.0,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        }
        if kwargs["do_sample"]:
            kwargs["temperature"] = request.temperature
            kwargs["top_p"] = request.top_p
            if request.top_k is not None:
                kwargs["top_k"] = request.top_k
        if request.seed is not None:
            torch.manual_seed(request.seed)

        with torch.no_grad():
            output = self._model.generate(**inputs, **kwargs)

        new_tokens = output[0][prompt_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        finish_reason = "length" if len(new_tokens) >= request.max_tokens else "stop"

        # Stop strings are applied after decoding, and the stop string itself
        # is excluded. That matches the OpenAI API, which documents the stop
        # sequence as not being returned.
        if request.stop:
            cut = _earliest_stop(text, request.stop)
            if cut is not None:
                text = text[:cut]
                finish_reason = "stop"

        return CompletionResult(
            text=text,
            tokens=tuple(int(t) for t in new_tokens),
            finish_reason=finish_reason,
            raw={"model": self.model_id},
        )

    # -- tokenizer ---------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, tokens: list[int]) -> str:
        return str(self._tokenizer.decode(tokens, skip_special_tokens=False))

    def render_chat(self, messages: list[Message]) -> str:
        return str(
            self._tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        )


def _earliest_stop(text: str, stops: tuple[str, ...]) -> int | None:
    """Index of the earliest stop string in ``text``, or None if absent."""
    hits = [text.find(s) for s in stops if s]
    hits = [i for i in hits if i >= 0]
    return min(hits) if hits else None
