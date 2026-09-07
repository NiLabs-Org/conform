# conform

A conformance suite for LLM inference engines.

vLLM, llama.cpp, Ollama, SGLang and TGI all claim to serve the same models. Feed
them identical inputs and they do not always produce identical outputs. The
disagreements are usually small, almost never documented, and get discovered one
confused bug report at a time.

Browsers solved this with a shared test suite. Inference engines do not have one.
This is an attempt at that.

## The idea

Send the same request to every engine. Compare against a reference
implementation that makes no optimisations. Publish the grid.

```
                                  reference  ollama  vllm-cpu  llama-cpp
test_roundtrip_is_lossless[cjk]     pass      FAIL     pass      pass
test_greedy_is_repeatable           pass      pass     FAIL      pass
test_stop_string_is_excluded        pass      FAIL     pass      FAIL
test_zero_max_tokens_generates_...  pass      skip     pass      FAIL
```

A test is written once and becomes a row for every engine automatically.

> **Status: early.** The harness works and the first tests are real, but the
> matrix above is illustrative, not measured. Do not cite it. Results land here
> once the suite has been run against released versions.

## What gets tested

| Area | Examples |
| --- | --- |
| Tokenizer | `decode(encode(x)) == x` across whitespace, unicode, emoji, code |
| Determinism | Greedy decoding repeatable; unaffected by `top_p` |
| Stop sequences | Excluded from output, matched across token boundaries |
| Length | `max_tokens` excludes the prompt, `finish_reason` is correct |
| Sampling | `top_k` and `top_p` ordering, seed reproducibility |
| Logprobs | Applied before or after temperature and penalties |
| Chat templates | Same messages render to the same prompt |
| Structured output | JSON schema constraints produce valid output |

The last three are not implemented yet. See the [issues](https://github.com/NiLabs-Org/conform/issues).

## Non-goals

- **Benchmarking.** This measures correctness, not throughput or latency.
- **Ranking engines.** A low score often means an engine exposes more surface
  area, not that it is worse.
- **Declaring winners on ambiguity.** Where the OpenAI API genuinely does not
  specify a behaviour, a difference is recorded as a divergence, not a failure.
  See [`docs/spec.md`](docs/spec.md).

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/NiLabs-Org/conform
cd conform
python -m venv .venv && .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[reference,dev]"
cp conform.example.toml conform.toml
```

Then point it at something. The easiest first engine is [Ollama](https://ollama.com):

```bash
ollama pull qwen2.5:0.5b
ollama serve
```

And run:

```bash
conform engines     # show what is configured
conform run         # run everything, write results/matrix.{json,md}
conform run -k tokenizer -v
```

The harness has its own self-tests, which run against a stub server and need
no engine, model or network:

```bash
pytest tests/harness
```

One of them points the suite at a deliberately broken server and asserts the
bug is caught. Without that, "everything passed" would be indistinguishable
from "nothing was reachable".

The first reference run downloads a 135M-parameter model, roughly 270MB. Every
run after that is offline.

## Adding an engine

Most engines need no code at all. If it speaks an OpenAI-compatible
`/v1/completions`, add a block to `conform.toml`:

```toml
[[engines]]
name = "my-engine"
kind = "openai"
base_url = "http://localhost:9000"
model = "Qwen/Qwen2.5-0.5B-Instruct"
capabilities = ["stop_sequences", "logprobs"]
```

Anything else means one subclass of `Engine` in `src/conform/adapters/`.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Reporting a divergence

Found two engines that disagree? That is the most valuable thing you can
contribute. Open a
[divergence report](https://github.com/NiLabs-Org/conform/issues/new?template=divergence.yml)
with the request, both outputs, and the versions.

## License

Apache-2.0. See [LICENSE](LICENSE).
