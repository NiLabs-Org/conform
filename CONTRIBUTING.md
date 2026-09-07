# Contributing to conform

The most valuable contribution here is a test that catches a real disagreement
between two engines. Second most valuable is a divergence report that does not
have a test yet. Everything else is plumbing.

## Getting set up

Python 3.11 or newer.

```bash
git clone https://github.com/NiLabs-Org/conform
cd conform
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[reference,dev]"
cp conform.example.toml conform.toml
```

`conform.toml` is gitignored. Your local ports, models and enabled engines stay
yours.

Before opening a pull request:

```bash
ruff format --check .
ruff check .
mypy
pytest tests/harness      # self-tests, must pass
conform run               # conformance, failures here may be findings
```

Note the difference. `tests/harness` checks the suite itself against a stub
server and must be green. The conformance tests check other people's software,
and a failure there is often a result rather than a problem.

## Writing a test

Every test takes the `engine` fixture and is run once per configured engine.
You write it once; it becomes a row for every engine.

```python
def test_stop_string_is_excluded(engine: Engine) -> None:
    """The OpenAI API documents the stop sequence as not returned."""
    result = engine.complete(
        CompletionRequest(prompt="Letters: a b c d", max_tokens=32, temperature=0.0, stop=("c",))
    )
    assert "c" not in result.text
```

Four rules, and they matter more than style:

**1. Test one behaviour.** A row in the matrix should mean one thing. If a test
can fail for two unrelated reasons, split it.

**2. Say why in the docstring.** Not what the code does, which is visible, but
which real bug this catches and why an engine would plausibly get it wrong. That
docstring is what a maintainer reads when your test fails on their engine at
2am. It is the most important line you will write.

**3. Be deterministic.** Default to `temperature=0.0`. If you must sample, set a
seed and mark the test `@pytest.mark.requires("seeded_sampling")`. A flaky test
in a conformance suite is worse than no test, because it teaches people to
ignore red.

**4. Skip, do not fail, on missing features.** Mark tests that need an optional
feature:

```python
@pytest.mark.requires("logprobs")
def test_logprobs_are_normalised(engine: Engine) -> None: ...
```

Not implementing a feature is not a conformance bug. Implementing it wrongly is.
Conflating the two makes the matrix useless.

## Writing an adapter

Only if the engine does not speak OpenAI-compatible HTTP. Otherwise just add a
block to `conform.toml`.

Subclass `Engine` in `src/conform/adapters/`, implement `complete()`, and
declare `capabilities` honestly. Under-declaring gives you false skips;
over-declaring gives you false failures.

Adapters must be **literal**. Do not normalise a quirk, retry a bad response, or
work around a known bug. Every such kindness hides exactly what the suite exists
to find. If an engine returns something strange, return the strange thing.

## What the reference is, and is not

`ReferenceEngine` runs the model with HuggingFace transformers on CPU in float32
with no optimisations. It is the slowest correct thing we can build.

It is authoritative where the OpenAI API documents a behaviour. Where the API is
silent, it is one reasonable reading and nothing more. If you think the
reference is wrong, say so in an issue. Being able to change the reference is a
feature.

## Reporting a divergence

Use the divergence issue template. Include:

- The exact request, as JSON
- Both outputs, verbatim, not paraphrased
- Engine names and **exact versions**, plus the model and its quantisation
- Which behaviour you believe is correct, and the docs that say so

"Model gave a weird answer" is not a divergence. Two engines given identical
inputs producing different outputs is.

## Pull requests

- One logical change per PR
- Branches: `test/`, `fix/`, `feat/`, `docs/`, `adapter/`
- New tests must include the docstring rationale described above
- If a test fails against a real engine, that is a finding, not a blocker. Say so
  in the description and link an upstream issue if you filed one

We do not require you to have found the root cause. Reporting that two engines
disagree is a complete contribution on its own.

## AI-assisted contributions

Allowed and welcome, with two conditions.

1. Disclose it in the pull request description.
2. Understand what you are submitting. You will be asked about it in review, and
   "the model wrote it" closes the PR.

See [AGENTS.md](AGENTS.md) if you are pointing a coding agent at this repo.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
