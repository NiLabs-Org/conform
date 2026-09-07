# AGENTS.md

Guidance for AI coding agents working in this repository. Humans should read
[CONTRIBUTING.md](CONTRIBUTING.md) first; this file only covers what is easy for
an agent to get wrong here.

## What this project is

A conformance suite that finds behavioural disagreements between LLM inference
engines. It is a test suite, not a library. The output is a matrix of which
engines diverge, not a passing build.

## The rule that matters most

**A failing test may be correct.**

In most repositories a red test means fix the code. Here it usually means an
engine really does behave differently, which is the finding the project exists
to produce.

Never do any of the following to make a test pass:

- Loosen an assertion so a real divergence stops being reported
- Add a retry, a tolerance, or a normalisation step that hides a difference
- Special-case an engine by name inside a test
- Mark a genuinely failing test `xfail` or `skip` to get the suite green

If a test fails, work out whether the test encodes the correct expectation. If
it does, leave it failing and say so. If it does not, fix the expectation and
explain why in the docstring.

## Setup

```bash
pip install -e ".[reference,dev]"
cp conform.example.toml conform.toml
```

Checks, all of which must pass:

```bash
ruff format --check .
ruff check .
mypy
pytest
```

With no engines reachable, the suite skips rather than fails. That is correct
behaviour and not something to fix.

## Layout

```
src/conform/types.py              shared request/result types
src/conform/registry.py           builds engines from conform.toml
src/conform/report.py             pytest plugin producing the matrix
src/conform/cli.py                `conform` command
src/conform/adapters/base.py      the Engine interface
src/conform/adapters/reference.py transformers on CPU, the oracle
src/conform/adapters/openai_compat.py  any OpenAI-compatible HTTP server
tests/                            the conformance tests themselves
```

## Conventions

- Python 3.11+, `from __future__ import annotations` at the top of every module
- Ruff, line length 100. Do not reformat files you did not otherwise change
- `mypy --strict` passes on `src/`. Tests are not strictly typed
- Type hints on every public function
- Comments explain *why*. The code already says what

## Writing tests

Every test takes the `engine` fixture and runs once per configured engine.

Required, not optional: a docstring naming the concrete bug the test catches
and why an engine would plausibly get it wrong. A test whose docstring restates
its assertion in English adds nothing. Compare:

```python
# useless
"""Checks that decode(encode(x)) equals x."""

# useful
"""llama.cpp reimplements tokenization from GGUF rather than using HF's, and
the two have drifted repeatedly. Leading whitespace is where it usually
breaks, because many tokenizers fold a leading space onto the next token and
then fail to restore it."""
```

Default to `temperature=0.0`. If a test needs an optional feature, mark it
`@pytest.mark.requires("capability")` so engines lacking it are skipped rather
than failed.

## Adapters

Adapters must be literal. Send what the request says, return what came back. Do
not normalise, retry, or work around engine quirks. Declare `capabilities`
accurately: under-declaring produces false skips, over-declaring produces false
failures.

## Do not

- Add dependencies without asking. The core stays `httpx` and `pytest`; heavy
  things go under an optional extra
- Edit `conform.toml`. It is gitignored and belongs to the user. Edit
  `conform.example.toml` instead
- Commit anything under `results/`
- Invent benchmark numbers, matrix results, or engine version claims. If you
  have not run it, say you have not run it. Fabricated results are the single
  worst possible contribution to a project whose only asset is trustworthy
  measurement

## Commits and pull requests

- Conventional-ish subjects: `test:`, `fix:`, `feat:`, `docs:`, `adapter:`
- One logical change per PR
- Disclose AI assistance in the PR description
- Explain in the PR body which divergence a new test catches, or that it catches
  none yet and is a guard
