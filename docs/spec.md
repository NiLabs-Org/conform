# What counts as correct

A conformance suite is only as useful as its answer to "says who?". This is that
answer.

## Three tiers

Every test in this suite falls into one of three categories, and the category
determines what a failure means.

### Tier 1: Specified

The OpenAI API documents the behaviour, or it follows from the definition of the
operation. A mismatch is a bug in the engine.

Examples:

- `stop` sequences are not included in the returned text
- `max_tokens` bounds generated tokens and does not count the prompt
- `finish_reason` is `"length"` on truncation and `"stop"` on a stop sequence
- `decode(encode(x)) == x`, which follows from what a tokenizer is

Tests in this tier assert directly and a failure is reported as `FAIL`.

### Tier 2: Implied

Not written down anywhere, but only one answer is defensible.

Examples:

- Greedy decoding is deterministic across identical requests
- Greedy decoding is unaffected by `top_p`, because the distribution is already
  a point mass
- Tokenizing the same string twice gives the same ids

These are asserted as failures too, but if a maintainer pushes back with a
coherent argument, the test moves to tier 3 rather than the argument being
dismissed. That has not happened yet. It probably will.

### Tier 3: Unspecified

Engines genuinely differ and no authority settles it. These are recorded as
**divergences**, not failures. The matrix shows what each engine does; it does
not name a winner.

Examples:

- Whether `top_k` is applied before or after `top_p`
- Whether `repetition_penalty` covers prompt tokens or only generated ones
- Whether `logprobs` are reported before or after temperature scaling
- What `max_tokens=0` means: empty output, an error, or one token

Divergences are arguably the most valuable output of the project. A `FAIL` tells
one team to fix something. A divergence tells the whole ecosystem that a
behaviour needs specifying.

## The reference implementation

`ReferenceEngine` runs the model through HuggingFace transformers, on CPU, in
float32, batch size one, with no custom kernels, no quantisation and no fused
attention. It is the slowest correct thing we can build, and it makes none of
the optimisations that cause divergence.

Its authority is bounded:

- **Tier 1**: the reference implements the documented behaviour. If it does not,
  that is a bug in the reference, and it gets fixed
- **Tier 2**: the reference is strong evidence
- **Tier 3**: the reference is just one more data point, with no special status

Being able to say "the reference is wrong here" is a feature. If you think it
is, open an issue.

## Tolerances

Floating-point comparisons use a documented tolerance in the test itself, never
a global default.

Token-identity comparisons have no tolerance. Either the same tokens came out or
they did not.

## Versioning results

A published matrix is meaningless without exact versions. Every result records
the engine version, the model id, and the quantisation. A result from a
different engine version is a different result, not an update.

## When an engine fixes a divergence

The test stays. It becomes a regression guard, which is the second-most useful
thing a conformance suite does.
