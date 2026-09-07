## What this changes

<!-- One or two sentences. -->

## Type

- [ ] New conformance test
- [ ] New or updated engine adapter
- [ ] Harness fix
- [ ] Documentation
- [ ] Other

## For a new test

**Which divergence does it catch?**
<!-- Link a divergence issue, or say "none yet, this is a regression guard". -->

**Which tier?** (see `docs/spec.md`)
<!-- Specified / Implied / Unspecified -->

**Why would an engine plausibly get this wrong?**
<!-- This belongs in the test docstring too. -->

## Results

<!--
Paste the relevant rows of your matrix. If a test fails against a real engine,
that is a finding, not a blocker: say so, and link an upstream issue if you
filed one.
-->

```
```

## Checklist

- [ ] `ruff format --check .` and `ruff check .` pass
- [ ] `mypy` passes
- [ ] New tests have a docstring explaining the concrete bug they catch
- [ ] Tests needing an optional feature are marked `@pytest.mark.requires(...)`
- [ ] No assertion was loosened to make a real divergence stop reporting
- [ ] I did not edit `conform.toml` (it is gitignored; edit the example instead)

## AI assistance

<!--
Required if you used an AI tool. State what it did. You will be asked about the
code in review.
-->
