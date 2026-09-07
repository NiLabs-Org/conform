# Security Policy

## Scope

`conform` is a test harness. It does not run in production, does not handle
credentials beyond an optional API key you supply, and does not execute
model-generated code.

The realistic risks are therefore narrow, and these are the ones worth
reporting:

- The suite sending an API key somewhere other than the configured `base_url`
- A crafted engine response causing code execution in the harness, for example
  through unsafe deserialization
- A malicious `conform.toml` or test fixture achieving code execution beyond
  what running `pytest` already implies
- Credentials leaking into `results/matrix.json`, log output, or CI artifacts

Out of scope:

- Vulnerabilities in the engines being tested. Report those to the engine, not
  here. If you would like a regression test added afterward, open a normal issue
- Vulnerabilities in models, weights, or their content
- The fact that `pytest` executes Python from the repository. That is what a
  test runner does

## Reporting

Please do not open a public issue for a security report.

Use GitHub's [private vulnerability
reporting](https://github.com/NiLabs-Models/conform/security/advisories/new),
or email **hello@nilabs.dev**.

Include the version or commit, what an attacker can achieve, and a reproduction
if you have one.

## What to expect

This project is maintained by one person, so response is best-effort rather than
contractual. Expect an acknowledgement within about a week. Fixes for confirmed
issues land as soon as practical, and you will be credited in the release notes
unless you would rather not be.

## Supported versions

`conform` is pre-1.0. Only the `main` branch is supported. There are no
backports.
