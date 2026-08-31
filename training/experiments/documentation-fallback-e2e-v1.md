# Documentation fallback end-to-end v1

Status: preregistered before runtime implementation or evaluation.

## Objective

Integrate the generic documentation compiler into the shipped Rust `shelliq
suggest` path as a fail-closed fallback. A model failure may produce an editable
command only when a vendored TLDR recipe can be fully bound from request-visible
values, lowered to `SemanticDocumentV2`, and accepted by the local installed-
command and option verifier. Missing operands must return an explicit
`needs_input` abstention with no command on standard output.

## Frozen data

The deterministic builder selects 96 distinct Linux command families in stable
TLDR record-id order. Every recipe is a simple single-command typed template
with at least one placeholder. Commands present in either the 610-record neural
training partition or the prior 100-family input-complete benchmark are excluded.

- Open development: 24 input-complete and 8 deliberately incomplete requests.
- Sealed test: 48 input-complete and 16 deliberately incomplete requests.
- Commands are disjoint across development and test as well as from both excluded
  sources above.

The committed manifest records the exact JSONL and documentation-index hashes.
The sealed rows must not be inspected or executed until implementation and all
development decisions are frozen.

## Required runtime path

Each case runs the actual model-feature `shelliq suggest` binary against a fresh
SQLite index and fake executable identity built through the public `index build`
CLI. The model endpoint is deliberately unavailable, forcing the production
fallback. The measured path therefore includes installed-command retrieval,
vendored TLDR retrieval, placeholder binding, semantic lowering and validation,
local command/option verification, CLI output and exit behavior.

No case-specific command rule, test-record lookup, expected-output lookup, or
test-command allow-list is permitted in runtime code.

## Frozen gates

Development is diagnostic. After development is frozen, run the sealed test once.
The test passes only if all of these hold:

1. At least 39/48 input-complete requests return the exact expected command.
2. At least 95% of commands emitted for input-complete requests are exact.
3. Every emitted command is valid `SemanticDocumentV2` and passes the local
   installed-command and option verifier.
4. All 16 incomplete requests exit nonzero, report `needs_input`, and write no
   command to standard output.
5. Mean complete-request latency is at most 250 ms and maximum latency is at most
   1 second on this development machine, excluding one warm-up invocation.

If the sealed gate fails, record the result and do not tune against its individual
outcomes. A new architecture and independently selected test would be required.
