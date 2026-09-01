# Documentation clarification continuation v1

Status: preregistered before runtime changes, dataset generation, harness
implementation, or evaluation.

The source-aligned clarification experiment passed, but 3/48 answered requests
remained safely incomplete. One selected recipe required more than one missing
value; two generic TLDR placeholders could be reported but not rebound because
their labels did not imply a known type. Free-text `--answer` also reruns global
retrieval even though the response already identifies the selected recipe.

## Contract

A documentation-backed `needs_input` envelope adds a compact continuation:

```json
{"v":1,"source":"tldr:linux:command:ordinal"}
```

The CLI accepts `--continue-from SOURCE` with one or more repeatable
`--answer VALUE` options. It validates the source against vendored documentation
and compiles only that exact recipe. It never accepts shell text, a template, or
a rendered command from the continuation. Invalid, stale, ambiguous, or
unavailable sources fail closed.

Answers are appended as typed request values and passed through the existing
generic binder. A selected recipe with further unresolved inputs returns another
commandless `needs_input` response with the same continuation and one focused
question. Clients retain the original instruction and accumulated answers until
the response becomes `ready`. Every placeholder the runtime reports must accept
an answer: label-derived kinds take precedence, numeric exemplars infer integer,
and otherwise unknown TLDR placeholders use the generic text kind.

No command-family rules are allowed. Source-pinned compilation still requires
semantic lowering and local command/flag verification before emitting a command.
Ranking, unpinned fallback behavior, and verifier policy remain unchanged.

## Fresh evaluation

The builder selects the next 64 eligible distinct Linux command families after
excluding neural training commands, the earlier compiler benchmark, fallback
v1-v4, clarification v1-v2, and source-aligned v1. Source instructions remain
literal-free; each dataset source recipe begins with one bindable placeholder,
but the runtime-selected recipe may contain multiple unresolved operands. The
first 16 cases are open development and the remaining 48 are sealed. Dataset and
documentation-index hashes are committed before either partition runs.

For each incomplete request, the evaluator resolves the reported source and
label against frozen documentation, generates an answer from the reported kind,
and calls the CLI again with the same source plus every accumulated answer. It
repeats for at most eight distinct questions. Repeated source/label states,
source changes, malformed continuations, or any premature command are failures.

## Passing gates

On the 48 sealed cases:

1. 48/48 initial requests and every intermediate step are nonzero, commandless
   structured abstentions with exactly one focused question.
2. 48/48 initial continuations are source-aligned, and every step retains the
   same valid documentation source.
3. At least 95% reach `ready` within eight distinct questions.
4. Ready precision is 100% against the pinned documented recipe and accumulated
   answers; every emitted command passes semantic round-trip and local
   command/flag verification.
5. Invalid continuation fixtures all fail closed with no stdout command.
6. Frozen source-aligned v1 retains 100% ready precision and commandless
   abstentions; frozen v3 plus v4 retain 100% ready precision and all 32
   incomplete cases remain commandless.
7. Mean complete-flow latency is at most 500 ms and maximum at most two seconds.

The sealed partition is opened once after runtime, harness, hashes, and open
development behavior are frozen. Failure of any gate rejects the experiment;
sealed outcomes do not authorize tuning or case replacement.
