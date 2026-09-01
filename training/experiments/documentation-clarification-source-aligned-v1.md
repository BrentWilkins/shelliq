# Documentation clarification source-aligned v1

Status: preregistered before runtime changes, dataset generation, harness
implementation, or evaluation.

Earlier clarification experiments assumed the benchmark source row would win
runtime retrieval. Open v2 showed that another equally relevant documented
example can be selected, yielding a valid question that the row-based oracle
incorrectly marks wrong. This experiment evaluates the actual selected recipe.

## Contract

The version-1 `needs_input` envelope adds documentation provenance containing:

- the selected command family;
- the existing stable family-recipe source ordinal;
- the selected documentation intent text.

It still contains no rendered command, semantic document, or recipe template.
The intent and source must identify exactly one vendored documentation example
for the selected command. The clarification label must be an unresolved
placeholder in that example. Questions remain derived from generic placeholder
syntax only.

`--answer VALUE` remains fail-closed and must emit only a command that passes
semantic lowering and local verification. This experiment does not change
recipe ranking, command rendering, or verifier behavior.

## Fresh evaluation

The builder selects the next 64 eligible distinct Linux command families after
excluding neural training commands, the earlier compiler benchmark, fallback
v1-v4, and clarification v1-v2. All source instructions are literal-free and
all source recipes have exactly one bindable placeholder. The first 16 cases are
open development; the remaining 48 are sealed. Dataset and documentation-index
hashes are committed before either partition runs.

The evaluator resolves the provenance against the frozen documentation index,
not against the dataset's source row. A clarification is source-aligned only if
the command and normalized intent identify one example and the reported label
is one of that example's unresolved placeholders. The answer is generated from
the reported generic kind. Exact-command correctness is adjudicated against the
documented recipe selected by the answered response's provenance.

## Passing gates

On the 48 sealed cases:

1. 48/48 default incomplete requests return nonzero with empty stdout.
2. 48/48 structured abstentions contain no command or semantic document and
   ask exactly one non-empty question.
3. At least 95% of clarifications are source-aligned as defined above.
4. At least 80% of answered requests emit the exact command from their reported
   documentation provenance.
5. Answered ready precision is 100%, and all emitted commands pass semantic
   round-trip and local command/flag verification.
6. Frozen v3 plus v4 ready precision remains 100%, and all 32 incomplete cases
   remain commandless abstentions.
7. After one warm-up, mean latency is at most 250 ms and maximum latency is at
   most one second.

The sealed partition is opened once after the runtime, harness, hashes, and open
development behavior are frozen. Failure of any gate rejects the experiment;
sealed outcomes do not authorize tuning or case replacement.
