# Evaluation annotations

This directory contains reviewed metadata used only to score model outputs. It
does not alter training examples.

`curated-option-arities-v1.json` maps curated `record_id` values to the flags in
their expected command and whether each flag consumes the following token:

- `0`: the flag consumes no following argument.
- `1`: the flag requires one following argument.

Mappings are record-scoped because subcommands can give the same spelling
different semantics. The checkpoint evaluator rejects unknown records, values
other than integer `0` or `1`, and simple-command annotations whose flag set is
missing or has extra entries. Compound expected commands remain outside the
basic flag/operand grammar and are reported separately.

The annotation file is intentionally manual and versioned. Do not infer arity
from token position: a boolean flag followed by an operand is indistinguishable
from an argument-taking flag without command documentation.

`curated-grounding-v1.json` identifies literal values that the instruction does
not specify, such as the source and destination in `cp -a src/ dest/`. It uses
JSON Pointer paths into `SemanticDocumentV2`; for example:

- `/s/0/c/0/a/1/s` means statement 0 (`s`), command stage 0 (`c`), argument 1
  (`a`), literal string (`s`).
- `/s/0/c/0/r/0/t/s` means redirect 0 (`r`), target (`t`), literal string (`s`).

The grounded exact-match metric replaces only those reviewed string values in
both expected and generated documents. Commands, flags, argument count and
order, redirects, quoting structure, and every other AST field remain exact.
Every selected record is present even when its path list is empty, making the
audit a closed, reviewable set rather than a permissive fallback.

`pipeline-compatibility-v1.jsonl` is a frozen eight-row regression suite for
pipeline record shape and delimiter compatibility. Its real alpha failure
prompt and expected response are excluded from `corpus/pipeline-contracts.jsonl`;
the accompanying test enforces exact instruction and target separation. The
suite's grounding audit intentionally has no variable literal paths, so grounded
document matching remains strict for command stages, formats, operands, and
pipeline operators.

`semantic-shadow-release-v1.jsonl` is the untouched final gate for the 0.5B
LoRA-rank sweep: 20 cases, two each across ten command families absent from every
training corpus file. Its grounding audit has no variable literals, so every
command, flag, value, operand, and structure must match. The existing 35-row
release suite is development/model-selection data now. Do not evaluate rank
variants on the shadow suite; evaluate it once after development and retention
choose one winner. Never merge either evaluation file into training.

## Teacher-selection tournament

`teacher-selection-v1.jsonl` is a frozen 48-record comparison set for choosing a
teacher model. It contains 40 Linux and 8 Darwin challenges, balanced across four
instruction-complexity categories. It is selection data only: never merge it into
teacher-authored training data, and do not treat its score as a student release
claim.

The set intentionally excludes `tldr-pages`. Those examples are widely published
and likely to have appeared in foundation-model training corpora, which would bias
the tournament toward familiarity. The checked-in set comes only from reviewed
`shelliq-curated` records. Regenerate it deterministically with:

```sh
uv run python scripts/build_teacher_selection.py \
  --semantic-dataset artifacts/distributable-semantic-v2.jsonl \
  --output evaluation/teacher-selection-v1.jsonl
```

The builder refuses TLDR even if requested explicitly. Future revisions should
prefer newly authored, project-specific challenges so teacher selection measures
generalization rather than exposure to any existing corpus.
