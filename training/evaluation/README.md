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
