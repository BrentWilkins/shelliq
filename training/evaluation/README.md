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
