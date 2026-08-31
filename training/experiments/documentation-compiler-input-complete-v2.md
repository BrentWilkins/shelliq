# Documentation compiler input-complete benchmark v2

Status: preregistered generic interface correction.

V2 retains v1's filtered TLDR index, exclusion of neural-training commands,
stable one-record-per-command selection, eligibility rules, deterministic typed
values, target-blind binding, Rust validation, 100-command limit, and 80%
coverage/95% precision gate.

Two corrections are frozen:

1. Append concrete values in typed-slot order without appending placeholder
   labels. This matches an input-complete user request and prevents labels from
   being reinterpreted as extra literal values.
2. After requested-option and requested-literal completeness, prefer a candidate
   using fewer documentation sources before unsupported-operand and similarity
   tie breaks. A single complete typed recipe is less speculative than a merged
   recipe.

No reference-derived or command-specific features are allowed.
