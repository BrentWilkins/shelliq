# Semantic-action compiler rules v2 outer-validation results

Status: failed the preregistered outer gate; stop this recipe.

The frozen hybrid produced 1/78 exact and reference-accepted outputs, 78/78
Rust-valid outputs, and 69/78 first-command matches. The outer gate required at
least 8 accepted, 71 valid, and 63 first-command matches. Validity and command
identity passed, but the primary acceptance requirement failed.

The unchanged compiler activated on 0/78 outer-validation records, so every
output came from the grounded-count-v2 neural fallback. This provides no outer
evidence that the four inner-developed `ss` and `pmap` rule families generalize
to the protected validation distribution. Per the frozen decision rule, do not
tune those rules against outer outcomes and do not open the protected test split.

Training used all 610 outer-training records for the inner-selected five epochs
with seed `20260826`; no outer-validation loss was computed during training. The
single protected generation-and-scoring pass followed training.

Raw outer report SHA-256:
`3e457bf87ad30760192581bd2d43807522e2082abac4b258db15ea8956a64c70`.

Checkpoint SHA-256:
`f6248cf7aeb0bebb56853b60b4bb6633f7b665948fb549f6dcfc7336871957e7`.
