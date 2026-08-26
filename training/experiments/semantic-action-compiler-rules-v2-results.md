# Semantic-action compiler rules v2 results

Status: passed the preregistered inner gate.

The hybrid produced 6/92 exact and reference-accepted outputs, 92/92
Rust-valid outputs, and 79/92 first-command matches. The frozen gate required at
least 5 accepted, 83 valid, and 74 first-command matches.

All six rule activations were exact and accepted: one `ss` summary, three
documented `ss` listings, one extended `pmap` request, and one kernel-detail
`pmap` request. Every rule output passed through the project-owned Rust action
encoder. The remaining 86 records used the unchanged grounded-count-v2 neural
fallback.

This is evidence that a conservative compiler can recover a small,
high-confidence subset while preserving the learned model's coverage. It is
not yet evidence that these inner-developed rules generalize. Outer validation
and test remain sealed; promotion requires a separate outer-validation decision.

Raw inner report SHA-256:
`39982216e37666862d5243799dc8912e920402fefa2f0353b66f68ac5d9237ce`.
