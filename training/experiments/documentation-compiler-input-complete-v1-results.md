# Documentation compiler input-complete v1 results

Status: failed the capability gate, with 72.7% ready precision.

The benchmark contains 100 distinct TLDR command families absent from the
610-record neural-training split. The compiler returned 22 ready documents, all
Rust-valid; 16 were exact, for 72.7% ready precision and 16% exact coverage.
Another 34 exact templates were produced but conservatively withheld, yielding
50/100 template-exact candidates before readiness.

Appending placeholder labels such as `path/to/file` beside concrete values made
those labels look like additional request paths to the generic literal extractor.
Multi-source compositions also won ties over an exact single documentation
template. V2 may remove slot labels while retaining values in document order and
prefer fewer sources after option and literal completeness. No command-specific
change is authorized.

Benchmark SHA-256:
`6b6de50bcf5d868012d0c7b9b74247744b923d163c6ce7d14c12f2e538834424`.

Raw report SHA-256:
`68152575594ddcafab87a5a0d64a929c2a88fb6882f15a93c16829991548c764`.
