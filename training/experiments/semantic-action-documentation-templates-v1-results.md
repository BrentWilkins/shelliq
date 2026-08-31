# Semantic-action documentation templates v1 results

Status: failed the retrieval-feasibility gate; stop single-template retrieval.

The TLDR-only index contains 29,786 independently authored templates across
5,073 commands. Every template passed the project-owned Rust action
encode/decode round trip. On the 92-record open inner split, same-command and
same-platform documentation existed for 87 records. All 92 emitted fallback or
retrieved documents were Rust-valid, and 85 retained the correct first command.

Only 3/92 references had a matching typed skeleton at rank one and 5/92 at rank
eight, spanning five command families. The frozen gate required at least 19
top-eight matches across ten command families, so it failed. Bound top-one
templates produced 0/92 conservative reference acceptances.

The all-template attribution found five records without platform-compatible
documentation, one retrieval-rank miss, and 81 records for which no single TLDR
template matched the reference skeleton. Ranking is therefore not the dominant
failure. Curated requests commonly compose multiple documented flags, values,
redirections, or operands that do not coexist in one TLDR example.

The evidence-supported follow-up is generic typed-fragment composition: retrieve
multiple independently typed examples, expose their structural and argument
fragments to a selector, and bind only request-visible values. Do not add
per-command templates or tune against outer-validation outcomes.

Raw retrieval report SHA-256:
`5a80ac4d0eca5d554b239de3b75ef598b2fecacc5e992bd13979a0efd6dca330`.

Raw coverage-attribution report SHA-256:
`0abd81a70053bdd5b9529c591152374024de7944301adb56bed0d0016687fc74`.
