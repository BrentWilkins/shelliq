# Semantic-action documentation fragments v1 results

Status: failed the composition-feasibility gate; stop whole-example
supersequence composition.

The generic composer generated 4,810 bounded candidates for the 92 open-inner
records by merging up to three of the top eight typed TLDR examples. Every
candidate passed the Rust action encoder and decoder. Skeleton oracle coverage
improved only from 5/92 for single templates to 6/92 across the same five command
families, far below the required 19 records and ten families. No candidate
matched a complete reference exactly.

Read-only observability attribution found 439 reference words after command
names. Of these, 213 appear in the request, 36 appear exactly in same-command
documentation, 96 can only be represented by a documented placeholder, and 94
are neither present nor covered by a documented placeholder. Only 16/92 records
have every exact reference word observable; 50/92 are observable when typed
placeholders are allowed.

Two requirements follow. First, the compiler must treat the authoritative prompt
context itself as fragment documentation instead of relying only on complete
TLDR examples. Second, unresolved literal slots must cause an explicit abstention
or remain typed placeholders; a target-blind system must not invent benchmark
fixture values. Strict lexical reference acceptance should remain reported, but
cannot be the sole correctness measure for template outputs.

Raw composition report SHA-256:
`d7826e7b2d62880dabfb46abb953fdfd210514febfde284e3b9c1fc689f8751d`.

Raw observability report SHA-256:
`2eec5aa039655053b253460cdd28d5bee9ea9f57b0d3e929a7bf92927f7d299f`.
