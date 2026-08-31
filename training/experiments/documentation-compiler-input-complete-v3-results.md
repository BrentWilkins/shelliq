# Documentation compiler input-complete v3 results

Status: passed the preregistered capability gate.

The generic compiler delivered 90 exact Rust-typed commands across 100 command
families absent from neural training. It returned 93 ready documents and withheld
seven; all 93 ready outputs passed Rust validation. Exact coverage was 90% and
ready precision was 96.77%, exceeding the frozen 80% and 95% floors.

The three false-ready outputs chose a simpler but semantically wrong recipe over
the directly matching recipe because unsupported-operand count preceded source
intent similarity. A final safety correction may rank semantic intent first,
then abstain if that recipe contains unresolved operands. This should exchange
coverage margin for precision without command-specific logic.

Benchmark SHA-256:
`d023fb742fbe7878a124d50c861a09296bda79dd7ee47cf171d45e1a1311d1db`.

Raw report SHA-256:
`8b190acd984d701b7a7f61677cd8335428eeda3ee31a199902ed8478f617686a`.
