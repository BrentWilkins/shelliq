# Documentation compiler input-complete v2 results

Status: passed coverage and missed precision by one false-ready case.

Across 100 distinct command families unseen in neural training, v2 delivered 88
exact typed commands, withheld seven, and returned five incorrect ready
documents. All 93 ready outputs passed the Rust action round trip. Exact coverage
was 88%, above the required 80%; ready precision was 94.62%, just below 95%.

All five false-ready outputs selected a different example for the correct command
even though the benchmark instruction begins with the source documentation's
exact instruction before adding explicit values. V3 may treat exact normalized
documentation-instruction prefix containment as stronger evidence than bag-of-
words similarity. This generic evidence applies to any documentation source and
does not use source IDs or command-specific logic.

Benchmark SHA-256:
`d023fb742fbe7878a124d50c861a09296bda79dd7ee47cf171d45e1a1311d1db`.

Raw report SHA-256:
`6c27e649017fc110aa64cd2c9862822f3ee03f75080841587e6f6e202899d5c3`.
