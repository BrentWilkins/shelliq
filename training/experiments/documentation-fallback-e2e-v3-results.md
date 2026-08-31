# Documentation fallback end-to-end v3 results

Status: failed status-label gate; all command-quality and actual abstention-safety
gates passed. Do not rerun this partition.

- Complete exact and emitted: 44/48 (floor 39).
- Ready precision: 44/44 (100%; floor 95%).
- Emitted semantic/local-verifier validity: 44/44.
- Incomplete requests with nonzero exit and empty stdout: 16/16.
- Incomplete status labels: 15 `needs_input`, 1 `no_documentation`; the frozen
  gate required 16 `needs_input` and therefore failed.
- Mean latency after one warm-up: 41.05 ms; maximum 47.17 ms.
- Dataset SHA-256:
  `308fe822c5b9fbb9491c0cb67cd223a0fd6dd74121d9841cd44e2e7836cece3f`.
- Raw report SHA-256:
  `3ebabb3a1be2e779fb334632559312b79616d801d2bf656438efcac04f593d84`.
- Binary SHA-256:
  `8977ce96dd44810768ffe1941970fc93631065f8654e6f232d9494bd626e62ff`.

The fallback was fail-closed for every incomplete request; the miss was solely
the distinction between its two explicit abstention states. Because v3 froze a
more specific label requirement, it remains a failed experiment. A new
independent evaluation may align its safety gate with the product invariant:
both `needs_input` and `no_documentation` must exit nonzero with empty stdout.
