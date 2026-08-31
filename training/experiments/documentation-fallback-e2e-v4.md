# Documentation fallback end-to-end v4

Status: preregistered before v4 dataset generation, harness change, or evaluation.
Runtime commit `8e227df` is frozen and receives no v4 changes.

V3 passed exact coverage, perfect ready precision, validity, latency, and the
actual shell safety invariant for all incomplete requests. Its experiment failed
because the gate required the narrower `needs_input` label even when
`no_documentation` was the truthful abstention. V4 corrects only that measurement
contract on a fresh command-disjoint set.

Selection continues the literal-free stable recipe sequence after excluding all
288 v1-v3 commands, the neural training commands, and the earlier compiler
benchmark. The next 96 commands form the same 24+8 open development and 48+16
sealed test partitions. Exact hashes are committed before either partition runs.

The real model-feature CLI, unavailable model endpoint, fresh executable/index,
vendored family retrieval, typed compiler, semantic lowerer, local verifier, and
latency measurement remain unchanged. Passing requires:

1. At least 39/48 complete requests emitted exactly.
2. At least 95% ready precision.
3. Every emitted command semantic-round-trip and locally verifier valid.
4. All 16 incomplete requests return nonzero, write empty stdout, and explicitly
   report either `needs_input` or `no_documentation`.
5. Mean latency at most 250 ms and maximum at most one second after one warm-up.

The sealed v4 partition is opened once after open development and an immutable
harness/runtime revision. No v1-v3 individual sealed outcome informs runtime or
v4 case selection.
