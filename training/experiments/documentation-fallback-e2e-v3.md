# Documentation fallback end-to-end v3

Status: preregistered before v3 dataset generation or evaluation. Runtime commit
`ed3caf6` is frozen; v3 changes only benchmark hygiene.

V2's sealed partition was never opened. Its open partition revealed one expected
target contaminated by a path-like fragment already present in source prose.
V3 preserves the v2 runtime and all v2 gates, continues stable selection after
excluding every v1 and v2 command, and additionally requires the generic request-
literal extractor to find no typed literal in the source instruction before
synthetic values are appended.

The next 96 eligible, distinct Linux command families form 24 complete plus 8
incomplete open-development cases and 48 complete plus 16 incomplete sealed-test
cases. They are disjoint from:

- the 610-record neural training partition;
- the earlier 100-family compiler benchmark;
- all 96 v1 runtime-evaluation commands;
- all 96 v2 runtime-evaluation commands; and
- each other across v3 development and test.

The frozen test is run once only after the literal-free invariant, hashes,
development behavior, implementation revision, and complete CLI harness have
been verified. Passing requires at least 39/48 exact complete deliveries, at
least 95% ready precision, semantic/local validity for every emitted command,
16/16 nonzero empty-stdout `needs_input` abstentions, mean latency at most 250 ms,
and maximum latency at most one second after one warm-up.
