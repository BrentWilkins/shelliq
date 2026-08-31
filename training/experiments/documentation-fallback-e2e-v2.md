# Documentation fallback end-to-end v2

Status: preregistered after v1 aggregate failure and before v2 implementation or
evaluation. No v1 case-level outcome is used.

## Frozen architecture change

V1 proved coverage, semantic validity, local verification, and latency, but
missed ready precision and one incomplete abstention. V2 changes selection
semantics generically:

1. Rank all retrieved documentation recipes before attempting compilation.
2. Lock compilation to the single highest-scoring documentation intent. A
   lower-scoring recipe may never become ready merely because the best recipe
   cannot be compiled or lacks locally indexed options.
3. A ready candidate must consume every typed request literal. Extra request
   paths, URLs, remote paths, integers, or quoted text cause `needs_input`
   instead of selecting a structurally different recipe.
4. Preserve all v1 constraints: command integrity, generic placeholder typing,
   local option alternatives, semantic round trip, local verifier, no
   command-family production rules, and no command output on abstention.

No threshold or rule may be tuned against v1's individual sealed outcomes.

## Independent data

The v1 builder's stable selection is continued only after excluding all 96 v1
development/test commands in addition to the neural-training and earlier
100-family compiler commands. The next 96 eligible families form:

- Open v2 development: 24 complete and 8 incomplete requests.
- Sealed v2 test: 48 complete and 16 incomplete requests.

All v2 commands are disjoint from v1, from each other across partitions, from
the 610-record neural training split, and from the prior compiler benchmark.
The manifest freezes exact hashes before implementation.

## Frozen gates

The complete production CLI path and fixture construction are unchanged from
v1. After open development, freeze and commit the implementation, then run the
sealed v2 test once. It passes only if:

1. At least 39/48 complete requests return the exact expected command.
2. At least 95% of commands emitted for complete requests are exact.
3. Every emitted command is semantic-round-trip valid and locally verified.
4. All 16 incomplete requests exit nonzero, contain `needs_input`, and write no
   command to standard output.
5. Mean latency is at most 250 ms and maximum latency at most one second after
   one warm-up.

Failure is recorded without tuning or rerunning the sealed partition.
