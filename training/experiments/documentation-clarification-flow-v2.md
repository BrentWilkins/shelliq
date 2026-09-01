# Documentation clarification flow v2

Status: preregistered before v2 dataset generation, harness changes, or
evaluation. Runtime `ef76b30` is frozen and receives no v2 changes.

V1 preserved every safety and precision invariant but failed its slot-kind gate
because the independent Python oracle uses broad substring matching while the
runtime uses token boundaries. That made labels such as `path/to/file.hurl`
disagree on kind despite agreeing on the exact unresolved placeholder. V2
corrects only this measurement contract. Kind remains descriptive metadata;
missing-slot identification is the exact documentation-placeholder label.

The builder selects the next 64 eligible, distinct Linux command families in
stable record-id order after all v1 exclusions plus every v1 clarification
command. The first 16 are open development and the remaining 48 are sealed.
Hashes are committed before either partition runs. The v2 sealed partition is
opened once after the unchanged runtime, builder, harness, hashes, and open
behavior are frozen.

Passing requires on the 48 sealed cases:

1. 48/48 incomplete default requests return nonzero with empty stdout.
2. 48/48 structured abstentions contain no command or semantic document and
   contain exactly one non-empty question.
3. At least 95% identify the exact frozen placeholder label.
4. At least 80% of answered requests emit the exact expected command.
5. Answered ready precision is 100%, and every emitted command passes semantic
   round-trip and local command/flag verification.
6. Frozen v3 plus v4 ready precision remains 100%, and all their incomplete
   requests remain commandless abstentions.
7. Mean latency after one warm-up is at most 250 ms and maximum latency is at
   most one second.

Failure of any gate rejects v2. Sealed outcomes do not authorize runtime tuning,
command-specific rules, or case replacement.
