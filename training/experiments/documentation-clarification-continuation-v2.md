# Documentation clarification continuation v2

Status: preregistered before v2 dataset generation, harness changes, or
evaluation. Runtime `2a50d93` is frozen and receives no v2 changes.

V1 retained perfect safety and ready precision but failed completion coverage.
Review showed the fake local index contained options only from the dataset source
row even when runtime selected another recipe, and the alignment oracle rejected
duplicate common/platform intent rows. V2 corrects only those fixture contracts.

For each command, the fake executable advertises the union of literal option
spellings present in every frozen documentation-index recipe for that command.
This represents the same installed command family available to runtime
retrieval; it does not add options absent from documentation. Source alignment
accepts one or more command/normalized-intent matches when the reported operand
exists in at least one matching recipe. All continuation, binding, lowering,
verification, and response code remains byte-for-byte frozen.

The builder selects the next 64 eligible distinct Linux command families after
all exclusions through continuation v1. The first 16 are open development and
the remaining 48 are sealed. Dataset and documentation-index hashes are
committed before either partition runs.

The v1 gates are unchanged: every intermediate response and invalid source must
fail closed; at least 95% must reach ready within eight distinct questions;
ready precision and local/semantic validity must be 100%; frozen source-aligned
v1 and fallback v3/v4 precision/safety invariants must hold; mean flow latency is
at most 500 ms and maximum at most two seconds.

The sealed partition is opened once after harness, hashes, and open behavior are
frozen. Failure rejects v2 and does not authorize tuning or case replacement.
