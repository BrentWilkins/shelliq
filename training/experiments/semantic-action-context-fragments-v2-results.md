# Semantic-action authoritative-context fragments v2 results

Status: failed the feasibility gate; stop bounded context-subset expansion.

The generic subcommand-prefix and version-suffix corrections are covered by
unit tests, but the open-inner aggregate was unchanged: 11/92 skeleton-covered
records across eight command families and three exact candidates. All 8,063
candidates passed the Rust action round trip.

The unchanged result confirms that expanding target-oracle candidate sets is no
longer the useful objective. The production compiler now needs a target-blind
selection contract: rank documentation and context fragments, bind only values
observable in the instruction, return a typed command only when every required
slot is resolved, and otherwise return an explicit abstention describing the
missing slots. Evaluation must separate correctness on input-complete requests
from safe abstention on under-specified requests.

Raw report SHA-256:
`692b38bfa35503682d736fda053e1614cdc0f10360952e34f9379df9c05c22da`.
