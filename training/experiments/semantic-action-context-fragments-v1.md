# Semantic-action authoritative-context fragments v1

Status: preregistered for open inner development; outer validation and test are
sealed from this experiment.

## Hypothesis

The prompt's authoritative context is itself retrieved documentation and often
names options absent from any one TLDR example. A generic compiler can extract
those option spellings, enumerate bounded subsets, and overlay them on the typed
structure and operand slots of independently authored TLDR templates.

## Frozen procedure

1. Retain the unchanged TLDR-only index and top-eight command/platform TF-IDF
   retrieval.
2. Extract distinct option spellings from context with one generic lexical
   pattern. Keep at most the first six; do not interpret option semantics or add
   per-command arity tables.
3. Enumerate nonempty option subsets in context order. Overlay each subset on
   every retrieved typed template using the ordered supersequence composer while
   preserving pipelines, redirections, and other structure from the base.
4. Bind only request-visible literals with the unchanged generic binder.
   Deduplicate and retain at most 512 candidates per request.
5. Candidate construction and ranking cannot use reference documents, record
   identifiers, or evaluation outcomes.

## Gate

On the open 92-record inner split, require every candidate to pass the Rust
action round trip and require reference-skeleton oracle coverage of at least 19
records across ten command families. Exact oracle coverage and unresolved slot
counts are diagnostic only. Passing authorizes selector work; failure stops this
lexical context composition rule.

The compiler must eventually abstain when required literal slots remain
unresolved. This feasibility experiment does not execute generated commands and
does not access outer validation or test.
