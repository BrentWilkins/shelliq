# Semantic-action documentation fragments v1

Status: preregistered for open inner development; outer validation and test are
sealed from this experiment.

## Hypothesis

The single-template failure is caused by absent complete recipes, not absent
documentation. A bounded, command-agnostic composer can recover useful recipes
by merging ordered argument sequences from up to three of the top eight
same-command, same-platform TLDR templates. Generic request-literal binding and
the Rust semantic-action codec remain unchanged.

## Frozen procedure

1. Use the unchanged TLDR v2.3 typed index from documentation-templates v1.
2. Retrieve eight templates with the unchanged command/platform-scoped TF-IDF
   ranker. Do not use curated targets or record identifiers.
3. Retain every retrieved document. For compatible single-command documents,
   form the shortest common supersequence of their argument sequences using up
   to three distinct source templates. Do not merge pipelines, redirections,
   declarations, different commands, or non-plain argument nodes.
4. Deduplicate typed documents, rank by mean source retrieval score with stable
   tie breaks, and retain at most 64 candidates per request.
5. Bind only generic placeholders to compatible literals explicitly extracted
   from the instruction. No per-command rules, weights, or corrections are
   allowed.

## Feasibility gate

On the already-open 92-record inner split, encode and decode every composed
candidate with the Rust action codec. Require all candidates to be valid and at
least 19 records across ten command families to have a reference-matching typed
skeleton somewhere in the bounded set. Report exact/reference oracle coverage
separately; no oracle result participates in candidate construction or ranking.

Passing authorizes a target-blind selector experiment. Failure stops this
composition rule and requires a different representation. No generated command
is executed.
