# Semantic-action grounded count v1

Status: preregistered before implementation evaluation.

V3's gold-count counterfactual reaches 4/92 accepted, while all 23 fully candidate-covered records decode exactly with gold words and counts. Per-decision inspection identifies two reusable defects. First, v1-v3 predict argument count at the command-name byte start, before the chosen command has entered decoder state. Second, global training-lexicon candidates compete with prompt-local and mechanically derived arguments even when authoritative prompt evidence exists. Several fully covered inner records have every gold candidate at rank 1-4 but a count rank outside the decoder's top four.

This experiment is a grounded retrieval/count hybrid, not a fourth candidate-embedding variant.

## Frozen changes

1. Retain the v3 pretrained encoder, candidate representations, role scoring, inventory source, optimizer, split, 50-epoch validation selection, Rust grammar, and project-owned action decoder.
2. Move each command's argument-count label from the first command byte to the decoder position immediately after that command's `WORD_END`. During generation, choose the count on first entry to `command_body`, after the complete selected command name is in decoder state.
3. Add deterministic spelled-number normalization for zero through twenty, plus `single`, `once`, and the ordinal forms first through twentieth. A matching source word makes its decimal candidate derived and role-eligible. No target text participates in normalization.
4. At argument-byte decisions, if any role-eligible local or derived candidates exist, mask out global-only candidates. Command-name decisions retain the full role mask. During training, apply the grounded mask only when the gold candidate is local or derived; global-only targets retain the full role mask.
5. Expand count choices from top four to top eight and use beam width 32. Candidate branching remains eight for arguments and four for commands. Scores, loss weights, action bound, and all other generation behavior remain unchanged.

## Gates

- CPU: deterministic number derivation and post-command count alignment are covered by tests; all losses and gradients are finite; bounded generation completes.
- Eight-record overfit: 8/8 first-command within 2,000 CUDA updates.
- Inner: same 518 training records and 92 open inner-development records; select only by combined validation loss.
- Floors: at least 90% Rust-valid and 80% first-command match.
- Success: at least 5/92 reference accepted.

Outer validation and test remain sealed. If this misses five accepted references, use its per-decision attribution to decide between a true documentation retriever and a deterministic compiler; do not tune constants or rules against individual reference outcomes.
