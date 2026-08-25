# Semantic-action pointer v1

Status: source-copy oracle passed; neural plumbing pending.

This follow-up addresses the failure isolated by `semantic-action-decoder-v1`:
a randomly initialized 17.1M-parameter byte decoder memorized eight examples but
could not spell any first command on the command-disjoint inner split. It keeps
the frozen dataset, splits, Rust `SemanticActionV1` codec, grammar masking, and
evaluation gates unchanged.

## Architecture

- Official CodeT5-small encoder and decoder weights at revision
  `b1ee9570c289f21b5922b9c768a1ce12957bf968`.
- Four transferred CodeT5 decoder blocks with cross-attention, rather than four
  randomly initialized generic Transformer blocks.
- The same 320-action tied embedding/output vocabulary and maximum 192 actions.
- A pointer distribution over UTF-8 bytes in the source prompt. Source bytes
  receive aligned CodeT5 encoder states plus byte identity embeddings.
- A learned generator/copy mixture. Structural actions and absent words use the
  action generator; exact source spans receive pointer and gate supervision.
- Final output remains ordinary `SemanticActionV1`; Rust decoding, canonical
  semantic JSON, grammar validation, and render/re-lower remain authoritative.

This design is intentionally mixed. Copying can solve nearly every unseen
command name, but it cannot solve all arguments, redirects, quoting, or derived
operands.

## Frozen gates

1. Copy oracle: on the existing 92-record inner validation split, require at
   least 95% exact source-span coverage for command words and 50% for all words.
2. Plumbing: portable CPU forward/backward/masked generation and an eight-record
   CUDA overfit within 2,000 steps to 8/8 exact actions, Rust-valid documents,
   and accepted references.
3. Inner development: train at most 50 epochs on the existing 518 records and
   select solely by lowest teacher-forced action loss on the existing 92. Keep
   the v1 thresholds: 90% Rust-valid, 80% first command, 5% accepted reference.
4. Only after passing inner development, freeze the epoch count and retrain once
   on all 610 outer-train records. Evaluate the 78-record outer validation once
   with the existing 90%/80%/10% minimum and 95%/90%/25% great thresholds.
5. Only after passing outer validation, evaluate the existing 98-record test
   once and compare with the locked CodeT5 checkpoint beginning `7d21790`.

Raw checkpoints and generations remain ignored. No release, shadow, execution,
runtime export, or quantization work is authorized by this experiment.

## Copy-oracle evidence

The gate passed:

- Inner-validation command words: 91/93 copyable (97.85%)
- Inner-validation command bytes: 420/434 copyable (96.77%)
- All inner-validation words: 258/460 copyable (56.09%)
- All inner-validation bytes: 1,393/3,655 copyable (38.11%)

The lower byte coverage confirms that the byte generator remains essential;
the high command coverage makes supervised pointing a direct response to the
observed 0/92 first-command failure.
