# Semantic-action lexical candidate v1

Status: neural plumbing and candidate-oracle gates passed; training gates in progress.

This experiment follows the stopped `semantic-action-span-v1`. The earlier model
could copy a correctly selected source span atomically, but its independently
learned start and end heads did not select reliable spans for unseen commands.
This version removes learned boundary prediction.

## Architecture

- The same official CodeT5-small encoder and first four transferred decoder
  blocks at revision `b1ee9570c289f21b5922b9c768a1ce12957bf968`.
- The same 320-action Rust-owned `SemanticActionV1` grammar and maximum 192
  actions. Rust remains authoritative for decoding, validation, rendering, and
  render/re-lower evaluation.
- A deterministic lexer extracts complete UTF-8 source spans: trimmed
  whitespace-delimited terms, shell-shaped atoms, and quoted contents, bounded
  to 128 bytes and deduplicated by source span.
- At the first byte position of a semantic word, the decoder scores pooled,
  contextual representations of those fixed candidates. A learned gate at its
  natural 0.5 threshold either emits the chosen candidate atomically or falls
  back to ordinary grammar-masked byte generation.
- Candidate boundaries are never learned or independently combined.

The model has 35,316,480 encoder parameters and 17,998,081 transferred
decoder/candidate parameters, 53,314,561 total. A prompt has at most 110
candidates in the frozen dataset.

For a first-principles explanation of the project, prior evidence, and this
design, see `semantic-action-candidate-v1-explainer.md`.

## Frozen gates

1. Candidate oracle on the existing 92-record inner-development split: require
   at least 95% command-word coverage and 50% all-word coverage.
2. CPU plumbing: finite forward loss, backward gradients, and generation that
   never leaves the Rust manifest grammar.
3. Eight distinct training commands: within 2,000 CUDA steps require 8/8 exact
   actions, Rust-valid render/re-lower, first-command match, and reference
   acceptance.
4. Inner development: train up to 50 epochs on the frozen 518 records. Select
   solely by lowest teacher-forced action loss on the frozen 92 records. Require
   90% Rust-valid, 80% first-command match, and 5% reference acceptance.
5. Only if inner development passes, freeze the selected epoch count, retrain
   once on all 610 outer-training records, and open the protected 78-record
   outer validation with 90%/80%/10% minimum gates.
6. Only if outer validation passes, open the protected 98-record comparison
   test against the locked CodeT5 checkpoint beginning `7d21790`.

No release, shadow, execution, quantization, or runtime-export work is
authorized by this experiment. Raw checkpoints, histories, and generations
remain ignored under `training/artifacts/semantic-action-candidate-v1/`.

## Pre-training evidence

The deterministic candidate oracle passed:

- Command words: 91/93 (97.85%).
- All words: 240/460 (52.17%).
- Arguments: 149/365 (40.82%).
- Redirect targets: 0/2.

This preserves the unrestricted source-substring oracle's command coverage
while deliberately excluding misleading word interiors. Candidate spans are
therefore adequate for the experiment's primary unseen-command hypothesis;
generator fallback remains necessary for derived operands and redirects.

The real CodeT5 CPU gate also passed with finite action, candidate-selection,
gate, and total losses and finite gradients. Bounded greedy generation stayed
inside every transition allowed by the Rust manifest.
