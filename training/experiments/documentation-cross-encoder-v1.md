# Documentation-conditioned cross-encoder v1

Status: preregistered before implementation, dataset generation, training, or
development/test scoring.

## Question

Can a small cross-encoder select a complete locally documented recipe for
commands absent from its supervised training set, while abstaining when the
candidate documentation is insufficient? This experiment does not generate
command words. It ranks bounded Rust-valid templates, binds only request-visible
literals, and delegates final safety and grounding to the production boundary.

## Frozen source and split

- Documentation index:
  `training/artifacts/documentation-template-index-v2/index.jsonl`
- Index SHA-256:
  `17b3b2e9feb215f858e893ba4f761088f9edfbd674ec89633c26d4332a6ab64b`
- Source: TLDR v2.3, 26,463 Rust-round-tripped templates for 4,219 commands.
- Eligible rows: Linux; command matches
  `[A-Za-z0-9][A-Za-z0-9+._-]*`; command has at least two templates.
- Split key: ascending SHA-256 of `documentation-cross-encoder-v1\0COMMAND`.
- First 128 commands: sealed test.
- Next 64 commands: open development/calibration.
- Next 512 commands: training.
- Every remaining command is unused. Commands are disjoint across all splits.

Within each command, templates sort by `record_id`. Training uses at most four
templates per command. Development and test use the first eligible bindable
template per command as the sufficient case.

## Frozen query and candidates

The sufficient query is the selected template instruction with generic
placeholders bound to deterministic `/tmp/shelliq-ranker-*` paths, integers,
URLs, or identifiers by the existing generic binder. A fixed command-agnostic
verb map makes the query non-identical to the documentation where applicable:
`display→show`, `print→show`, `list→show`, `remove→delete`, `create→make`,
`download→fetch`, `upload→send`, `search→find`, `execute→run`,
`convert→transform`, and `install→add`.

Each sufficient candidate pool contains every template for the requested
command, capped at 32 by record ID, plus the first template from seven
deterministic same-split distractor commands. The positive source template must
remain in the cap. Candidate text contains only its command, platform, summary
context, documentation instruction, and documented shell shape. The semantic
target is never model input.

Each insufficient case uses the same query but removes every candidate from the
requested command, leaving the seven distractor recipes. It must abstain. This
represents retrieval that found plausible documentation but no documentation for
the requested capability.

## Frozen model and training

- Base: `Salesforce/codet5-small` revision
  `b1ee9570c289f21b5922b9c768a1ce12957bf968`.
- Input: one CodeT5 sequence containing tagged query and candidate fields.
- Representation: attention-mask mean of final encoder states.
- Score: learned linear scalar head.
- The pretrained encoder is frozen; only the head is trained.
- Maximum sequence length: 256 tokens; any truncation is reported.
- Training pairs per selected template: its positive candidate, one distinct
  same-command hard negative, and two cross-command negatives.
- Loss: binary cross entropy with logits.
- Optimizer: AdamW, learning rate 0.001, no weight decay.
- Batch size: 64 cached embeddings; 50 epochs; seed 20260901.
- Model selection: highest development macro accuracy, then higher sufficient
  top-1, then earlier epoch.

The frozen encoder embeddings may be cached because the encoder never changes.
No dependency changes are authorized.

## Abstention calibration

For every development query, score its sufficient and insufficient pools.
Choose the smallest observed threshold that yields 100% insufficient
abstention; among ties retain the threshold with greatest sufficient ready rate.
A sufficient case is ready only when its top score meets the threshold, the
selected `record_id` is the positive source template, generic binding resolves
every slot, and the resulting semantic document passes the Rust action codec.

The threshold, head checkpoint, split manifest, dataset hashes, and development
report are frozen before the sealed test is scored once.

## Gates

- Split integrity: zero command overlap.
- Candidate integrity: every candidate is from the frozen documentation index;
  no semantic target enters model text; every ready output passes Rust codec.
- Development authorization: at least 70% sufficient ready, 100% insufficient
  abstention, and at least 45 distinct development commands ready.
- Sealed test: at least 70% sufficient ready (90/128), 100% insufficient
  abstention (128/128), and zero commands seen in supervised training.
- CPU warm p95 for scoring and compiling one complete pool: at most 5,000 ms.

Failure of development authorization stops before test. Failure of the sealed
gate rejects this ranker; no case-level tuning or replacement split is allowed.
Passing authorizes a thin runtime integration behind ShellIQ's existing
deterministic production boundary.
