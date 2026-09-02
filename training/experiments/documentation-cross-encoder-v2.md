# Documentation-conditioned cross-encoder v2

Status: preregistered before fresh test construction or scoring.

## Rationale

V1 selected the exact source recipe for 123/128 unseen commands and compiled
109/128 safely at 164.89 ms CPU p95, but its development-maximum threshold
allowed one unrelated-documentation pool to cross ready. V2 changes only the
abstention boundary. It does not retrain, inspect new cases, change candidates,
or reuse any v1 test command.

## Frozen candidate

- V1 checkpoint SHA-256:
  `138b6c79ed36458308fff038672e195a6ccc54c3fd4f0c584200b3e533f15cd4`
- Frozen CodeT5 encoder, scalar head, pair representation, generic binder, Rust
  codec validation, candidate cap, and seven-distractor construction are
  unchanged.
- V2 threshold: logit `0.0`, the binary classifier's natural decision boundary.
  This is stricter than v1's `-0.31467324495315546` and is frozen before fresh
  cases are constructed.

## Fresh split

Recompute the v1 eligible command order from the same documentation index and
split key. V2 test commands are positions 704 through 831 inclusive: the next
128 commands after all 512 training, 64 development, and 128 v1 test commands.
Thus every v2 command is absent from supervised training and every opened v1
partition.

For each command, select the first bindable template by record ID. Generate the
same command-agnostic paraphrase and deterministic request-visible slot values as
v1. Each sufficient pool contains the requested command's templates plus seven
fresh-split distractors; each insufficient pool removes every requested-command
template.

The v2 dataset SHA-256 and derived checkpoint SHA-256 are recorded before one
scoring pass. No v2 development set or threshold calibration exists.

## Gates

- Split integrity: 128 distinct commands, zero overlap with all v1 splits.
- Sufficient unseen-command ready: at least 90/128.
- Insufficient-documentation abstention: 128/128.
- Rust validation: every ready document passes encode/decode; failures abstain.
- CPU warm p95 for one sufficient pool: at most 5,000 ms.

Any miss rejects this architecture. Passing authorizes runtime integration
behind the existing local index, deterministic safety policy, and operand
grounding boundary.
