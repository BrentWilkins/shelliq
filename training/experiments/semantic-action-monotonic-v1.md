# Semantic-action monotonic copy v1

Status: stopped after the selected-checkpoint decoding diagnostic failed.

This sequential inner-development experiment follows the stopped
`semantic-action-pointer-v1`. It changes only grammar-masked decoding: while a
word is in pointer mode, the next copied byte must come from the next contiguous
source position. The model can end the word or leave pointer mode when its copy
gate falls below 0.5. Generator-only words remain unchanged.

The change targets the observed v2 failure directly. V2 matched 41/92 first
commands, but corrupted 49 other copyable commands by jumping among independent
source-byte positions. It retains the same pretrained four-block CodeT5 model,
training losses, optimizer, Rust actions, copy oracle, 518/92 split, epoch
selection rule, and semantic gates. Because the inner split has now informed an
architecture decision, this result is sequential development evidence; the
78-record outer validation remains the protected gate.

## Gates

1. CPU masked-generation tests and the deterministic eight-record CUDA overfit
   must pass with monotonic decoding.
2. Retrain for 50 epochs and select solely by inner teacher-forced action loss.
   Require 90% Rust-valid, 80% first command, and 5% accepted reference.
3. Only if the inner gate passes, freeze the epoch count, retrain once on all
   610 outer-train records, and evaluate the protected 78-record outer
   validation once.

Outer test, release, shadow, execution, export, and quantization remain closed
unless their prior gates pass.

## Outcome

Applying monotonic decoding to the already selected v2 epoch-16 checkpoint
improved the same 92 inner-development examples:

- Rust-valid: 79/92, up from 75/92
- First-command match: 53/92, up from 41/92
- Reference acceptance: 0/92, unchanged
- Exact actions: 0/92, unchanged

This remains below the frozen 90% valid, 80% first-command, and 5% acceptance
thresholds. A full retrain was not run because monotonicity changes decoding
only: training losses, optimizer updates, deterministic seed, selection loss,
and selected checkpoint are byte-for-byte the v2 recipe. Repeating those 50
epochs would not test a different hypothesis.

The improvement confirms that non-contiguous jumps were one failure source,
but constraining one byte at a time is insufficient. The next plausible design
must select and emit an exact source span atomically, with byte generation only
as a fallback for words absent from the prompt.
