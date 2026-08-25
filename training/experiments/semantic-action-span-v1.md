# Semantic-action atomic span v1

Status: neural plumbing passed; inner development pending.

This experiment replaces independent or monotonic byte copying with an atomic
source-span decision. At the first byte of a copyable semantic word, the model
predicts start and inclusive end byte positions. Decoding expands the selected
UTF-8 span into ordinary `SemanticActionV1` byte actions plus `WORD_END` in one
step. Low copy-gate confidence falls back to the pretrained byte generator.

The Rust action format, manifest grammar, canonical semantic JSON, final
render/re-lower validation, official CodeT5 revision, four transferred decoder
blocks, 518/92 sequential inner-development split, and protected outer splits
remain unchanged.

## Gates

1. CPU forward/backward and grammar-safe atomic expansion, then 8/8 exact,
   Rust-valid, accepted CUDA overfit within 2,000 steps.
2. Train up to 50 epochs, selecting solely by lowest inner teacher-forced action
   loss. Require 90% Rust-valid, 80% first command, and 5% accepted reference.
3. Only after passing, freeze the selected epoch count and retrain once on all
   610 outer-train records before opening the 78-record outer validation.
4. Open the 98-record comparison test only after passing outer validation.

The inner split has informed the span architecture through prior sequential
experiments. It remains useful development evidence, but the untouched outer
validation is required before any claim of generalization.

## Neural-plumbing evidence

The added span-end query brings the model to 53,576,705 parameters. CPU
forward/backward produced finite action, start-pointer, span-end, gate, and
combined losses, and bounded UTF-8 spans replayed through the manifest grammar.

The initial overfit diagnostic found two decoder defects without using inner
development data. A 32-by-32 start/end candidate cross-product combined
boundaries from different occurrences; atomic decoding now accepts only the
model's top start and top end as one pair and falls back when that pair is
invalid. Generator-only words on the overfit sample had copy confidence 0.906
and 0.932 while supervised copy spans were at least 0.999, so the frozen atomic
activation threshold is 0.95.

From a fresh initialization with those corrections, the strict CUDA gate passed
at step 200 of the 2,000-step cap:

- Exact actions: 8/8
- Rust-valid decode and render/re-lower: 8/8
- First command: 8/8
- Reference acceptance: 8/8
- Combined loss: 16.846510 initial to 0.524267 final

Raw checkpoints, generations, and diagnostics remain ignored under
`training/artifacts/semantic-action-span-v1/`.
