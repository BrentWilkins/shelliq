# Semantic-action decoder v1

Status: stopped after the inner-development gate failed.

This experiment pairs the official CodeT5-small encoder at revision
`b1ee9570c289f21b5922b9c768a1ce12957bf968` with a compact, project-owned
four-layer action decoder. It is evidence-first: each gate is committed before
the next one starts, and a failed gate stops the experiment.

## Frozen representation

`SemanticActionV1` is owned by the Rust syntax crate. It represents every
reachable `SemanticDocumentV2` structure with fixed structural actions and
role-specific UTF-8 byte spans. Rust owns encoding, decoding, canonical
semantic JSON, grammar transitions, and final render/re-lower validation.
Python must consume the manifest emitted by `semantic-actions manifest`; it
must not maintain a second handwritten grammar.

- Action schema: 1
- Vocabulary: 320 actions (`0..63` reserved, `64..319` raw bytes)
- Maximum target length: 192 actions, without truncation
- Source maximum: 256 CodeT5 tokens, without truncation
- No pointer-generator in v1

## Gates

1. Codec: all 786 curated semantic targets must round-trip and pass Rust final
   validation; vocabulary at most 320; maximum target at most 192.
2. Plumbing: CPU forward/backward/masked generation, then CUDA overfit of eight
   deterministic real records within 2,000 steps to 8/8 exact actions,
   Rust-valid outputs, and accepted references.
3. Inner development: command-disjoint 15% split of the existing 610 outer
   training records with seed `20260826`; select the epoch with lowest inner
   teacher-forced action loss over at most 50 epochs. Require 90% Rust-valid,
   80% first-command accuracy, and 5% reference acceptance.
4. Outer validation: freeze the selected epoch count and recipe, retrain once
   on all 610 records, and evaluate the 78 outer-validation records once.
   Require 90% Rust-valid, 80% first-command accuracy, and 10% reference
   acceptance. Great means 95%, 90%, and 25%, respectively.
5. Comparison: only after passing outer validation, evaluate the existing 98
   test records once against the locked CodeT5 checkpoint whose SHA-256 begins
   `7d21790`. This remains development evidence; deployment needs a fresh
   shadow evaluation.

The encoder and decoder are fine-tuned together with AdamW, zero weight decay,
gradient clipping at 1.0, encoder learning rate `5e-5`, decoder learning rate
`5e-4`, BF16 CUDA, and batch size 8. Generated weights and raw generations stay
under ignored `training/artifacts/` paths.

## Codec gate evidence

The frozen `curated-semantic-v7.jsonl` corpus contains 786 records.

- Encode success: 786/786
- Exact semantic JSON after action decode: 786/786
- Rust-valid decode and render/re-lower: 786/786
- Exact original shell spelling after normalized render: 765/786 (the 21
  differences are existing canonical declaration ordering, not semantic loss)
- Action length: median 46, p95 89, maximum 147; 0 over 192
- Vocabulary size: 320

Rust tests cover pipelines, multiple commands, declarations, all grammar
redirect actions, non-ASCII UTF-8, malformed transitions, out-of-vocabulary
tokens, incomplete sequences, invalid and incomplete UTF-8, non-canonical
field order, and deterministic malformed-token fuzzing. `tree-sitter-zsh`
0.63.4 does not parse the otherwise modeled `<>` redirect, so that action is
covered as a grammar transition but cannot pass the pre-existing final
render/re-lower boundary until the upstream grammar supports it.

## Model-plumbing gate evidence

The frozen manifest uses the existing outer split unchanged and creates an
exact command-disjoint inner split of 518 training and 92 validation records
with seed `20260826`. No source or action sequence is truncated: the observed
maxima are 181 CodeT5 tokens and 147 actions.

The instantiated model has 35,316,480 pretrained encoder parameters and
17,079,296 project-owned decoder parameters, for 52,395,776 total. A real CPU
forward/backward pass produced finite loss and gradients, while the focused
portable test exercised grammar-masked greedy generation. The Python manifest
interpreter also accepted a full real Rust target through its complete state.

On the local RTX 4090, the deterministic eight-record gate reached all stopping
criteria at step 100 of the 2,000-step cap:

- Exact actions: 8/8
- Rust-valid decode and render/re-lower: 8/8
- First command: 8/8
- Reference acceptance: 8/8
- Loss: 12.146155 initial to 0.000988 final

The raw checkpoint, generations, and detailed reports remain ignored under
`training/artifacts/semantic-action-decoder-v1/`.

## Inner-development gate outcome

The full 50-epoch run selected epoch 3 solely by the lowest teacher-forced
inner-validation action loss, as frozen. Its loss was 2.416366; later epochs
continued lowering training loss while validation loss rose to 3.660949 at
epoch 50.

The selected checkpoint failed every semantic threshold on the 92 unseen-command
inner-validation records:

- Rust-valid decode and render/re-lower: 49/92 (53.3%; required 90%)
- First-command match: 0/92 (required 80%)
- Reference acceptance: 0/92 (required 5%)
- Exact actions: 0/92
- Incomplete at the 192-action ceiling: 43/92

Valid outputs were still semantically wrong and showed byte-level collapse to
seen commands and repeated punctuation or option fragments. This is evidence
that the first hybrid recipe can memorize eight examples but does not transfer
command identity across the deliberately command-disjoint split. Per the
frozen protocol, outer validation and the 98-record comparison test were not
evaluated, and no release, runtime export, or quantization work is justified.
