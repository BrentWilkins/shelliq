# Preference pilot v1

This pilot compared chosen-answer SFT with offline DPO from the same rank-8
step-10,000 checkpoint. Both conditions used the same 32 manually reviewed
preference pairs, command-family-disjoint 25/7 train/validation split, 128-row
curated supervised replay stream, three epochs (75 updates), learning rate
`1e-5`, and rank 8 / alpha 4. DPO used beta `0.1` and cached frozen-reference
log probabilities, so it did not keep a second model in VRAM.

The 32 accepted pairs came from 64 valid rank-8 near misses. Manual review
excluded 24 task-equivalent answers, 6 pairs whose apparent preference depended
on operands absent from the prompt, and 2 ambiguous requirements. Accepted
pairs are balanced across precise (9), multi-constraint (7), pipeline (8), and
compositional (8) tasks.

| checkpoint | preference validation margin delta | release flags | release grounded | retention flags | retention grounded |
| --- | ---: | ---: | ---: | ---: | ---: |
| rank-8 baseline | — | 12/35 | 10/35 | 14/20 | 10/20 |
| chosen SFT | -0.331 | 1/35 | 0/35 | 1/20 | 0/20 |
| DPO | -0.215 | 12/35 | 10/35 | 14/20 | 10/20 |

DPO fit the training pairs (96% improved reference-relative margins) but did
not generalize to held-out command families (43%, below chance). Its release
metrics were unchanged except one new wrong-command regression on
`readlink-canonical`; retention was exactly preserved. The chosen-only SFT
control catastrophically overfit this tiny corpus and is rejected.

This is a plumbing success and a model-quality negative result. Do not sweep
beta, learning rate, or epochs on these same 32 pairs: the seven-pair validation
set is too small and neither objective improved it. Expand the reviewed corpus
with diverse rank-8 failures, preserve command-family-disjoint validation, and
repeat the same matched control before any shadow evaluation or promotion.

No generated shell command was executed. The Rust verifier only decoded,
rendered, reparsed, and relowered SemanticDocumentV2. The frozen shadow suite
was not consulted.
