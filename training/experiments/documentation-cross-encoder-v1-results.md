# Documentation-conditioned cross-encoder v1 results

Status: failed sealed abstention gate; do not integrate v1.

The frozen ranker generalized to commands absent from supervised training and
comfortably passed its sufficient-selection and latency floors, but one of 128
insufficient-documentation pools crossed the calibrated threshold.

| Sealed gate                           |  Required |          Result |
| ------------------------------------- | --------: | --------------: |
| Sufficient unseen-command ready       |   ≥90/128 | 109/128 (85.2%) |
| Insufficient-documentation abstention |   128/128 | 127/128 (99.2%) |
| CPU warm p95                          | ≤5,000 ms |       164.89 ms |
| Pair truncation                       |  reported |               0 |

The exact source documentation record ranked first on 123/128 sufficient cases.
Ready was lower because the frozen threshold rejected some low-scoring correct
records and generic binding made some selected documents fail Rust syntax
validation. Those failures correctly remained abstentions.

The insufficient failure was the `setup-alpine` query against unrelated
documentation. `pkgctl-auth:1` scored `-0.182469442486763`, above the calibrated
threshold `-0.31467324495315546`. No command was emitted or executed.

## Decision

Reject v1 under its preregistered all-or-nothing abstention rule. Do not tune the
threshold against this opened case. The evidence supports one fresh follow-up:
retain the unchanged encoder/head and use the classifier's preregistered natural
decision boundary, logit zero, on commands unused by all v1 splits. This is more
conservative than the development-maximum threshold and can be tested without
training on or reusing any v1 test command.

- Checkpoint SHA-256:
  `138b6c79ed36458308fff038672e195a6ccc54c3fd4f0c584200b3e533f15cd4`
- Sealed test SHA-256:
  `d0d9e6c8ce97bcdbbd4834e44ee43f56294cc77d181152446f3dc8753e42ba2f`
- Raw test report SHA-256:
  `79d70e441c96ffca6193e0f7c173dafc779aa1233103235efa33e642f9f36bc7`
