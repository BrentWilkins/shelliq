# Documentation fallback end-to-end v1 results

Status: failed frozen sealed gate; do not tune or rerun this partition.

Commit `6aa2c59` was evaluated once on the independently preregistered 64-record
test partition. The model endpoint was unavailable for every case, so every
delivery or abstention traversed the production documentation fallback.

- Complete exact: 40/48 (frozen floor 39).
- Complete emitted: 43/48.
- Ready precision: 40/43, 93.02% (frozen floor 95%; failed).
- Emitted semantic/local-verifier validity: 43/43.
- Safe incomplete abstentions: 15/16 (required 16; failed).
- Mean latency after one warm-up: 40.50 ms (ceiling 250 ms).
- Maximum latency: 53.17 ms (ceiling 1 second).
- Frozen test dataset SHA-256:
  `6872761f909b2ced5bf9aee6bd59378581b98adc09c1cd90ca2de5b572d7d09e`.
- Raw test report SHA-256:
  `dc6d3308b702347d1e0ce6d579eb8d15f387a4999c4ca6d494725c4a72f795eb`.
- Evaluated binary SHA-256:
  `524bf81570a9e6188ee585358001f3b6afd5e2951fce546ddba2657aa2627a28`.

Coverage, structural validity, local option verification, and latency passed,
but the precision and complete abstention requirements are release-blocking.
Per preregistration, individual outcomes are not used for tuning and this set
will not be rerun. A materially stricter generic binding architecture must be
evaluated on a newly selected command-disjoint partition.
