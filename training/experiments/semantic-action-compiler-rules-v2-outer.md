# Semantic-action compiler rules v2 outer validation

Status: preregistered; outer validation remains sealed.

The inner hybrid passed its frozen gate at 6/92 accepted outputs, 92/92
Rust-valid outputs, and 79/92 first-command matches. This authorizes one outer
validation run of the unchanged v2 compiler rules over the unchanged
grounded-count-v2 neural fallback.

## Frozen procedure

1. Use the existing semantic-action decoder manifest and its 610-record outer
   training split and 78-record protected outer-validation split.
2. Initialize the grounded-count-v2 model from the same pinned CodeT5 weights
   and seed `20260826`.
3. Train once on all 610 outer-training records for exactly five epochs, the
   epoch selected by inner validation. Do not select on or compute loss against
   outer validation during training.
4. Generate the 78 outer-validation actions once. Apply the unchanged v2
   compiler before scoring, with the neural output as fallback wherever no rule
   activates.
5. Require at least 71 Rust-valid outputs (90%), 63 first-command matches (80%),
   and 8 reference-accepted outputs (10%). Record aggregate and per-rule metrics.
6. Do not inspect or modify rules from individual outer outcomes. If the gate
   fails, stop this recipe. If it passes, freeze it and preregister the protected
   98-record test comparison before opening test.

No generated command is executed. The outer test split remains sealed.
