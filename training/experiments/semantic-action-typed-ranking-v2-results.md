# Semantic-action typed ranking v2 results

Status: failed the frozen progress gate.

The CPU gate passed and the eight-record overfit gate reached 8/8 exact and accepted at step 100. On the 92-record inner evaluation, validation loss selected epoch 4 at 3.819856. That checkpoint produced 1 exact action, 92 Rust-valid actions, 81 first-command matches, and 1 reference-accepted action. The generic runner marked one acceptance as progress, but the v2 preregistration required exceeding the v1 result of one; the frozen v2 progress gate therefore failed.

Attribution explains the failure. Relative to v1, command top-1 improved from 83 to 85 of 91 and argument-count top-1 improved from 16 to 19 of 93. Argument candidate top-1 fell from 84 to 76 of 229, however, and global-only top-1 fell from 3 to 1 of 66. The content-only byte objective moved some global words into top-4 and top-8 but did not make the correct absent word rank first. This rejects byte-order and role scaling as a sufficient answer to the missing semantic retrieval signal.

Outer validation and test remain sealed. Raw checkpoints and reports remain ignored under `training/artifacts/semantic-action-typed-ranking-v2/`; their hashes are recorded in the companion JSON.
