# Semantic-action pretrained candidates v3 results

Status: failed the progress gate; stop the typed-model path.

The CPU gate passed, and the eight-record overfit gate reached 8/8 exact and accepted at step 100. On the 92-record inner evaluation, combined validation loss selected epoch 4 at 3.826066. The selected checkpoint produced 0 exact actions, 92 Rust-valid actions, 82 first-command matches, and 0 reference-accepted actions. It therefore missed both the 2/92 progress gate and the 5/92 milestone. Outer validation and test remain sealed.

Attribution shows what the pretrained representation did and did not buy. Argument top-1 recovered from v2's 76 to 84 of 229, matching v1, while local-only top-1 improved from v1's 108 to 111 of 158. Global-only top-1 remained 1 of 66, and derived candidates were 0 of 14. Mean-pooled pretrained CodeT5 token embeddings help source-local lexical ranking but do not provide the prompt-to-absent-word retrieval relationship this task needs.

Gold argument counts would raise the selected v3 checkpoint to 4/92 exact and accepted, so count prediction is a secondary blocker. That counterfactual still misses the five-acceptance milestone, and the preregistered v3 rule requires stopping rather than opening another typed variant after seeing results.

The evidence-supported next architecture is retrieval-first or compiler-backed: use explicit knowledge to propose the missing semantic values, then reuse the project-owned grammar, typed candidate masks, argument-count diagnostics, and validity-preserving decoder. Raw checkpoints and reports remain ignored under `training/artifacts/semantic-action-pretrained-candidates-v3/`; their hashes are recorded in the companion JSON.
