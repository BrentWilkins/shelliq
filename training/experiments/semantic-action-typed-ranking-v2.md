# Semantic-action typed ranking v2

Status: preregistered before implementation and evaluation.

This is the first evidence-driven follow-up to `semantic-action-typed-slots-v1`.
The frozen attribution study showed that all 23 fully covered references decode
exactly with gold words and counts, while gold words alone recover 13/23 and gold
counts alone recover only 2/92. Command candidates are 91.2% top-1, but argument
candidates are 36.7%; global-only candidates are just 4.5% top-1. This variant
changes only candidate representation and ranking supervision.

Outer validation and the comparison test remain sealed.

## Frozen changes

1. Retain the v1 typed oracle, deduplicated inventory, provenance features,
   explicit argument-count head, action model, optimizer, split, and decoder.
2. Add a deterministic 512-bucket UTF-8 bigram histogram to every candidate,
   using bucket `(first * 257 + second) mod 512` and normalized counts. Add its
   learned embedding sum to the existing unigram representation. This restores
   character-order information without material candidate-encoding memory.
3. Add one learned multiplicative query scale per grammar byte-payload role.
   Candidate scores become role-specific during both training and beam decoding;
   command and argument ranking no longer share an identical query geometry.
4. Add a content-only auxiliary candidate loss for training targets observed in
   the training-global inventory. It removes local source context and local/
   derived provenance from candidate keys, retains order-sensitive content and
   global provenance, masks candidates by role and global availability, and
   teaches the selector to retrieve a word when prompt-local context is absent.
5. Total loss is action cross-entropy plus 0.25 normal role-specific candidate
   loss plus 0.25 argument-count loss plus 0.10 content-only global loss.

No inventory expansion, beam-width change, count-decoder change, penalty,
reranker, repair, or protected-data access is allowed.

## Gates

- CPU: finite action, candidate, count, content-only, and total losses and
  gradients; bounded grammar generation.
- Eight-record overfit: 8/8 exact, valid, first-command, and accepted within
  2,000 CUDA steps.
- Inner development: at most 50 epochs on the same 518 records, selected solely
  by lowest combined teacher-forced validation loss.
- Evidence progress must retain at least 90% Rust validity and 80% first-command
  accuracy and exceed v1's 1/92 accepted references.
- The target milestone is at least 5/92 accepted with the same floors.

Failure to exceed 1/92 stops v2 and leaves one typed variant allowance. Reaching
at least 2/92 is new progress; reaching 5/92 completes the requested milestone.
