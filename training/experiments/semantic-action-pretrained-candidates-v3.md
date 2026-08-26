# Semantic-action pretrained candidates v3

Status: preregistered before implementation evaluation.

This is the final typed-variant allowance. V2 showed that order-aware bytes and role-specific scoring improve command ranking but do not retrieve prompt-absent argument words: global-only top-1 was 1/66 and argument top-1 was 76/229. The missing reusable signal is lexical semantics, which the existing pretrained CodeT5 encoder already owns.

## Frozen changes

1. Retain the v2 candidate inventory, grammar, role masks, argument-count head, width-8 beam, split, optimizer, epoch budget, and loss weights.
2. Tokenize every candidate with the same frozen CodeT5 byte-BPE tokenizer used for the source. Pad only to the maximum candidate-token length in the batch and carry token IDs plus a mask in `TypedActionBatch`.
3. Mean-pool the encoder's pretrained input-token embeddings over each candidate. Add this vector to both the normal and content-only candidate keys before their existing layer normalization. The embedding remains tied to the encoder; no second vocabulary matrix or candidate encoder is added.
4. Retain unigram bytes, byte bigrams, provenance, local context, role query scales, and the content-only global auxiliary loss unchanged. No new data, inventory expansion, beam change, reranker, repair, protected-data access, or pretrained-model call is allowed.

## Gates

- CPU: candidate tokenization/masking is deterministic; all five losses and gradients are finite; bounded grammar generation completes.
- Eight-record overfit: 8/8 first-command within 2,000 CUDA updates.
- Inner: 50 epochs on the same 518 records, selected only by combined validation loss on the same 92-record inner evaluation.
- Floors: at least 90% Rust-valid and 80% first-command match.
- Progress: at least 2/92 reference accepted. Target: at least 5/92 reference accepted.

If v3 fails to reach 2/92, stop the typed-model path and retain the compiler or retrieval-first architecture as the evidence-supported route. Outer validation and test remain sealed regardless of outcome.
