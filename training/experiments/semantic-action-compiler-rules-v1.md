# Semantic-action compiler rules v1

Status: preregistered before hybrid evaluation.

Neural candidate experiments now show a stable limit: valid structure and command choice are adequate, but exact option/operand recipes do not transfer reliably across command-disjoint families. The next evidence-supported layer is a tiny deterministic compiler for families with an authoritative, inspectable rule, falling back to the selected grounded-count v2 checkpoint everywhere else.

## Frozen rules

Rules inspect only `command`, `platform`, `instruction`, and `context`. They must not inspect record IDs, responses, semantic targets, or evaluation outcomes at runtime.

1. `ss` summary: when the instruction asks for a summary or counts by state and context documents `ss -s`, emit `ss -s`.
2. `ss` socket listing: recognize requested TCP, UDP, listening, all-state, numeric, process-owner, and established-state constraints. Select only single-letter options documented in context, preserve their first appearance order, and combine them into one short-option word. Numeric output is included for socket listings when context documents `-n`; protocol alternatives not requested are excluded. An established-state request appends `state established`.
3. `pmap` detail mode: when context documents `-x` for an extended map or `-X` for kernel-detail columns and the instruction requests that mode, extract exactly one decimal PID from the instruction and emit `pmap <documented-mode> <pid>`.
4. Encode every rule result as `SemanticDocumentV2` through the Rust `semantic-actions` encoder. If no rule matches or any rule cannot produce a complete supported recipe, use the neural output unchanged.

## Evaluation gate

- Unit tests cover positive and negative rule matching, documented option order, protocol exclusion, PID extraction, and Rust action encoding.
- Evaluate the already-selected grounded-count v2 checkpoint on the same 92 open inner records, replacing only matched outputs.
- Require at least 90% Rust-valid, 80% first-command match, and 5/92 reference accepted.
- Report rule coverage and per-rule acceptance separately from fallback metrics.

These rules were developed from open inner-development failures. Even if the inner gate passes, outer validation is required before any promotion; test remains sealed.

## Pre-implementation amendment

The initial draft proposed a Darwin `xargs` empty-input rule that emitted `rm` even though the instruction did not request removal. It was removed before implementation or evaluation because it violated the contract to derive operands only from the instruction. The general, context-grounded `pmap` detail rule above replaces it.
