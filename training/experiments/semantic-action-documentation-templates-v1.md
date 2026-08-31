# Semantic-action documentation templates v1

Status: preregistered for open inner development; outer validation and test are
not used by this experiment.

## Hypothesis

A command- and platform-scoped retriever over independently authored TLDR
documentation can supply complete typed command recipes for command families
absent from neural training. Generic placeholder binding can then substitute
values explicitly present in the request without command-specific rules. The
Rust semantic-action codec remains the authority for structural validity.

## Frozen inputs and leakage controls

1. Documentation comes only from `tldr-pages` records in the converted
   distributable semantic corpus at revision v2.3. `shelliq-curated` records are
   excluded from the retrieval index.
2. Evaluation uses only the already-open 92-record inner-development split from
   `semantic-action-decoder-v1.manifest.json`.
3. Retrieval may use command, platform, instruction, and context. It must not use
   the reference response, record identifier, outcome, or target-derived fields.
4. Candidate templates must match the requested command and platform. No
   per-command weights, rules, aliases, or corrections are allowed.
5. Slots are detected from generic TLDR placeholder syntax. Binding may use only
   literal values extracted from the request instruction. Unbound slots retain
   their documented placeholder rather than inventing a value.

## Frozen measurements and feasibility gate

- Report documentation coverage, top-1 and top-8 typed-template skeleton
  coverage, distinct commands covered at top eight, exact/reference acceptance
  after top-1 binding, first-command match, and Rust validity.
- Every emitted top-1 document must pass the Rust action encoder and decoder.
- Continue to selector integration only if top-eight skeleton coverage reaches
  at least 20% of all 92 inner records, spans at least ten distinct command
  families, and exceeds top-one skeleton coverage. This is a retrieval
  feasibility gate, not a promotion gate.
- If the feasibility gate fails, stop and attribute missing coverage between
  absent command documentation, absent recipe, retrieval rank, and binding.

No generated command is executed. The previously opened outer-validation result
is not consulted, and the 98-record protected test remains sealed.
