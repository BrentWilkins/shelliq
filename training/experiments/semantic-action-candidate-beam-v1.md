# Semantic-action candidate beam v1

Status: preregistered before implementation and evaluation.

This is the first of at most two candidate-only sequence-decoding experiments
authorized after `semantic-action-candidate-v1`. It reuses that experiment's
selected epoch-17 inner checkpoint without retraining. The 78-record outer
validation and 98-record comparison test remain unopened.

## Hypothesis

Candidate-only greedy decoding reached 82/92 Rust-valid and 75/92 first-command
matches but accepted no references. The model may assign useful probability to
the correct full sequence without placing every structural or candidate choice
at rank one. A small grammar-constrained beam can preserve alternatives across
the whole semantic document and avoid greedy argument loops.

## Frozen decoder

- Beam width: 8.
- At fixed-action grammar states, expand every Rust-manifest-allowed action and
  add its log probability after renormalizing over only allowed actions.
- At the start of every byte-payload word, disable generator fallback and the
  learned copy gate. Expand the eight highest-scoring deterministic lexical
  candidates and add candidate log probability after renormalizing over the
  full candidate inventory.
- Emit each candidate plus `WORD_END` as one atomic macro. Learned byte
  boundaries and byte generation are never used.
- Merge hypotheses with identical action sequences, retaining the highest
  score.
- Rank by the unmodified sum of decision log probabilities. Use no length,
  repetition, coverage, or task-specific penalty.
- Stop when the best completed hypothesis scores at least as highly as every
  active hypothesis, because all future log-probability increments are
  non-positive. Otherwise stop at the existing 192-action bound.
- Return the highest-scoring completed hypothesis, or the highest-scoring
  bounded incomplete hypothesis only if none completes.

No threshold, beam width, scoring term, or stopping rule will be tuned after
viewing results.

## Gates and stopping rule

Focused tests must establish grammar validity, atomic candidate emission,
deduplication, score ordering, the 192-action bound, and deterministic output.

The frozen 92-record inner-development gate requires all of:

- Rust-valid render/re-lower at least 90%.
- First-command match at least 80%.
- Reference acceptance at least 5%.

Failure stops this recipe. One and only one further candidate-only decoder may
then be preregistered and evaluated. Passing freezes this decoder and earns one
outer-validation attempt after retraining the model for the already-selected 17
epochs on all 610 outer-training records. The comparison test opens only after
outer validation passes its existing 90%/80%/10% gates.
