# Semantic-action global lexicon beam v1

Status: preregistered before implementation and evaluation.

This is the second and final candidate-only decoder authorized by the bounded
program. It follows `semantic-action-candidate-beam-v1`, which achieved 92/92
Rust-valid documents and 84/92 first-command matches but no accepted references
because prompt-local candidates could not express many arguments.

## Frozen candidate inventory

For each training run, construct a sorted, deduplicated global lexicon from every
complete semantic word in that run's training targets. Never use evaluation
targets to construct it.

- Inner development: lexicon from the frozen 518 inner-training records.
- Outer validation, only if earned: lexicon from all 610 outer-training records.
- Comparison test, only if earned: reuse the outer-training lexicon unchanged.

At each semantic word, candidates are the entire global lexicon followed by the
prompt's deterministic local lexical spans. If a supervised word occurs in the
global lexicon, train that stable global choice. Otherwise train its first exact
local candidate. Words in neither inventory receive no candidate loss.

The inner global lexicon has 1,025 words. Its union with prompt-local candidates
covers 308/460 inner target words (66.96%) and every target word in 19/92 inner
examples. The oracle is therefore above the five-example acceptance requirement
without using inner target words in the lexicon.

## Frozen model and training

- Reuse the CodeT5-small encoder, four transferred decoder blocks, Rust action
  grammar, local candidate pooling, and action head from candidate v1.
- Add one learned 512-dimensional embedding per global word. Initialize each to
  the mean of the frozen CodeT5 input embeddings for that word's tokenizer IDs.
- Score global and local candidates with the same learned key/query projections.
- Remove copy-gate supervision. Total loss is action cross-entropy plus 0.25
  candidate-selection cross-entropy.
- Retain the existing optimizer: `5e-5` for transferred encoder/decoder weights,
  `5e-4` for project-owned parameters, zero weight decay, batch size 8.
- Tiny gate: derive its lexicon only from the selected eight records and require
  8/8 exact, valid, first-command, and accepted within 2,000 CUDA steps.
- Inner gate: train at most 50 epochs on 518 records. Select solely by lowest
  teacher-forced combined validation loss. No decoding result selects an epoch.

## Frozen decoder

Use the exact width-8 grammar beam from candidate-beam v1, except its atomic word
choices span the global-plus-local inventory. Global words emit their stored
UTF-8 bytes; local candidates emit their fixed source span. There is no byte
generator, learned copy gate, length penalty, repetition penalty, reranker, or
post-generation repair. The same summed normalized decision log-probability and
192-action bound apply.

## Gates and terminal rule

Inner development passes only with at least 90% Rust-valid, 80% first-command,
and 5% reference acceptance on all 92 records. Any failure ends the bounded
program without opening outer validation or test.

If inner passes, freeze its selected epoch count and recipe, initialize a fresh
model and lexicon from all 610 outer-training records, train once, and evaluate
the protected 78-record outer validation exactly once. It requires 90% valid,
80% first-command, and 10% acceptance. Failure ends the program. Only an outer
pass opens the 98-record comparison test exactly once.
