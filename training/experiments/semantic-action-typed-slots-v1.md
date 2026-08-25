# Semantic-action typed slots v1

Status: preregistered before implementation and evaluation.

This is the first of at most three typed-slot variants. It follows the bounded
candidate-decoder program: local candidates plus grammar beam achieved perfect
syntax and 91.3% first-command accuracy but no accepted references; a separate
global candidate table then collapsed command routing to 0%. This experiment
removes that global/local classification boundary and adds explicit argument
count decisions.

Outer validation and the comparison test remain sealed throughout this program.

## Frozen typed oracle

Build candidates using only the input prompt and target words from the current
training split. Never use evaluation target words.

1. Prompt-local candidates are the existing trimmed whitespace terms,
   shell-shaped atoms, and quoted contents.
2. Split prompt compounds on internal hyphens.
3. Substitute prompt numbers into digit-bearing option shapes observed in
   training targets.
4. Compose path-like prompt values and identifiers with `::`.
5. Prefix filename-like prompt values with `@`.
6. Compose prompt long options and numbers with `=`.
7. Bundle consecutive documented single-letter short flags in source order,
   using lengths two through five.
8. Add each training target word only in the byte-payload grammar role or roles
   where training observed it.

Deduplicate by exact UTF-8 bytes. A local or derived value is allowed in every
byte-payload role; a training-only value is allowed only in its observed roles.

On the frozen 518/92 inner split this inventory covers 321/460 target words
(69.78%) and every target word in exactly 23/92 references (25%). This passes the
preregistered oracle floor. Command coverage remains 91/93.

## Frozen model

- Reuse the CodeT5-small encoder, four transferred decoder blocks, 320-action
  Rust grammar, and width-8 grammar beam.
- Represent every candidate from its UTF-8 byte embeddings. Add pooled aligned
  CodeT5 source context when it has an exact local span. Add learned boolean
  feature embeddings for local, derived, and training-global provenance.
- Score the one deduplicated inventory with a shared candidate key/query, then
  mask it by the current grammar byte-payload role. There are no separate global
  and local output classes and no byte generator.
- Add a 13-class argument-count head for counts zero through twelve. Supervise it
  at the first command-name byte for every command.
- Total loss is action cross-entropy plus 0.25 candidate cross-entropy plus 0.25
  argument-count cross-entropy.
- Retain the existing optimizer and batch size: `5e-5` for transferred weights,
  `5e-4` for project-owned weights, zero weight decay, batch size 8.
- Select up to 50 epochs solely by lowest combined teacher-forced validation
  loss. Decoded metrics never select an epoch.

## Frozen decoder

- At a command-name slot, jointly expand the top four role-valid word candidates
  and top four argument counts, then prune to the width-8 beam.
- At other word slots, expand the top eight role-valid candidates.
- Once a command name is emitted, allow `ARGUMENT_START` while emitted arguments
  are below the selected count and forbid it after the count is reached. Before
  the count is reached, do not allow command termination or redirect start.
- Increment the emitted count after each atomic `argument_bytes` candidate.
- Retain summed normalized decision log-probability, duplicate sequence merging,
  no heuristic penalties, and the 192-action bound.

## Gates

The real CPU path must have finite losses and gradients and grammar-bounded
generation. An eight-record lexicon derived only from those records must reach
8/8 exact, Rust-valid, first-command, and accepted within 2,000 CUDA steps.

Inner-development evidence of progress requires at least one accepted reference
while retaining at least 90% Rust validity and 80% first-command accuracy. The
original promotion target remains at least 5/92 accepted with those same
validity and command floors. A result below the progress definition stops this
variant; at most two evidence-driven typed variants may follow. No result in
this typed-development program opens outer validation or test.
