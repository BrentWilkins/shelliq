# Why ShellIQ is trying lexical candidates next

This document explains the project as if the reader has not followed the model
experiments so far. It describes the goal, the evidence we have, and why the
next experiment replaces free-form copying with deterministic lexical
candidates plus a learned selector.

## The product goal

ShellIQ is trying to turn a natural-language request into a shell command. For
example, a person might ask it to show running Docker containers and provide
local context that contains the command name `docker`. The eventual system
should produce the intended shell structure, not merely text that looks
shell-like.

The important constraints are unusual compared with a typical chatbot:

- It should be fully local. A request should not have to be sent to a hosted
  language model.
- It should be small enough to be practical on ordinary hardware.
- Its output must preserve exact command names, options, operands, quoting,
  pipes, and redirections. A nearly correct command can still be wrong or
  dangerous.
- The Rust implementation, rather than a Python approximation, remains the
  authority on what ShellIQ's structured output means.

“Tiny” here is relative to modern general-purpose language models. The current
research model is about 53 million parameters, not a tiny embedded classifier.
It is nevertheless much smaller and narrower than a multi-billion-parameter
chat model, and it can run without a cloud service. Compression and runtime
engineering would come later, after there is evidence that the model is useful.

This is best thought of as a **semantic compiler**. A conventional compiler
turns one precise representation into another. This model has the harder front
end: it must infer a precise shell program from ordinary language and supplied
context. Once it has made those semantic choices, deterministic project code
should do as much of the remaining work as possible.

## What the model produces

### `SemanticDocumentV2`: the meaning of the command

ShellIQ represents a shell command as a structured Rust-owned document called
`SemanticDocumentV2`. It is analogous to an abstract syntax tree: instead of a
single string, it records such things as commands, words with semantic roles,
pipelines, declarations, and redirections in explicit fields.

That distinction matters. In a plain string, `|`, `>`, an option, and a file
operand are just characters. In a semantic document, they have different
roles. Project code can validate the document, render it as shell text, parse
the rendered command again, and lower it back to semantic form. That
render/re-lower check catches many structurally invalid generations without
executing any generated command.

### `SemanticActionV1`: a small program that builds the document

Generating the full JSON form of `SemanticDocumentV2` asks a model to spend
capacity on braces, field names, ordering, and other serialization details.
The project therefore introduced `SemanticActionV1`, a compact sequence of
instructions for constructing the same document.

Some actions express fixed structure: start a command, add a word of a certain
role, finish a pipeline, and so on. Text inside a word is represented by UTF-8
byte actions, followed by an action such as `WORD_END`. The vocabulary has 320
actions, of which 256 are the possible raw byte values. The complete curated
corpus round-tripped through this action representation: 786/786 documents
encoded, decoded, and passed Rust validation. No target exceeded the frozen
192-action limit.

An action sequence is therefore not a second, approximate output format. Rust
decodes it into the authoritative semantic document and performs the final
validation.

### Rust grammar masking: forbid impossible next steps

At each decoding step, many of the 320 actions are impossible. For example,
the decoder should not finish a word before it has supplied a valid, nonempty
UTF-8 payload, nor start a redirect operand in a state where no redirect
exists.

Rust emits a manifest describing the allowed state transitions. During model
generation, Python reads that manifest and sets the probability of disallowed
actions to zero. This is called **grammar masking**. Python does not maintain a
separate handwritten copy of the grammar.

Grammar masking guarantees that each chosen transition is locally permitted.
It does not guarantee that the completed command answers the request. A model
can produce a valid structure containing the wrong command, option, or
operand. This distinction has been central to the results so far.

## The current neural architecture

The model has two main parts:

1. A pretrained CodeT5-small encoder reads the natural-language request and
   authoritative local context. Pretraining gives it a useful starting point
   for understanding language and code-like text.
2. A compact, project-owned decoder emits `SemanticActionV1` under Rust grammar
   masking. It attends to the encoder's representation while deciding the next
   structural or textual action.

This hybrid was chosen after a roughly 60-million-parameter model trained from
scratch memorized a tiny sample but produced 0/98 valid documents on held-out
commands. In a matched earlier comparison, ordinary fine-tuned CodeT5 produced
91/98 Rust-valid documents and chose the first command correctly on 86/98, but
still achieved 0/98 conservative reference acceptance. Its outputs were often
plausible yet had incorrect, duplicated, or misplaced options and operands.

Those results suggested keeping pretrained language understanding while
moving output structure into small, project-controlled machinery. They did
not establish that the hybrid would work; they established a reasonable next
hypothesis.

## Why the data is divided into several gates

The full curated dataset has 786 records. Its split is **command-disjoint**:
the same first-command identity cannot appear on both sides of a split. This is harder
than randomly withholding phrasings of commands the model already saw, but it
tests the capability the product actually needs—handling commands not present
in its small training set.

The established outer split contains:

- 610 outer-training records;
- 78 protected outer-validation records;
- 98 protected test records.

For architecture development, the 610 outer-training records are divided
again into 518 inner-training and 92 inner-validation records. We can inspect
inner-validation failures while improving the architecture. We must not keep
checking the outer validation or test sets after every idea.

The reason is the same as not letting a student see an exam while revising the
course. Every time researchers inspect held-out failures and change the model,
they indirectly learn from that set. Eventually the model can be tailored to
the “exam” even if no gradient was computed from its answers. The inner split
is the development workspace. Outer validation is a one-time promotion gate
for a frozen recipe, and test is reserved for a recipe that already passed.

## What has failed, and what each failure taught us

Every neural version first passed a deterministic eight-record overfit test.
That means the implementation could learn and reproduce eight examples
exactly. It is an important plumbing check, but it is not evidence of
generalization.

### 1. Action decoder with generated bytes

The first compact decoder generated every byte of every word itself. On the
eight-record check it reached 8/8 exact, valid, accepted outputs. On the 92
unseen-command inner examples it achieved:

- 49/92 Rust-valid outputs;
- 0/92 correct first commands;
- 0/92 accepted references;
- 0/92 exact action sequences.

The model could memorize spelling but could not reliably spell unseen command
names. Many sequences also ran into the 192-action ceiling.

### 2. Independent byte pointers

The next version allowed the decoder to copy a byte from the input instead of
generating it. A copy oracle had shown that 91/93 command words (97.85%) in the
inner set appeared as exact source spans, so copying directly addressed a real
bottleneck.

This improved the inner result to 75/92 Rust-valid and 41/92 correct first
commands, but acceptance remained 0/92. The problem was that each byte was a
separate choice. A model could copy plausible letters from unrelated input
positions, producing corruptions such as `pytho3`, `dock`, repeated flags, or
repeated operand fragments. Seeing the complete word `docker` in the input did
not force six independently chosen pointers to stay inside that occurrence.

### 3. Monotonic byte copying

A decoding-only restriction then prevented pointers from moving backward
while constructing a word. Applied to the same selected checkpoint, it
improved Rust validity from 75/92 to 79/92 and first-command accuracy from
41/92 to 53/92. Acceptance was still 0/92.

This confirmed that pointer jumps were one cause of corruption. It did not
solve the problem because moving forward one byte at a time can still skip,
stop early, or switch to another later fragment.

### 4. Learned atomic source spans

The next decoder predicted a start and end position and copied the entire
source span in one operation. Atomic expansion means that after selecting the
correct span for `docker`, the decoder cannot accidentally emit `dock` by
jumping between its letters.

Again, the eight-record gate passed 8/8. On the inner split, however, the free
start and end heads did not learn reliable boundaries from only 518 records:

- 62/92 Rust-valid outputs;
- 41/92 correct first commands;
- 0/92 accepted references;
- 0/92 exact action sequences.

It regressed relative to both independent-byte and monotonic decoding.
Generated outputs still contained truncated words such as `pyth` and `dock`
and repeated fragments. Atomic copying made a *correctly selected* span safe;
it did not make span selection reliable.

## The next hypothesis: extract choices deterministically, learn only selection

The common failure is now fairly specific. The model is being asked to invent
text boundaries, even though conventional code can enumerate sensible lexical
units from the input exactly.

The next experiment separates those responsibilities:

1. A deterministic lexer scans the supplied source. It considers complete
   non-whitespace tokens after removing common surrounding wrappers, smaller
   shell-shaped atoms inside those tokens, and the contents of quoted text.
   Candidates longer than 128 UTF-8 bytes are excluded, and exact source spans
   are deduplicated. It performs no learning and makes the same candidates
   every time for the same input.
2. CodeT5 already produces a contextual representation for every source
   position. Representations aligned with the bytes of each candidate are
   pooled into one candidate representation. Thus two identical spellings in
   different contexts can still carry different contextual evidence.
3. At the first byte position of each semantic word, the transferred CodeT5
   decoder state scores the complete candidates. A learned gate chooses
   between atomic candidate copying and ordinary grammar-masked byte generation.
   The natural gate threshold is 0.5: above it the candidate path wins; below
   it the byte generator remains the fallback for text absent from the source.
4. If copying wins, the selected candidate is emitted atomically as its exact
   UTF-8 bytes plus `WORD_END`. The model never predicts candidate boundaries.
5. Rust grammar masking and final render/re-lower validation remain unchanged.

In the Docker example, the choice presented to the model is the complete
candidate `docker`, with known start and end boundaries. The model must still
decide whether `docker` is the requested command, an operand, or irrelevant
context. But if it selects that candidate, boundary prediction cannot silently
turn it into `dock`. This converts a difficult sequence of character-level
decisions into one classification-like decision.

This is not a dictionary of commands and it does not make the answer
deterministic. The extractor only proposes text spans; it does not label one as
the answer. The hard semantic problem remains: selecting the right command,
options, operands, and structure for the request. The narrower claim is that a
neural network should not have to rediscover token boundaries that the source
already exposes.

## Why this is currently the best option

The evidence favors this route for four reasons:

- Correct command words are already present in the source in almost every
  inner example. Candidate extraction can preserve them exactly.
- Each prior copying result points to boundary and traversal errors, rather
  than a lack of accessible source text.
- Deterministic extraction removes degrees of freedom from the learned model.
  This is valuable with only hundreds of training records.
- It preserves the useful pieces already built: the CodeT5 encoder, action
  decoder, Rust action codec, grammar mask, split manifests, evaluation code,
  and source-alignment tests.

It is still a risky experiment. Exact candidates do not teach the model which
candidate is correct. The frozen lexical extractor's development oracle found
candidates for 91/93 inner-validation command words (97.85%) but only 238/460
words overall (51.74%), so generated fallback text remains important. The
earlier unrestricted source-substring oracle covered 258/460 words; the lexical
inventory deliberately gives up some arbitrary substrings so that the selector
sees better-formed choices. Candidate lists can still contain many distractors.
Finally, 0/92 reference acceptance in every action-decoder experiment says
that command spelling is not the only bottleneck.

For those reasons, the expectation is not “this will probably work great.” The
expectation is “this is the cleanest test of the strongest remaining
explanation for the failures, and it produces useful infrastructure even if
the selector fails.”

## What remains useful if this experiment fails

The deterministic extractor and its coverage report are model-independent.
They can later support a rules system, retrieval, reranking, another compact
classifier, or constrained decoding in a different model. Candidate indices
also provide interpretable diagnostics: we can distinguish “the correct text
was never offered” from “the model was offered it and selected something
else.”

The Rust codec and grammar continue to define valid outputs. The pretrained
encoder, action decoder, protected splits, evaluators, and tiny-overfit harness
also remain reusable. A failed selector would therefore narrow the problem
instead of discarding the project.

## Gates and stopping rules

The experiment should progress in increasing order of cost and evidentiary
importance:

1. **Deterministic extraction and oracle coverage.** Tests must prove that
   candidates map to exact source bytes, preserve UTF-8, obey the 128-byte
   bound, and expand to valid action payloads. The measured development oracle
   is 91/93 command words and 238/460 words overall. If implementation changes
   lose that command coverage, fix or stop the extractor rather than train
   around it.
2. **Portable CPU plumbing.** A small batch must complete a finite forward
   loss, backward pass, and grammar-safe generation on CPU. This is a fast,
   hardware-independent correctness test—not the intended full training
   environment.
3. **Tiny deterministic overfit.** On CUDA, the model must reproduce all eight
   selected records exactly, with 8/8 Rust-valid documents, first commands, and
   accepted references, within the existing 2,000-step cap. Failure means the
   selector or integration cannot even express and learn known examples.
4. **Inner development.** Train on the frozen 518 records for at most 50 epochs
   and select the checkpoint only by lowest teacher-forced inner-validation
   action loss. On the 92 inner records, promotion requires at least 90%
   Rust-valid output, 80% first-command accuracy, and 5% conservative reference
   acceptance. Anything below any threshold stops this recipe.
5. **Protected outer validation.** Only after the inner gate passes, freeze the
   recipe and epoch count, retrain once on all 610 outer-training records, and
   evaluate the 78 outer-validation records once. Promotion requires at least
   90% validity, 80% first-command accuracy, and 10% acceptance. The previously
   defined “great” levels are 95%, 90%, and 25% respectively.
6. **Protected test comparison.** Only a recipe that passes outer validation
   may open the 98-record test comparison against the already locked CodeT5
   result. Test results are evidence, not another tuning signal.

No generated command is executed during these experiments. A passing model is
also not automatically a release model: shadow evaluation, deployment safety,
runtime export, and quantization are separate later decisions.

## The decision this experiment can support

A pass would show that a small local model can combine pretrained language
understanding, deterministic lexical preservation, and Rust-constrained
semantic generation well enough to justify protected evaluation.

A failure would be equally actionable if the diagnostics are kept separate:

- low candidate-oracle coverage means extraction is inadequate;
- high oracle coverage but poor candidate accuracy means learned selection is
  inadequate;
- correct words but invalid documents means structural decoding is inadequate;
- correct first commands but low acceptance means options, operands, quoting,
  redirects, or composition are now the dominant bottleneck.

That is why this is the next experiment. It does not assume the model will
succeed. It removes a failure mode we can solve exactly, protects the scarce
evaluation evidence, and gives the next result a clear interpretation.
