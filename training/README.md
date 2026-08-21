# ShellIQ training

Development-only JAX/Flax NNX code adapting
`Qwen/Qwen2.5-Coder-0.5B-Instruct`. Nothing in this directory ships in the
Rust binary.

## Current status

- Hand-written Qwen2 decoder: RMSNorm, RoPE, SwiGLU, grouped-query attention,
  causal masking, and tied embeddings.
- Hugging Face safetensors loader with the required linear-weight transpose.
- Padding-aware positions and attention for batched supervised fine-tuning.
- Masked next-token loss using `-100` for prompt and padding labels.
- Scaled LoRA on q/k/v/o and gate/up/down projections. A distinct NNX parameter
  type ensures that the optimizer updates adapters only.
- Strict JSONL records with per-row source, license, provenance, command,
  platform, and distributable/personal corpus tags.
- Command-grouped deterministic splits, optional source/platform holdouts,
  Qwen chat-template tokenization, and fixed-shape right-padded batches.
- Orbax checkpoints containing LoRA weights, optimizer moments, step, and
  compatibility metadata.
- Source builders for the pinned tldr ZIP, reviewed NL2Bash rows, successful
  Claude Code Bash calls, and local-index examples, plus a project-owned
  reviewed corpus for compositional commands.
- Mandatory self-testing privacy gate for personal builders, synthetic training
  canaries, and a post-training extraction gate.
- Held-out evaluation with functional-equivalence hooks and separate flag,
  option-argument, operand, exact-match, and option-known metrics.
- Zsh-first lossless syntax contract, native-Zsh corpus audit, and measured
  parser bake-off. See [`SHELL_AST.md`](SHELL_AST.md) and the
  [AI pipeline architecture](../docs/AI_PIPELINE.md).
- PEFT adapter and merged Hugging Face safetensors export, plus guarded llama.cpp
  adapter/GGUF conversion at Q6_K or Q8_0.

The builders and export path are implemented and unit tested. A real fine-tuning
run, post-training benchmark, and real checkpoint-to-GGUF load test have not
been claimed yet.

## Setup and checks

From this directory:

```sh
uv sync --dev
uv run pytest -q
uv run ruff check .
uv run python scripts/check_parity.py
uv run python scripts/smoke_train.py
cargo run --release -p shelliq-syntax --bin audit-zsh-corpus -- \
  artifacts/tldr-v2.3.jsonl
```

The parity script requests highest-precision float32 JAX matrix multiplication.
On NVIDIA, JAX's normal float32 policy may use TF32; that is a reasonable
training policy but can make an architecture comparison against PyTorch CPU
float32 fail for numerical rather than structural reasons.

## Measured GPU smoke test

On 2026-08-06, an RTX 4090 with 23,028 MiB ran the real 0.5B model using
rank-16 LoRA, bfloat16 base parameters, batch size 1, and sequence length 128:

- 494,032,768 base parameters and 8,798,208 LoRA parameters
- 14.79 seconds for compilation plus the first step
- 0.063 seconds for the second step
- 2.74 GiB peak JAX device memory

These are smoke-test measurements, not capacity estimates. Longer sequences,
larger batches, optimizer choice, rematerialization, and allocator state all
change peak memory.

## Batch and dataset contract

`train_step` accepts three arrays shaped `[batch, sequence]`:

- `input_ids`: tokenizer IDs
- `attention_mask`: 1 for real tokens and 0 for padding
- `labels`: desired token IDs, with `-100` wherever loss is ignored

Labels are shifted inside the loss. For instruction tuning, the complete prompt
is masked so only assistant tokens contribute loss.

Each schema-version-1 JSONL row carries enough metadata to audit its origin:

```json
{"schema_version":1,"record_id":"tldr:find:largest","corpus":"distributable","source":"tldr-pages","license":"CC-BY-4.0","provenance":"common/find.md@v2.3:example-1","command":"find","platform":"linux","instruction":"List five largest files.","response":"find . -type f -printf '%s %p\\n' | sort -nr | head -5","context":"find: -type f selects regular files."}
```

`load_jsonl`, `write_jsonl`, `split_records`, `tokenize_record`, and
`collate_sft` live in `shelliq_training.data`. They fail on missing provenance
or license, mixed corpora, duplicate IDs, unstable assistant boundaries, and
records that exceed the configured sequence length.

Splits hash the command name, not the row. Sources or platforms can be forced
into test; if one held-out row names a command, every row for that command is
forced to test. `validate_heldout_splits` is the hard no-command-leakage gate.

## Source builders

The checked-in tldr v2.3 archive currently produces 30,351 records for 5,085
commands with the default Linux common-page mapping, plus Darwin-specific
`osx/` rows:

```sh
uv run python scripts/build_dataset.py tldr \
  --archive ../crates/harvest/vendor/tldr-pages.en.zip \
  --revision v2.3 \
  --output data/tldr.jsonl
```

For an unattended plumbing check, build that corpus in the project-local,
gitignored `artifacts/` directory and overfit four deterministic real rows on a
GPU:

```sh
uv run python scripts/build_dataset.py tldr \
  --archive ../crates/harvest/vendor/tldr-pages.en.zip \
  --revision v2.3 \
  --output artifacts/tldr-v2.3.jsonl

uv run python scripts/smoke_finetune.py \
  --dataset artifacts/tldr-v2.3.jsonl \
  --checkpoint artifacts/tldr-smoke/checkpoint
```

`smoke_finetune.py` uses command-grouped splits, rejects unresolved templates
and invalid shell quoting, evaluates four command-disjoint rows, and requires
the four-row training loss to fall by at least 10 percent. It does not read
NL2Bash or personal sources. On the 2026-08-06 RTX 4090 run, automated preflight
accepted 30,345 of 30,351 rows, training loss moved from 2.8963 to 0.0000 in 80
steps, and the selected completion changed from prose to the exact expected
`az tag create -n tag_name`. Held-out loss worsened from 3.2943 to 5.5581, as
expected for deliberate four-example memorization. This proves the real-corpus
training/checkpoint path works; it is not a quality result or a substitute for
the later corpus audit.

### One-pass baseline pilot

After building and auditing `artifacts/distributable-v1.jsonl` as described
below, an RTX 4090-class GPU can run a first signal-finding pilot:

```sh
uv run python scripts/smoke_finetune.py \
  --dataset artifacts/distributable-v1.jsonl \
  --sequence-length 256 \
  --train-examples 4000 \
  --eval-examples 256 \
  --batch-size 4 \
  --steps 1000 \
  --learning-rate 2e-4 \
  --priority-source shelliq-curated \
  --heldout-probe \
  --checkpoint artifacts/distributable-pilot-v1/checkpoint \
  --report artifacts/distributable-pilot-v1.report.json
```

This retains every usable curated row in each split, then fills the 4,000-row
training and 256-row evaluation selections with deterministic, command-distinct
TLDR examples. Train and evaluation remain command-disjoint. The report pins the
dataset and selected record IDs by SHA-256 and
records initial/final train and held-out losses plus train and held-out
generation probes. A useful first signal is falling held-out loss and a held-out
probe that becomes more command-like without merely copying the training probe.

This is deliberately a raw-shell baseline. It can tell us whether the data and
LoRA recipe carry useful NL-to-command signal, but it is not the final
Semantic-AST training target or a task-success evaluation.

The 2026-08-07 RTX 4090 curated-priority run selected 493 curated plus 3,507
TLDR training rows and 35 curated plus 221 TLDR held-out rows. In 1,000 steps
(226 seconds), train loss moved from 3.1927 to 0.7205 and held-out loss from
3.3188 to 1.5158. The curated training probe became the exact expected
`just deploy staging`. The unseen curated probe became a plausible command but
missed the required preservation semantics: it produced
`cp -r path/to/source_directory path/to/destination_directory` instead of
`cp -a src/ dest/`. This is positive evidence for instruction-to-command
generalization, but also concrete evidence that held-out flag and semantic
accuracy need broader evaluation and further training work.

Evaluate the saved adapter against the same deterministic held-out split:

```sh
uv run python scripts/evaluate_checkpoint.py \
  --dataset artifacts/distributable-v1.jsonl \
  --checkpoint artifacts/distributable-pilot-curated-v1/checkpoint \
  --output artifacts/distributable-pilot-curated-v1.eval-64.json \
  --examples 64 \
  --sequence-length 256 \
  --priority-source shelliq-curated \
  --option-arities evaluation/curated-option-arities-v1.json
```

The first 64-example comparison contained 35 curated and 29 TLDR rows. From
base model to trained adapter, primary-command accuracy moved from 0% to 88.5%,
native-Zsh validity from 53.1% to 96.9%, simple-command parse rate from 62.3%
to 98.4%, and exact match from 0% to 7.8%. Curated flag F1 reached 0.542. Two
trained outputs were invalid: one truncated an unmatched quote and one emitted
`<m>`. The versioned sidecar annotates all 35 curated held-out rows and covers
34 simple-command examples after the compound-command exclusion. On that
annotated subset, trained option-argument accuracy was 30.0% and operand exact
match was 26.5%, confirming that argument binding and operand selection are the
next quality bottlenecks. The evaluator also retains its zero-arity lexical
floor separately so annotated and unannotated scores cannot be confused.

NL2Bash ingestion requires a file of reviewed 1-based line numbers, an audited
license identifier, and a revision. It cannot bulk-accept the upstream files:

```sh
uv run python scripts/build_dataset.py nl2bash \
  --instructions path/to/all.nl --commands path/to/all.cm \
  --reviewed-lines reviewed-lines.txt --revision COMMIT --license LICENSE-ID \
  --output data/nl2bash.jsonl
```

The checked-in [`corpus/`](corpus/) contains first-party schema-v1 JSONL rows
that `load_jsonl` reads directly. It is separate from generated `artifacts/`
and is guarded by cross-file schema, provenance, command-label, native Zsh,
portable-shell, lossless CST, and semantic-lowering tests. See the corpus README
for its row conventions and explicit syntax/dialect exemption ratchets.

Merge the pinned TLDR artifact with every curated family into a new training
candidate without editing either source:

```sh
uv run python scripts/merge_corpus.py \
  --tldr artifacts/tldr-v2.3.jsonl \
  --output artifacts/distributable-v1.jsonl
```

The merge preserves TLDR row order, appends curated files in filename order,
rejects record IDs repeated across any input, and refuses to overwrite existing
outputs. It also writes `distributable-v1.jsonl.manifest.json` with per-input
SHA-256 hashes and record, command, source, platform, and curated-family counts.

Freeze the candidate's quality and split facts in a separate reproducible audit:

```sh
uv run python scripts/audit_corpus.py \
  --dataset artifacts/distributable-v1.jsonl \
  --output artifacts/distributable-v1.audit.json \
  --seed 2026
```

The audit fingerprints the dataset, applies the same preflight used by the GPU
smoke run, validates command-disjoint splits, and reports composition, duplicate
groups, field lengths, lexical option coverage, and split distributions. It
refuses to overwrite an existing report so a reviewed audit cannot change in
place.

For the pinned TLDR v2.3 artifact, five rows are intentionally rejected for
unmatched shell quotes inherited from their upstream pages:
`tldr:linux:common/exo:5`, `tldr:linux:common/lwp-request:3`,
`tldr:linux:common/mu:6`, `tldr:linux:linux/logwatch:1`, and
`tldr:darwin:osx/rargs:3`. The escaped-brace `fd-format:5` row is valid and has
regression coverage so it cannot again be mistaken for an unresolved
`{{placeholder}}`.

Convert the curated source rows into versioned semantic training targets from
the repository root:

```sh
cargo run -p shelliq-syntax --bin convert-semantic-corpus -- \
  training/corpus \
  training/artifacts/curated-semantic-v2.jsonl \
  training/artifacts/curated-semantic-v2.manifest.json
```

The converter preserves prompt and provenance fields, keeps the original shell
response beside the structured target, validates semantic render/re-lowering,
and enforces both exemption files as ratchets. The current result is 626 of 626
rows converted, with no CST exemptions, no semantic exemptions, and 19
explicitly reported rows whose semantic rendering normalizes whitespace.

Train and evaluate the semantic-target pilot separately from the raw-shell
baseline:

```sh
uv run python scripts/smoke_finetune.py \
  --semantic-dataset artifacts/curated-semantic-v2.jsonl \
  --sequence-length 384 \
  --train-examples 400 \
  --eval-examples 34 \
  --batch-size 2 \
  --steps 1000 \
  --learning-rate 2e-4 \
  --priority-source shelliq-curated \
  --heldout-probe \
  --checkpoint artifacts/semantic-pilot-v2/checkpoint \
  --report artifacts/semantic-pilot-v2.report.json

uv run python scripts/evaluate_semantic_checkpoint.py \
  --semantic-dataset artifacts/curated-semantic-v2.jsonl \
  --checkpoint artifacts/semantic-pilot-v2/checkpoint \
  --output artifacts/semantic-pilot-v2.eval.json \
  --examples 35 \
  --sequence-length 384
```

On the 2026-08-07 RTX 4090 pilot, train loss moved from 2.4801 to
0.0408 and held-out loss from 2.4368 to 0.4498 in 1,000 steps (126
seconds). Across all 35 command-disjoint held-out intents, the adapter moved
from 0% to 94.3% JSON/v2-envelope rate, 85.7% first-command accuracy, and
11.4% decoded structural exact match. The envelope metric checks the compact
top-level contract, not Rust render/re-lowering validity. These results show
that the target format is learnable while identifying exact arguments and
structure as the next quality bottleneck. Semantic generation uses a
192-token cap so the observed target-length tail is not scored as artificial
truncation.

New semantic runs default to the versioned `context-authoritative-v1` prompt.
It labels retrieved context as authoritative evidence for command names and
option spellings, labels the instruction separately, and tells the model to
derive operand count and values from that instruction. Use
`--prompt-contract legacy-user-v1` only to reproduce or evaluate an older
adapter; schema-v1 and schema-v2 checkpoints are permanently classified as
legacy prompts.

For the audited merged dataset, use the file converter. It preserves the same
row envelope but records every CST and semantic rejection ID instead of using
the curated corpus exemption ratchets:

```sh
cargo run -p shelliq-syntax --bin convert-semantic-dataset -- \
  training/artifacts/distributable-v1.jsonl \
  training/artifacts/distributable-semantic-v2.jsonl \
  training/artifacts/distributable-semantic-v2.manifest.json
```

The 2026-08-08 conversion produced 30,412 validated rows from 30,977 inputs
(98.2%), with 349 CST rejects, 216 semantic rejects, and all rejection IDs in
the manifest. A 4,000-row broad pilot followed by a 492-row curated finishing
stage reached 100% JSON/v2-envelope rate, 85.7% first-command accuracy, and
37.1% exact first-command-plus-flag sequence on the same 35 curated held-out
rows. Full decoded-document exact match was 5.7%. The versioned
`evaluation/curated-grounding-v1.json` audit marks only literal values that the
prompt does not specify; after normalizing those values, whole-document exact
match is 20.0%. Commands, flags, argument positions, redirects, and AST shape
remain exact.

Exported GGUF models can be evaluated through the same loopback HTTP boundary
planned for the alpha runtime:

```sh
uv run python scripts/evaluate_semantic_server.py \
  --semantic-dataset artifacts/distributable-semantic-v2.jsonl \
  --reference-report artifacts/semantic-distributable-curated-finish-v2.eval.json \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --prompt-contract legacy-user-v1 \
  --output artifacts/model.eval.json
```

The evaluator disables proxies and redirects, accepts only uncredentialed
loopback HTTP endpoints, caps response size at one MiB, and records latency for
every generation. Multiple sampled candidates additionally report an explicitly
label-aware oracle upper bound; it is diagnostic and never a deployable score.

Compare a candidate against the incumbent on the identical audited rows before promotion:

```sh
uv run python scripts/analyze_semantic_errors.py \
  --baseline artifacts/semantic-distributable-curated-finish-v2.eval.json \
  --candidate artifacts/model.eval.json \
  --grounding-audit evaluation/curated-grounding-v1.json \
  --output artifacts/model.error-analysis.json \
  --gate
```

The analyzer accepts checkpoint and served-GGUF reports, recomputes metrics from individual generations,
rejects mismatched benchmark IDs, and classifies wrong commands, missing/extra/reordered flags, and
operand/structure errors. The promotion gate requires 100% JSON and v2 envelopes, at least 31/35 first
commands, and strict improvements over the incumbents' 13/35 command-plus-flag and 7/35 grounded-document scores.

The focused finishing family targets those classified failures without training on release-benchmark
commands. Build its derived semantic artifact with the checked leakage and split gate:

```sh
uv run python scripts/build_targeted_semantic_dataset.py \
  --dataset corpus/targeted-finishing.jsonl \
  --holdout-report artifacts/semantic-authoritative-curated-v3.eval.json \
  --output artifacts/targeted-finishing-semantic-v1.jsonl \
  --manifest artifacts/targeted-finishing-semantic-v1.manifest.json \
  --converter ../target/release/convert-semantic-dataset
```

The v1 family has 28 rows across 14 command families: 24 train, two validation, and two test rows under
seed 2026. It teaches multi-flag retention, repeated option values, bundled short flags, `--` separation,
and operand placement. Its builder rejects shared holdout IDs or instructions and excludes every command
family present in the 35-row release benchmark.
The Q8_0 alpha export exactly reproduced the JAX checkpoint metrics and averaged
95 ms per request on the RTX 4090. Four candidates at temperature 0.2 provided
no oracle improvement, so targeted training—not sampling—is the next quality
lever.

`smoke_finetune.py --resume-checkpoint INPUT --checkpoint OUTPUT` performs a
format-checked finishing stage without overwriting the source checkpoint and
records the starting optimizer step in its report.

The v2 expansion adds 72 rows in `corpus/targeted-finishing-v2.jsonl`, making
100 focused rows total. Build both immutable source families into one derived
artifact by repeating `--dataset`:

```sh
uv run python scripts/build_targeted_semantic_dataset.py \
  --dataset corpus/targeted-finishing.jsonl \
  --dataset corpus/targeted-finishing-v2.jsonl \
  --holdout-report artifacts/semantic-authoritative-curated-v3.eval.json \
  --output artifacts/targeted-finishing-semantic-v2.jsonl \
  --manifest artifacts/targeted-finishing-semantic-v2.manifest.json \
  --converter ../target/release/convert-semantic-dataset
```

The checked result is 100/100 semantic conversions with zero rejects and a
96/2/2 train/validation/test split. A controlled weighted-rehearsal run resumes
the broad checkpoint and repeats only selected focused IDs three times:

```sh
uv run python scripts/smoke_finetune.py \
  --semantic-dataset artifacts/curated-semantic-v4.jsonl \
  --sequence-length 384 --train-examples 588 --eval-examples 36 \
  --batch-size 2 --steps 500 --learning-rate 5e-5 \
  --priority-source shelliq-curated \
  --rehearsal-record-prefix curated:targeted-finishing \
  --rehearsal-weight 3 --heldout-probe \
  --resume-checkpoint artifacts/semantic-authoritative-broad-v3/checkpoint \
  --checkpoint artifacts/semantic-weighted-rehearsal-v1/checkpoint \
  --report artifacts/semantic-weighted-rehearsal-v1.report.json
```

This selects 588 unique rows and 780 effective examples: all 96 focused
training rows receive three exposures, for 288 focused and 492 ordinary
exposures. Weighting happens only after command-grouped splitting and selection;
validation and test rows are never repeated. The report records unique and
effective counts, the ID prefix, and the multiplier.

The 2026-08-12 controlled run completed in 123 seconds. Relative to the same
broad checkpoint, training loss moved from 0.3007 to 0.0303 and command-grouped
held-out loss improved from 0.4295 to 0.3414. That healthy loss curve did not
translate to the frozen release gate: JSON/envelope stayed 35/35 and first
commands stayed 31/35, but exact command-plus-flag sequences fell to 8/35 and
grounded documents to 5/35, versus the incumbent's 13/35 and 7/35 gate floors.
On the separate retention suite, first commands stayed 20/20 and grounded
documents improved from 8/20 to 9/20, while exact flag sequences regressed from
14/20 to 12/20. The candidate was rejected and not exported.

Conclusion: expanding from 28 to 100 rows removes the extreme tiny-dataset
failure, but a 3x weight (37% of effective examples) still shifts flag behavior
too far toward the focused command families and does not transfer to the
release families. Do not treat this recipe as a default. The next controlled
candidate should reduce focused exposure and/or train fewer lower-rate steps,
with both frozen gates unchanged.

### Purpose and result of the targeted finishing experiment

The 28-row family is a diagnostic and curriculum seed, not a replacement for
the 654-row curated corpus. Its deterministic split leaves only 24 rows for
training. Those rows isolate the failure modes measured by the release gate
while using command families disjoint from the gate, so an improvement would
show transferable flag and operand learning rather than benchmark
memorization. The first experiment deliberately resumed the best adapter and
trained on only those 24 rows. This aggressive test asks two questions: does
the small family contain a learnable corrective signal, and how much rehearsal
of the broad corpus is required to retain existing behavior?

Only LoRA parameters are optimized; the Qwen base weights remain frozen. A
finishing run updates the existing LoRA adapter rather than stacking a second
adapter. Consequently, the observed catastrophic forgetting is interference
inside the adapter: the underlying pretrained base model is unchanged, but the
composed base-plus-adapter model behaves worse. A merged HF or GGUF export
folds that adapter delta into exported weights, so the same regression would
be visible in the runtime model even though the original base checkpoint is
still intact.

The pure-targeted sweep overfit quickly: targeted training loss fell while
held-out loss rose, and release JSON, command, flag, and grounded-document
scores all regressed. Mixing all 654 semantic rows with the targeted family
preserved more behavior and produced the best first-command score, but did not
beat the incumbent flag and grounded-document promotion thresholds. A lower
learning-rate continuation from the curated checkpoint preserved JSON,
command, and grounded scores but plateaued below the flag threshold. No
candidate was promoted or exported. The result is evidence that 24 focused
training rows are too small for a standalone finishing stage; the next useful
dataset experiment is a broader, more varied targeted family mixed with the
full corpus, followed by the same command-disjoint release gate.

### Holdout lifecycle and retention claims

A holdout is not knowledge the project promises never to teach. It is evidence
reserved from a particular model-selection cycle. Keep the 35-row release
benchmark untouched while choosing data, learning rate, and step count. Teach
the same underlying skills with different commands and paraphrases, then use
the holdout to measure transfer. If its exact rows or command families are
later added to training for a final fit, that benchmark is spent: retire it
from promotion decisions and create a new command-disjoint shadow holdout
before claiming generalization. Training on the test rows and continuing to
report their score would measure memorization, not the last five percent of
generalized capability.

The current adapter was not trained from only the 654 curated rows. Its broad
stage selected 4,000 semantic rows: 3,507 TLDR and 493 curated. The following
curated stage trained on 492 command-grouped training rows while retaining 34
for its loss probe, and the separate release evaluation covers 35 held-out
curated intents. During that curated stage, held-out loss improved from 0.4116
to 0.3265. The promoted checkpoint also retained 100% JSON and v2-envelope
rates, 31/35 first commands, 11/35 exact command-plus-flag sequences, and 7/35
grounded documents on the release set. The later low-dose mixed continuation
kept JSON, envelope, command, and grounded scores unchanged and improved flags
to 12/35, although that still failed the strict promotion gate.

This is useful retention evidence for the ShellIQ prompt and semantic-output
task, not proof that every pretrained capability is preserved. The current
suite does not broadly score paraphrase robustness, unrelated coding/chat
tasks, malformed or out-of-domain requests, or safety/abstention behavior.
Before calling a final adapter generally regression-safe, add a larger stable
retention suite spanning those categories and run it beside the focused
release benchmark after every finishing stage.

The first frozen positive-retention slice is
[`evaluation/semantic-retention-v1.jsonl`](evaluation/semantic-retention-v1.jsonl):
20 cases across common text processing, filesystem/process inspection, and
modern tooling. It lives outside `corpus/`, is never merged into training, and
uses command families disjoint from both the release benchmark and targeted
finishing family. Build its ignored semantic artifact and evaluate every row:

```sh
cargo run --release -p shelliq-syntax --bin convert-semantic-dataset -- \
  training/evaluation/semantic-retention-v1.jsonl \
  training/artifacts/semantic-retention-v1.jsonl \
  training/artifacts/semantic-retention-v1.manifest.json

uv run python scripts/evaluate_semantic_checkpoint.py \
  --semantic-dataset artifacts/semantic-retention-v1.jsonl \
  --selection all \
  --sequence-length 448 \
  --checkpoint artifacts/semantic-authoritative-curated-v3/checkpoint \
  --grounding-audit evaluation/semantic-retention-grounding-v1.json \
  --output artifacts/semantic-retention-incumbent-v1.eval.json
```

Freeze that incumbent report, evaluate a candidate with the same command, then
require no aggregate regression in JSON, envelope, first-command, ordered-flag,
or grounded-document metrics:

```sh
uv run python scripts/analyze_semantic_retention.py \
  --baseline artifacts/semantic-retention-incumbent-v1.eval.json \
  --candidate artifacts/semantic-retention-candidate.eval.json \
  --grounding-audit evaluation/semantic-retention-grounding-v1.json \
  --output artifacts/semantic-retention-candidate.analysis.json \
  --gate
```

The analyzer also lists per-example improvements and regressions for review and
refuses mismatched IDs, instructions, or expected targets. This v1 slice covers
positive semantic generation only. Abstention, unsafe requests, malformed
inputs, and unrelated coding/chat probes require a separate response contract
and remain explicit follow-up work rather than being assigned fake command
targets.

The 2026-08-12 incumbent baseline scored 20/20 JSON envelopes and first
commands, 14/20 exact ordered flag sequences, and 8/20 grounded documents.
Three flag misses are functionally plausible alternate spellings (`-nP`,
`--state=open`, and split rather than bundled grep flags), so the frozen report
is a regression floor while functional-equivalence scoring remains follow-up
work. Do not silently normalize those cases after seeing candidate output;
version the evaluator and re-baseline every candidate together if equivalence
rules change.

### Where reinforcement learning could fit

Large-model post-training commonly combines supervised examples with later
optimization against preferences or verifiable rewards. The closest analogue
here would start from a stable supervised LoRA, sample several semantic
documents for each request, execute or statically verify them in a sandbox,
and reward schema validity, command/flag correctness, grounding, and task
success. This could improve selection among behaviors the model already knows;
it does not replace missing command coverage or a retention suite.

Do not make reinforcement learning the next training stage. First expand the
focused supervised data, train it with broad rehearsal, and establish stable
task, retention, and safety graders. RL on 24 examples or on proxy metrics such
as JSON validity would invite the same narrow over-specialization—or reward
hacking—seen in the pure finishing sweep. Reconsider it once graders can score
functional command success and reject unsafe behavior without relying on the
reference answer's exact text.

Personal builders require canaries and always pass through `PrivateDataGate`.
They write the corpus, a canary-probe manifest, a deterministic 20-row scrubbed
audit sample, and a report containing counts and IDs but no rejected text:

```sh
uv run python scripts/build_dataset.py claude path/to/session.jsonl \
  --platform linux --canary-count 10 --canary-seed 2026 \
  --output private/claude.jsonl

uv run python scripts/build_dataset.py local-index --database path/to/shelliq.sqlite \
  --canary-count 10 --canary-seed 2026 \
  --output private/index.jsonl
```

Claude ingestion accepts individual JSONL files or directories, expands
directories recursively in deterministic path order, and keeps only `Bash`
calls with a linked non-error tool result. It fails closed if any transcript is
unreadable or malformed. `--allow-partial` permits valid sibling files to
continue, but prints every skipped path to stderr and records its hashed
transcript ID and sanitized reason in the scrub report. Privacy-dropped record
IDs and finding counts are always reported as well; rejected command text is
never copied into the report.

Local-index ingestion reads `examples`; `.zsh_history` command text has no
builder and never enters a training corpus.

The gate redacts configured home paths, usernames, hostnames, email addresses,
and private IPv4/IPv6 addresses. It drops the pair for AWS/GitHub/Slack tokens,
JWTs, private keys, URL credentials, secret environment assignments,
suspicious credential assignments, and long high-entropy hexadecimal/Base64
runs. Every detector class is canary-tested when the gate is constructed.
Synthetic post-scrub canaries are planted only in masked prompt context;
`run_canary_extraction_gate` fails if a trained model reproduces one.

## LoRA rank sweep

The first fixed-recipe result and plots are recorded in
[`experiments/lora-rank-fixed-recipe-v1.md`](experiments/lora-rank-fixed-recipe-v1.md).
Rank 8 performed best of 8/16/32/64 but did not pass the release gate. Treat the
result as a recipe-specific baseline, not proof that rank 8 is intrinsically
optimal; larger ranks had worse held-out loss and may require different
optimization schedules.

The controlled 0.5B capacity comparison uses ranks 16, 32, and 64 while holding
the base model, data split/order, seed, steps, and effective LoRA scale fixed.
Each rank trains broad semantic SFT first and resumes into the curated v7
finishing pass. It then runs the release-development and retention evaluations.
The runner never evaluates the pipeline suite or the final shadow release gate.

Inspect the exact commands and immutable input manifest without starting work:

```sh
uv run python scripts/run_rank_sweep.py
```

Start the GPU sweep, or resume stages already recorded by the exact same
manifest:

```sh
uv run python scripts/run_rank_sweep.py --execute
uv run python scripts/run_rank_sweep.py --execute --resume
```

Training requires CUDA; it does not silently fall back to CPU. Interactive
terminals show Rich loss/example progress with elapsed time and ETA, while
redirected runs keep sparse plain-text logs. The default rule `alpha = 2 * rank`
keeps `alpha / rank` constant. Override rank or schedule options only as a new
experiment with a new output directory.

Curriculum order can affect the result, so the rank sweep intentionally keeps
the same broad -> curated order. After choosing a rank, compare this baseline
with an interleaved/replay schedule that retains broad examples during the
finishing phase. Keep curriculum-weight sweeps separate so capacity, ordering,
and weighting are not changed in one experiment. Use additional seeds when a
winning margin is small, and evaluate only the final selected configuration on
`evaluation/semantic-shadow-release-v1.jsonl`.

The follow-up convergence study is deeper than the fixed 750-step optimization
screen: it permits 3,000 steps, evaluates held-out loss every 250 steps, uses
early stopping, and records pre-clipping gradient norms. It compares ranks
4/8/16 at learning rates `5e-5` and `1e-4`, with additional rank-8
`alpha / rank` checks at 0.5 and 2. Inspect commands without training, then run
or resume in an interactive terminal for Rich progress:

```sh
uv run --frozen python scripts/run_convergence_study.py
uv run --frozen python scripts/run_convergence_study.py --execute
uv run --frozen python scripts/run_convergence_study.py --execute --resume
```

This screen writes compact reports rather than adapter checkpoints and never
uses the release, retention, pipeline, or shadow evaluations. Retrain and save
only the recipe selected from its learning curves.

The full-corpus finalist phase trains rank 8 (`alpha=4`, peak LR `2e-4`) and
rank 16 (`alpha=16`, peak LR `5e-5`) on all 23,646 usable training-split rows.
Each run gets at most two deterministically reshuffled epochs (11,823 steps per
epoch), uses 500-step warmup followed by cosine decay to 10% of peak LR, and
cannot early-stop before completing one epoch. The corpus validation split is
checked every 1,000 steps, the corpus test split remains untouched, and only
the lowest-validation-loss checkpoint is retained:

```sh
uv run --frozen python scripts/run_full_corpus_finalists.py
uv run --frozen python scripts/run_full_corpus_finalists.py --execute
uv run --frozen python scripts/run_full_corpus_finalists.py --execute --resume
```

This remains an SFT optimization comparison. Its manifest prohibits release,
retention, pipeline, and shadow evaluation until one recipe is selected.

The reviewed chosen/rejected data contract and verifier lifecycle for the next
training phase are specified in [`PREFERENCE_DATA.md`](PREFERENCE_DATA.md).
`shelliq_training.preference_data` provides the strict initial loader; it does
not yet authorize a DPO run or treat model/teacher output as reviewed data.

## Checkpoints

Create the base model, load its pinned Hugging Face weights, inject the same
LoRA configuration, and create the optimizer before restoring:

```python
metadata = save_checkpoint(
    'checkpoints/step-00001000',
    model,
    optimizer,
    model_id='Qwen/Qwen2.5-Coder-0.5B-Instruct',
    corpus=Corpus.DISTRIBUTABLE,
    target_format=TargetFormat.RAW_SHELL,
    prompt_contract=PromptContract.LEGACY_USER_V1,
)
metadata = restore_checkpoint(
    'checkpoints/step-00001000',
    model,
    optimizer,
    model_id='Qwen/Qwen2.5-Coder-0.5B-Instruct',
    corpus=Corpus.DISTRIBUTABLE,
    target_format=TargetFormat.RAW_SHELL,
    prompt_contract=PromptContract.LEGACY_USER_V1,
)
```

Checkpoint directories are immutable. Restore refuses model ID, corpus, target
format, prompt contract, rank, alpha, and target-module mismatches. Schema-v1
checkpoints remain readable as raw-shell checkpoints; schema-v2 checkpoints
retain their target format; both are read as `legacy-user-v1`. Schema-v3 records
the prompt contract explicitly. Base weights are not duplicated.

## Evaluation

`evaluate_predictions` requires one prediction for every held-out record. It
can call an index-backed `options_known` callback, reported only as a floor, and
a container-backed `functional_equivalent` callback for examples explicitly
marked safe for execution, reported as the primary intent-success metric.

It also reports parse rate, exact match, command accuracy, case-sensitive flag
precision/recall/F1, option-argument accuracy, operand exact match, and latency
p50/p95. Compound shell syntax scores as unparsed and is never executed by this
module. Passing the canary manifest makes extraction a blocking gate.

## Export

`export_peft_adapter` writes only transposed LoRA A/B tensors in standard PEFT
names plus adapter and shelliq metadata. `export_merged_hf_model` applies each
LoRA delta and writes a llama.cpp-convertible Hugging Face directory. Outputs
are immutable. Distributable artifacts are marked publishable; personal
artifacts are marked non-publishable.

Export a checkpoint and optionally invoke the local llama.cpp toolchain:

```sh
uv run python scripts/export_checkpoint.py \
  --checkpoint checkpoints/step-00001000 \
  --base path/to/Qwen2.5-Coder-0.5B-Instruct \
  --output exports/adapter --kind adapter --corpus personal \
  --target-format raw-shell \
  --prompt-contract legacy-user-v1 \
  --gguf-output exports/adapter-f16.gguf --llama-cpp ../../llama.cpp

uv run python scripts/export_checkpoint.py \
  --checkpoint checkpoints/step-00001000 \
  --base path/to/Qwen2.5-Coder-0.5B-Instruct \
  --output exports/merged --kind merged --corpus distributable \
  --target-format semantic-document-v2-json \
  --prompt-contract context-authoritative-v1 \
  --gguf-output exports/model-q8.gguf --llama-cpp ../../llama.cpp \
  --quantization Q8_0
```

Standalone quantization rejects Q4 and accepts only Q6_K or Q8_0. Conversion
uses a temporary directory, checks GGUF magic, and publishes the final path only
after conversion and quantization succeed. `smoke_test_gguf` then loads the
artifact through `llama-cli`; that real load test remains required for every
trained release artifact.

The expensive integration smoke exports a zero-initialized adapter over the real
0.5B base through every format and actually loads the standalone GGUF:

```sh
uv run python scripts/smoke_export.py \
  --base path/to/Qwen2.5-Coder-0.5B-Instruct \
  --llama-cpp ../../llama.cpp --output /tmp/shelliq-export-smoke
```

On 2026-08-06 that smoke passed with the local llama.cpp checkout: the adapter
GGUF was 16.8 MiB, merged bfloat16 safetensors were 942 MiB, the Q8_0 GGUF was
507 MiB with valid GGUF v3 magic, and `llama-cli` loaded and generated from it.
This proves format/tool compatibility only; it says nothing about fine-tuned
quality because the smoke adapter is intentionally zero-initialized.

## Pipeline-contract curriculum

`corpus/pipeline-contracts.jsonl` contains 12 training-only examples that teach
producer record shape and consumer framing through varied `find`, `sort`,
`xargs`, `stat`, and `du` pipelines. The exact alpha failure wording is kept in
the separate eight-row `evaluation/pipeline-compatibility-v1.jsonl` suite.
`tests/test_pipeline_compatibility_dataset.py` requires disjoint instructions
and targets and a closed grounding audit.

The 2026-08-20 conservative experiment resumed
`semantic-authoritative-curated-v3`, rehearsed 600 broad rows, weighted the 12
pipeline rows 4x, and trained at `2e-6`. At 100 and 200 additional steps,
pipeline command/flag exact match improved from 1/8 to 3/8 and 4/8. The original
large-file sorting failure became functionally correct. Retention improved with
no regressions at 100 steps, but the frozen release benchmark plateaued at
12/35 command/flag matches and 7/35 grounded documents. Because the promotion
gate requires more than 13/35 and more than 7/35 respectively, neither
checkpoint is a release candidate. Do not export or serve them.

The pipeline rows remain part of every later full curated build. They are not a
discarded experiment or a standalone fix. `corpus/intent-fidelity.jsonl` adds a
balanced 48-row curriculum across 12 command families disjoint from both frozen
release and retention gates. Four contrastive examples per family teach exact
action/subcommand selection, complete multi-constraint flag sets, option-value
binding, operand preservation, separators, and redirections. The next candidate
should mix both curricula with the entire curated corpus at low dose and be
selected on release plus retention before consulting the pipeline suite once for
final confirmation.

The first broader-curriculum candidate resumed the incumbent for 100 steps at
`2e-6`, selected all 636 usable broad training rows, and repeated the 36
intent-fidelity rows in the training split once (672 effective examples). Train
loss moved 0.0672 to 0.0594 and command-grouped held-out loss 0.4058 to 0.3985.
The frozen release gate remained at 12/35 exact command/flag sequences and 7/35
grounded documents, so the candidate was rejected before pipeline evaluation.
Retention passed without regressions at 15/20 flag sequences and 9/20 grounded
documents, improving over the incumbent's 14/20 and 8/20. This supports keeping
the broad curriculum, but another low-rate continuation is unlikely to clear the
release plateau. Do not select further schedules against the pipeline suite.

A post-run audit against installed command help found that two new zstd rows used
unsupported long spellings for output and thread count. The checked-in rows now
use portable `-o` and `-T0`; `curated-semantic-v7` is the corrected 786-row
derived dataset for future runs. The rejected candidate used pre-audit v6 and is
not being re-scored or promoted.
