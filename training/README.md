# shelliq training

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
  --heldout-probe \
  --checkpoint artifacts/distributable-pilot-v1/checkpoint \
  --report artifacts/distributable-pilot-v1.report.json
```

This makes one pass over 4,000 deterministically selected, command-distinct
training examples and evaluates 256 examples from command-disjoint held-out
groups. The report pins the dataset and selected record IDs by SHA-256 and
records initial/final train and held-out losses plus train and held-out
generation probes. A useful first signal is falling held-out loss and a held-out
probe that becomes more command-like without merely copying the training probe.

This is deliberately a raw-shell baseline. It can tell us whether the data and
LoRA recipe carry useful NL-to-command signal, but it is not the final
Semantic-AST training target or a task-success evaluation.

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
)
metadata = restore_checkpoint(
    'checkpoints/step-00001000',
    model,
    optimizer,
    model_id='Qwen/Qwen2.5-Coder-0.5B-Instruct',
    corpus=Corpus.DISTRIBUTABLE,
)
```

Checkpoint directories are immutable. Restore refuses model ID, corpus, rank,
alpha, and target-module mismatches. Base weights are not duplicated.

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
  --gguf-output exports/adapter-f16.gguf --llama-cpp ../../llama.cpp

uv run python scripts/export_checkpoint.py \
  --checkpoint checkpoints/step-00001000 \
  --base path/to/Qwen2.5-Coder-0.5B-Instruct \
  --output exports/merged --kind merged --corpus distributable \
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
