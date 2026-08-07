# shelliq training

Development-only JAX/Flax NNX code adapting
`Qwen/Qwen2.5-Coder-0.5B-Instruct`. Nothing in this directory ships with the
Rust binary.

## Current status

- Hand-written Qwen2 decoder: RMSNorm, RoPE, SwiGLU, grouped-query attention,
  causal masking, and tied embeddings.
- Hugging Face safetensors loader with the required linear-weight transpose.
- Padding-aware positions and attention for batched supervised fine-tuning.
- Masked next-token loss using `-100` for prompt and padding labels.
- Scaled LoRA on q/k/v/o and gate/up/down projections. LoRA parameters use a
  distinct NNX type, so the optimizer updates adapters only.
- Strict JSONL SFT records with per-record source, license, provenance, command,
  platform, and distributable/personal corpus tags.
- Command-grouped deterministic splits, optional source holdouts, real Qwen
  chat-template tokenization, and fixed-shape right-padded batches.
- Atomic Orbax checkpoints containing LoRA weights, optimizer moments, step,
  and compatibility metadata.
- GPU parity gate against Hugging Face PyTorch.

Source-specific dataset builders, the required private-data scrubber/canary
gate, adapter-only safetensors serialization, evaluation, and GGUF export are
not implemented yet.

## Setup and checks

From this directory:

```sh
uv sync --dev
uv run pytest -q
uv run ruff check .
uv run python scripts/check_parity.py
uv run python scripts/smoke_train.py
```

The parity script intentionally requests highest-precision float32 JAX matrix
multiplication. On NVIDIA, JAX's normal float32 policy may use TF32; that is a
reasonable training policy but would make an architecture comparison against
PyTorch CPU float32 fail for numerical rather than structural reasons.

## Measured GPU smoke test

On 2026-08-06, an RTX 4090 with 23,028 MiB ran the real 0.5B model using
rank-16 LoRA, bfloat16 base parameters, batch size 1, and sequence length 128:

- 494,032,768 base parameters and 8,798,208 LoRA parameters
- 14.79 seconds for compilation plus the first step
- 0.063 seconds for the second step
- 2.74 GiB peak JAX device memory

These numbers are a smoke-test measurement, not a capacity estimate. Longer
sequences, larger batches, optimizer choice, rematerialization, and allocator
state all change peak memory.

## Batch contract

`train_step` accepts a mapping containing three arrays shaped
`[batch, sequence]`:

- `input_ids`: tokenizer IDs
- `attention_mask`: 1 for real tokens and 0 for padding
- `labels`: desired token IDs, with `-100` wherever loss must be ignored

`labels` is shifted inside the loss. For instruction tuning, mask the prompt
portion with `-100` so only assistant tokens contribute to loss.

## Dataset contract

Each JSONL row uses schema version 1 and must carry enough metadata to audit its
origin:

```json
{"schema_version":1,"record_id":"tldr:find:largest","corpus":"distributable","source":"tldr-pages","license":"CC-BY-4.0","provenance":"pages/common/find.md@<commit>","command":"find","platform":"linux","instruction":"List the five largest files.","response":"find . -type f -printf '%s %p\\n' | sort -nr | head -5","context":"find: -type f selects regular files."}
```

`load_jsonl(..., corpus=...)`, `write_jsonl`, `split_records`,
`tokenize_record`, and `collate_sft` live in `shelliq_training.data`. They
deliberately fail on:

- missing provenance or license fields;
- distributable/personal pipeline mixing;
- duplicate record IDs;
- chat templates without a stable assistant boundary;
- records that exceed the configured sequence length.

Splits hash the command name, not the row, so paraphrases and the same command
from different sources stay together. Entire sources can be forced into the
test split with `heldout_sources`.

## Checkpoint contract

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

Checkpoint directories are immutable: saving refuses to overwrite an existing
path. Restore refuses model ID, corpus, rank, alpha, or target-module
mismatches. Base weights are intentionally not duplicated in each checkpoint.
