# Custom semantic compiler v1

Status: plumbing and real-record overfit gates passed; held-out comparison not
started.

## Question

Can a small encoder-decoder Transformer trained from scratch learn ShellIQ's
narrow compilation task from an instruction plus authoritative local context to
compact `SemanticDocumentV2` JSON?

This is an architecture and learnability probe, not a release-model candidate.
The existing Qwen causal-LM results remain the comparison baseline. CodeT5 is the
next matched pretrained baseline if the custom compiler learns the task at all.

## Frozen boundaries

- Reuse the existing tokenizer so tokenizer and architecture novelty are not
  confounded.
- Initialize every neural weight from scratch. Do not load pretrained model
  weights.
- Use the existing versioned prompt contract and semantic corpus loader.
- Keep distributable and personal records structurally separate.
- Generate autoregressively; teacher-forced loss alone is insufficient evidence.
- Validate generated documents with the Rust `SemanticDocumentV2` validator.
- Do not consult release, pipeline, or shadow evaluation suites.
- Do not export, serve, or promote weights from this experiment.

## Session gates

1. A portable CPU test exercises forward loss, backpropagation, and greedy
   decoding with a tiny configuration.
2. A deterministic subset of real semantic records can be overfit on the GPU.
3. Generated outputs for that subset are measured for exact target match and
   Rust semantic round-trip validity.
4. The training command, selected record IDs, configuration, seed, metrics, and
   checkpoint hash are recorded reproducibly.

Passing these gates establishes plumbing and local learnability only. It does
not establish held-out generalization, useful model size, deployability, or an
advantage over pretrained CodeT5.

## Next decision

If the real-record overfit gate passes, train a preregistered held-out custom
baseline and compare it with a parameter-conscious CodeT5 baseline on the same
development data. If it fails, diagnose tokenization, decoding, and optimization
before allocating a full-corpus run.

## Session result

The implementation uses PyTorch 2.13, learned source and target positions, two
encoder and two decoder layers, shared token/output embeddings, and the existing
Qwen tokenizer only. No pretrained neural weights are loaded. The default probe
has 20,388,480 parameters (`d_model=128`, four heads, feed-forward size 512) and
separate 192-token source and target limits.

The portable CPU suite passed forward, backward, loss, and exact greedy-decoding
checks. The deterministic real-data run used seed 2026 and four command-disjoint
`curated-semantic-v7` rows:

- `curated:observability:linux:pidstat-command-io`
- `curated:modern-tools:linux:just-recipe-args`
- `curated:observability:linux:mpstat-all-cpus`
- `curated:modern-tools:darwin:brew-outdated`

On an RTX 4090, 2,000 full-batch steps took 26.19 seconds. Teacher-forced loss
fell from 11.9358 to 0.000205. Greedy decoding produced 4/4 exact compact targets;
4/4 passed Rust semantic round-trip validation and 4/4 passed conservative
reference verification. No generated command was executed.

The source dataset SHA-256 is
`d5bb17a6e13f95051a0a97971a07ddfb8a806fdf63dc3d47d160cbf47f62d7ca`.
The ignored 152 MB proof checkpoint remains under
`artifacts/custom-semantic-compiler-v1/` with SHA-256
`fecb3b1aca41e8e7c2ca17d9b557208bd5a16b07cc954b4cfc8cb6e7c2d6487a`.
It is a local research artifact, not a release candidate.

This passes the session's learnability gate. It provides no held-out evidence:
the next experiment must freeze a development split and matched custom/CodeT5
budgets before either model is trained on it.

Reproduce the proof from `training/` with:

```sh
UV_CACHE_DIR=/tmp/shelliq-training-uv-cache \
  uv run --frozen python scripts/train_semantic_compiler.py \
  --semantic-dataset artifacts/curated-semantic-v7.jsonl \
  --examples 4 --steps 2000 --batch-size 4 --device cuda \
  --checkpoint artifacts/custom-semantic-compiler-v1/checkpoint.pt \
  --report artifacts/custom-semantic-compiler-v1/report.json \
  --validator ../target/debug/validate-semantic-documents
```
