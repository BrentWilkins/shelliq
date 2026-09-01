# Experimental two-pass alpha

> **Model release status:** rejected by the preregistered P1B production gate.
> The model-free CLI alpha remains valid, but the 0.5B GGUF must not be shipped
> as an AI release. See
> [`p1b-runtime-v1-results.md`](../training/experiments/p1b-runtime-v1-results.md).

The context-free alpha is a Rust client plus a separately served local model. It never
executes a suggestion. Automatic mode searches the local documentation index, offers the
model at most six installed commands, retrieves at most sixteen cited flags for the selected
command, and checks the final command against the index.

## Current model artifact

- Base: `Qwen/Qwen2.5-Coder-0.5B-Instruct`, snapshot
  `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`
- Checkpoint: `training/artifacts/semantic-authoritative-curated-v3/checkpoint`, step 2500
- Prompt contract: `context-authoritative-v1`
- Quantization: Q8_0
- GGUF size: 506.5 MiB
- SHA-256: `3eaa6a1c982402ef6448c882738d4fe71ed851a08a9b349b4c13c2bd8714095f`

The GGUF is intentionally not committed. A published alpha must distribute it separately
with this digest and the checkpoint's distributable-corpus provenance.

## Local launch

```sh
cargo build --release --features model
./target/release/shelliq index scan

llama-server \
  --model /path/to/shelliq-semantic-alpha-q8_0.gguf \
  --host 127.0.0.1 --port 8080 --ctx-size 4096 --gpu-layers all --no-webui

./target/release/shelliq suggest \
  copy a tree while preserving timestamps ownership and symlinks
```

For a release candidate, run the sequential smoke gate against a dedicated index:

```sh
uv run python training/scripts/evaluate_two_pass_alpha.py \
  --shelliq target/release/shelliq \
  --index /path/to/index.sqlite \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --output /tmp/shelliq-two-pass-alpha.json
```

The gate requires the expected command to appear in every supported shortlist, each final
suggestion to pass local validation, and an unsupported video-transcoding task to abstain
before contacting the model.

Measured on 2026-08-12 with the artifact above, an RTX 4090, and a seven-command focused
index (`cp`, `grep`, `find`, `sort`, `tar`, `curl`, `ls`): shortlist recall 5/5,
end-to-end locally validated suggestions 5/5, unsupported abstention passed, mean latency
164 ms, and maximum latency 239 ms. This is a smoke gate, not a broad accuracy claim.

## Lightweight development installation

On the development machine, `~/.local/bin/shelliq` may point at the small
`shell/shelliq-alpha` launcher, which executes `target/release/shelliq` and pins the
alpha index independently of editor-specific XDG variables. A future
`cargo build --release --features model` atomically replaces that build output, so the next
invocation uses the new client without a separate installer. The user service in
`shell/systemd/shelliq-model.service` similarly points at
`training/artifacts/current.gguf`; update that symlink only after a model passes the alpha
gate, then restart the service.
