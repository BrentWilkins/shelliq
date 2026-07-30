# shelliq — a local, verifiable CLI assistant

## The complaint

"Is it `-r` or `-R`?" Every shell user loses minutes a week to this. The fix does not
require intelligence — it requires the man page on _this_ machine, indexed and queryable.

The original framing was "pretrain an LLM on all my man pages with JAX." Measurement
reshaped that, and the reshaping is the interesting part.

## What the numbers said

- The man corpus here is ~5,200 pages, **~8–25M tokens**. Chinchilla-optimal for that is a
  ~1M-parameter model. Anything pretrained from scratch on it emits fluent man-page prose
  and _invents flags_ — the exact failure being solved for.
- Flags extract **deterministically**. The P0 parser reproduces `curl --help all` exactly —
  258 long flags, zero difference in either direction — and `ls --help` exactly at 44.
  (The figures quoted earlier in planning, `curl` 396 and `rsync` 382, were an overcount:
  that heuristic also matched flags *mentioned inside description bodies*, such as curl's
  "--alt-svc can be used several times". The measured numbers are in
  `crates/harvest/tests/real_pages.rs`.)
- Emitting command _syntax_ is easy (small formal grammar). Understanding **English** is
  the hard half, and it cannot come from 8–25M tokens of terse man prose. So: fine-tune an
  existing small model rather than start cold.
- **`ollama`, `kubectl`, `cargo`, `uv`, `pnpm`, `ruff`, `just` have no man pages at all.**
  The harvester must also crawl `--help`.
- Shell history is full of **bundled short flags** — `-fsSL`, `-sirn`, `-xzf`, `-LsSf`. The
  verifier must decompose bundles, because a wrong case hides invisibly inside `-sirn`.

**Architecture — facts and fluency separated:**

| Concern                             | Owner                               | Why                               |
| ----------------------------------- | ----------------------------------- | --------------------------------- |
| Flag facts, case, args, subcommands | SQLite index, generated per-machine | Deterministic, exact, citable     |
| English → command shape             | Fine-tuned Qwen2.5-Coder-0.5B       | Needs pretrained language ability |
| Correctness of the answer           | Verifier, index-backed              | Catches model hallucination       |

The trust story is not "it's local" — a local 0.5B alone is _less_ reliable than cloud
Copilot. It is **the verifier**: every flag checked against the man page on this machine,
with a citation. Local is the privacy story, not the accuracy story.

**Why this beats what exists:** `tldr` is hand-written markdown, no model, answers only
pre-written questions. Warp AI and GitHub Copilot CLI are cloud calls to frontier models
with no man-page grounding — they know a generic GNU world and will hand you GNU `sed -i`
on macOS. None read the man pages on _your_ machine at _your_ installed versions.

## Footprint: what runs where

The 4090 is a **build-time** dependency, not a runtime one. Training happens once, here,
and ships a GGUF. No user ever needs a GPU.

| Tier              | Needs                        | Size         | Covers                                    |
| ----------------- | ---------------------------- | ------------ | ----------------------------------------- |
| **0 — no model**  | nothing but the binary       | ~8MB + index | `explain`, fuzzy flag search, Tab, verify |
| **1 — CPU model** | 0.5B GGUF, fetched on demand | +~500MB      | English → command                         |
| **2 — GPU**       | any CUDA/Metal device        | same file    | same, faster                              |

**Tier 0 is the default and handles the original complaint with zero inference.** It runs
on a Raspberry Pi. The model is not bundled, not required, and not linked into the default
build — it sits behind a Cargo feature and is downloaded only if asked for.

Measured, not assumed: `qwen2.5-coder:1.5b` with `num_gpu: 0` — **51 tok/s, 1.82s
end-to-end, correct answer** (`find /var/log -type f -mtime -2`). That is the _1.5B_ on
CPU. The 0.5B is 3× smaller, so Tier 1 has comfortable margin under 1s on a laptop.

**WebLLM is the sanity check.** It runs Qwen2.5-0.5B _in a browser tab_ over WebGPU/WASM.
Anything that fits a browser sandbox fits a native CPU process easily. Two consequences:
0.5B is already the floor worth targeting — there is no need for a smaller tier — and a
future browser demo is a real option, not a stretch.

### Laptops, iGPUs, and the M1 — inherit acceleration, don't build it

**shelliq embeds no inference engine.** Tier 1 speaks HTTP to whatever the user already
runs — `ollama` or `llama-server`. This is the decision that makes laptops a non-problem,
because those projects have already solved backend portability:

| Machine               | Backend               | Who provides it              |
| --------------------- | --------------------- | ---------------------------- |
| M1/M2/M3 MacBook      | Metal, unified memory | ollama ships it              |
| Intel Iris / Arc iGPU | Vulkan or SYCL        | ollama / llama.cpp ship both |
| AMD Radeon iGPU       | Vulkan or ROCm        | same                         |
| Anything else         | CPU                   | always available             |

Confirmed locally: `llama.cpp` carries `ggml-metal`, `ggml-vulkan`, and `ggml-sycl` as
**dynamically-loadable** backends alongside `ggml-cpu`, and ollama 0.32.1 is installed
here. Porting an iGPU backend is work someone else has already done and continues to
maintain; duplicating it inside shelliq would mean a Vulkan dependency, driver-variance
bugs across three vendors, and a far larger binary — in exchange for tokens on a model
small enough not to need them.

**On the iGPU specifically:** worth trying, not worth depending on. For a 0.5B answering in
~20 tokens, the wall-clock is dominated by process start and prompt ingest, not by
generation. An iGPU may roughly halve an already-sub-second number. The honest framing is
that it is a free win when the user's ollama picks it up automatically, and never a
requirement. That is exactly what the HTTP boundary buys.

**The M1 MacBook Pro is the best target in the fleet, not the hardest.** Unified memory
means the 500MB Q6_K model needs no host↔device copy, and Metal is mature. Tier 0 is even
better off there: `mandoc` ships with macOS, so the _preferred_ man parser is available on
the Mac while Linux needs an `apt install`. The Rust side is clean for `aarch64-apple-
darwin` — `rusqlite` (bundled C), `clap`, `shlex`, and `nucleo-matcher` are all portable,
and the default build links no CUDA. Real M1 numbers get measured in P5 rather than
guessed at here.

Embedding `llama-cpp-2` for a zero-dependency single-binary install stays available as a
later opt-in Cargo feature. It is not how v1 ships.

## Approach

```text
harvest (Rust, per machine, incremental)
  man pages via mandoc -T markdown  ──┐
  --help crawl, recursive subcommands ├──> index.sqlite (+FTS5)
  zsh/bash builtins                  ─┘
                                             │
                                             v
                                     deterministic router
                                     ╱          │          ╲
                            exact_flag    fuzzy_search   generate
                            (exhaustive)  (top-k)        (must verify)
                                 │             │              │
                                 └─────────────┴──────► verifier ──► cited answer
```

Repo: `/home/brent/code/shelliq/` — name free on crates.io, PyPI, and npm.
Binary `shelliq`, aliased to `q` locally. It is read far more than typed; the primary UX is
keybindings.

```text
crates/shelliq/       clap CLI entrypoint
crates/harvest/       man parser, --help crawler
crates/index/         schema, FTS5, fuzzy ranking, RRF, refresh
crates/verify/        tokenizer, bundle splitter, flag checker
shell/                shelliq.zsh, shelliq.bash
training/             Python+JAX: nnx Qwen, HF loader, data gen, train (dev-only)
```

**Language split: Rust ships, Python trains.** A Rust static binary starts in ~2ms, so
index lookups are a plain process invocation — **no custom daemon**, which a Python runtime
would have forced (30–80ms interpreter start blows the <10ms Tab budget). macOS then needs
no Python at all. The interface between halves is the GGUF file and the SQLite schema.

Verified present: rustc 1.95, cargo 1.95, and crates `rusqlite` (86M dl), `shlex` (683M),
`clap` (1B), `ureq` (167M), `nucleo-matcher` (3M), `llama-cpp-2` (911k).

### 1. Index

```sql
commands(id, name, platform, section, version, synopsis, description,
         source_path, source_hash, harvested_at)
subcommands(id, command_id, path, summary)      -- 'ollama list', 'git remote add'
flags(id, command_id, subcommand_id, short, long, arg_type, arg_required,
      description, group, source_line,
      rank_personal,    -- local shell history frequency
      rank_tldr)        -- appears in tldr examples
examples(id, command_id, text, description, source)
commands_fts, examples_fts, flags_fts           -- FTS5 over descriptions
```

`flags.short` preserves case exactly; `-r` and `-R` are distinct rows. That property alone
answers the original complaint with zero inference.

### 2. Harvester

**Man pages** — parse source, not rendered text, where possible. `mandoc -T markdown`
handles both `man` and `mdoc` macros, so BSD/macOS pages work through the same path.
mandoc is **not installed here** (`apt install mandoc`; macOS ships it), so P0 ships the
validated rendered-text indentation heuristic and treats mandoc as an upgrade. Both paths
are validated against the same fixtures.

**`--help` crawler** — covers the no-man-page tools. Run `TOOL --help`, parse the near
universal `-x, --xxx  description` shape that clap, cobra, and argparse all emit, extract
subcommands, recurse to depth 3.

Safety, plainly: this executes arbitrary binaries. Mitigations — 5s timeout, only ever pass
`--help`/`help`, skip setuid, and gate the first full `PATH` sweep behind a user-reviewed
`~/.config/shelliq/allowlist.toml`.

**Staleness via fingerprint, not just hash.** Every row carries `source_hash` +
`harvested_at`, and the index as a whole carries a **pipeline fingerprint** (parser version

- options). A parser change therefore invalidates the index the same way a package upgrade
  does — a `source_hash` alone would silently serve rows built by old, buggier code.
  `shelliq index --refresh` re-parses only what changed: full build ~2–5 min, refresh
  seconds. Auto-trigger via `DPkg::Post-Invoke` on Ubuntu, `launchd` on macOS. Because the
  index is generated locally and never shipped, **macOS needs no data transfer and cannot
  drift**.

### 3. Router — the speed win

Borrowed from `~/code/author-corpus-rag`, which solved the same shape of problem: a vector
search was asked how many articles an author wrote and confidently answered from top-k.
Exact questions must inspect the complete catalog, not infer totals from semantic matches.

shelliq routes **deterministically, before doing anything expensive**, and each route
carries an explicit coverage contract:

| Route          | Engine              | Coverage                 | Latency | Trigger                           |
| -------------- | ------------------- | ------------------------ | ------- | --------------------------------- |
| `exact_flag`   | SQLite lookup       | **Exhaustive**           | <5ms    | Input parses as a command line    |
| `fuzzy_search` | FTS5 + nucleo, RRF  | Non-exhaustive top-k     | <10ms   | Partial flag or description words |
| `generate`     | 0.5B, then verified | Unverified until checked | ~500ms  | Prose that isn't a command        |

Every decision records `rule_id`, `rationale`, and `matched_signals` — inspectable, with
**no invented confidence score**. A labeled `RoutingCase` set measures router accuracy the
same way that repo does.

The speed argument: the overwhelming majority of real queries — "what's the recursive flag
for grep" — never reach a model, never touch an embedding, and answer from SQLite in
single-digit milliseconds.

**Fusion without embeddings.** author-corpus-rag needs dense retrieval because prose is
semantic. Flag descriptions are short and keyword-dense, so shelliq fuses **FTS5 (BM25) and
nucleo fuzzy** by **reciprocal-rank fusion** — combining _ranks_, never raw scores, since
BM25 and fuzzy scores are not a common scale. This keeps Tier 0 at **zero ML dependencies**:
no embedding model, no vector index, no ONNX runtime. That is the single biggest leanness
decision in the design.

### 4. Display: making 258 flags navigable

Mostly, you don't show them. Four paths, one of which ever approaches the full set:

1. **`C-x C-h` explain a typed line** — bounded by what was written, 2–4 flags. Common case.
2. **Fuzzy over flag names** — `curl --ret` → 3 results.
3. **Fuzzy over flag _descriptions_** — search meaning, not prefix, so you need not know
   the name you forgot. See the measured limit below.
4. **Bare `curl -<TAB>`** — rank by `rank_personal`, cap ~15, footer `…243 more`.

```
$ curl -<TAB>
  -s  --silent   ★     -o  --output   ★     -f  --fail  ★
  -L  --location ★     -H  --header   ★
  ─ your 15 most-used · …243 more, keep typing ─
```

`flags.group` preserves man page section grouping. Descriptions truncate to terminal width.

**Measured limitation — the showcase example does not work, and it is instructive.**

The original claim here was that typing "follow redirect" would find `-L, --location`. It
does not. curl's man page describes that flag as:

> (HTTP) If the server reports that the requested page has **moved to a different
> location** (indicated with a Location: header and a 3XX response code), this option makes
> curl redo the request on the new place.

The words "follow" and "redirect" never appear. BM25 over descriptions therefore returns
`--max-redirs`, `--post301`, and `--location-trusted`, but not `-L` itself. This is not a
ranking bug to be tuned away; man pages describe **mechanism**, and users search by
**task**.

Three remedies, cheapest first:

1. **Index tldr examples into `examples_fts`.** tldr's wording is task-oriented ("follow
   redirects") where man's is mechanism-oriented. This promotes the `examples` table from a
   P3 dataset input to a **Tier 0 index input** — it is the vocabulary bridge, and it needs
   no model.
2. **Cross-reference expansion.** curl descriptions cite each other: `--location-trusted`
   reads "Like -L, --location, but…", and it *did* rank in the top 5. Extracting flag
   spellings mentioned inside matched descriptions and boosting those targets would surface
   `-L` from its neighbours. Pure SQL, no new dependency.
3. **The model.** Bridging mechanism-language to task-language is exactly what a language
   model is for. This is real evidence for the P1 decision point rather than an assumption.

Remedies 1 and 2 land in P1. Until then, `shelliq search` states its coverage honestly —
"10 of 258, matched on description" — rather than implying it found the best answer.

### 5. Verifier — the crown jewel

1. Tokenize into pipeline segments via `shlex`, each `(command, subcommand, flags, args)`.
2. **Decompose bundled shorts**: `-sirn` → `-s -i -r -n`, `-fsSL` → `-f -s -S -L`.
3. Look up each command in the index.
4. Per flag, **case-sensitive** exact match:
   - hit → OK, attach citation
   - miss but case-insensitive hit → **auto-correct and say so**: "`-R` isn't valid for
     `grep`; did you mean `-r`?"
   - miss entirely → mark unverified; never silently pass
5. Check arg-taking flags got args.
6. Unknown command → say so rather than trusting the model.

Emits the command annotated with per-flag citations (`grep(1):142`).

### 6. Training (JAX — the learning half, Python, dev-only)

Base: **Qwen2.5-Coder-0.5B-Instruct**, reimplemented in `flax.nnx`: RMSNorm, RoPE, SwiGLU
MLP, GQA (24 layers, hidden 896, 14 Q heads, 2 KV heads, head_dim 64, tied embeddings). A
real step up from the course MiniGPT.

Hand-write the HF→nnx weight loader. Two trip hazards: HF stores linears `[out, in]` while
nnx `Linear` wants `[in, out]` (transpose), and Qwen2 has attention biases on q/k/v but
_not_ o.

**Hard gate before training:** same prompt through HF `transformers` and the nnx port,
logits within 1e-3. Most ports fail silently; this catches it.

Full fine-tune fits easily — 0.5B bf16 weights+grads plus fp32 Adam moments ≈ 6GB of 24GB.
Full FT first; LoRA (r=16, q/k/v/o/gate/up/down) as the second exercise, since
`convert_lora_to_gguf.py` is already in the local llama.cpp checkout. Reuse `grain` and
`orbax` from the course; `optax.adamw`, cosine schedule, warmup.

**Dataset** (~30–50k pairs), each formatted with a `<context>` block of retrieved index
entries so training matches retrieval-augmented inference, tagged `# platform: linux|darwin`:

1. **tldr-pages** (CC-BY-4.0) — ~4k commands × ~6 examples ≈ 25k clean pairs. The bulk.
2. **Claude Code transcripts** — measured: **2,231 description+command pairs, 2,166
   unique, 580 distinct flag tokens** across 50 local session logs. Every `Bash` tool call
   carries a human-quality `description`, so these are _pre-paired_ NL→command data with no
   annotation step. Better still, the transcripts record outcomes — **5,274 clean vs 205
   errored** tool results — so pairs can be filtered to commands that **actually ran and
   worked**. That correctness signal is something raw shell history cannot offer.
   Caveats, honestly: the distribution skews to code exploration (`grep -iE`, `head -n`,
   `sed -n`) rather than general sysadmin, and descriptions are terse imperatives ("Check
   QD subcommand help") rather than the questions a user would actually type. So this is a
   _high-quality supplement needing rephrasing_, not a drop-in replacement for tldr. It
   also grows with every session, for free.
3. **NL2Bash** — ~9k pairs, older and noisier; filter hard.
4. **Synthetic from the index** — local `qwen3-coder:30b` generates NL queries + commands
   from real harvested flags. **Every pair filtered through the verifier**; only clean
   pairs kept. A self-cleaning dataset.

Local `.zsh_history` stays out of the training set and is used **only** for
`rank_personal` completion ordering — that is what "personal" should mean, and it keeps the
dataset free of a user's private command lines. Any transcript-derived data is scrubbed for
secrets and paths before it is used, and never leaves the machine.

**Export:** nnx → safetensors in HF layout → existing `convert_hf_to_gguf.py` →
`llama-quantize`. Use **Q6_K or Q8_0**, not Q4 — 0.5B degrades noticeably at Q4, and Q8 is
still only ~500MB.

### 7. Shell integration

Designed not to fight oh-my-zsh, zsh-autosuggestions, or zsh-syntax-highlighting:

- `C-x C-n` — ZLE widget: buffer treated as English, replaced with the verified command.
  Never touches Tab.
- `C-x C-h` — explain/fuzzy-search flags for the current buffer. Index-only, no model.
- Tab, safely — a **fallback** completer, never a replacement:
  ```zsh
  zstyle ':completion:*' completer _complete _shelliq _approximate
  ```
  `_shelliq` runs only when `_complete` produces nothing, so `_git`, `_docker`, and every
  tool-provided completer keep priority. Inherits the existing `menu select` dropdown UI.

**bash:** `bind -x '"\C-x\C-n": _shelliq_widget'`, plus `complete -D`, which likewise fires
only where nothing else is registered.

Note: `~/.zshrc` calls `compinit` twice (lines 128 and 133). Worth collapsing — will
confirm before touching that file.

## Reliability invariants

Adapted from author-corpus-rag, which states its guarantees as a numbered contract rather
than leaving them implicit. These are testable claims, not aspirations.

1. Flag facts — existence, case, arity — come **only** from the index, never from the model.
2. A flag is matched **case-sensitively**. `-r` and `-R` are never conflated; a
   case-insensitive hit is reported as a _correction_, never silently accepted.
3. Any generated command passes the verifier before display. Unverifiable segments are
   labeled unverified; they are never presented as correct.
4. An unknown command is reported as unknown. It is never assumed to be GNU.
5. Exhaustive answers ("all flags for `curl`") come from SQL. Fuzzy results always report
   non-exhaustive coverage.
6. Fuzzy and BM25 scores are never blended into one number, and never shown as confidence.
   Fusion combines ranks.
7. Every displayed flag carries a citation to its source page and line.
8. The index records platform. A `darwin` row is never served on `linux` or vice versa.
9. The harvester passes only `--help`/`help`, never other arguments, and never runs setuid
   binaries.
10. Tier 0 has no ML dependency at runtime — no model, no embeddings, no network.
11. The model is never required. Every Tier 0 path works with the model absent.
    11a. shelliq embeds no inference engine and no GPU backend. Acceleration is inherited from
    the user's ollama or llama-server over HTTP, so Metal, Vulkan, SYCL, and CUDA are
    supported without shelliq containing a line of backend code.
12. Routing decisions are inspectable: `rule_id`, `rationale`, `matched_signals`, no score.
13. A stale index is reported as stale, never served silently as current.
14. Local shell history informs ranking only, never the training set. The ranking it
    produces is a frequency table of command and flag tokens — never command text.
15. No data derived from this machine leaves it. There is no telemetry, no upload, and no
    default that sends a command line anywhere.
16. Any locally-derived corpus is scrubbed before it can enter a training set, and the
    scrubber fails closed: a line it cannot confidently clean is dropped, not repaired.
    The dataset build aborts rather than emitting an unscrubbed pair.

## Phases

Each ends with something usable.

- **P0** — Rust harvester + index + `shelliq explain grep -r`. Solves the original
  complaint with no model at all. **This is Tier 0 and it is the whole point.**
- **P1** — verifier + fuzzy display + shell widgets, using ollama `qwen2.5-coder:1.5b` as a
  placeholder generator over HTTP. Full UX, no custom training. **Decision point: if this
  is good enough, P2–P4 become a JAX learning exercise rather than a requirement.**
- **P2** — nnx Qwen port + logit-parity gate.
- **P3** — dataset build + fine-tune + eval.
- **P4** — GGUF export, swap into `llama-server`, benchmark vs the placeholder.
- **P5** — macOS: run harvester there, verify mdoc parsing, platform-tagged divergence.

## Acceptance criteria

Each phase has an exit gate. A phase is not done until every gate passes, and a gate is a
measurement, not an opinion. Measured values below are from this machine.

### P0 — index and verifier, no model — **PASSED**

| Gate                                              | Target      | Measured             |
| ------------------------------------------------- | ----------- | -------------------- |
| Parser matches `curl --help all`                   | exact       | 258 = 258, 0 diff [done] |
| Parser matches `ls --help`                         | exact       | 44 = 44, 0 diff [done]   |
| `-r` and `-R` distinct rows with citations         | required    | [done]                   |
| Wrong case inside a bundle detected                | `-sirN`→`-n`| [done]                   |
| Arg-taking flag not split as a bundle              | `-A5`       | `-A` + `5` [done]        |
| Every man section indexed, not just the default    | required    | `signal` 2 and 7 [done]  |
| Bare `kill` resolves to the command, not the syscall | section 1 | [done]                   |
| Fabricated flag never reported as valid            | required    | [done]                   |
| Unindexed command reported, not assumed GNU        | required    | [done]                   |
| `explain` exit code non-zero when problems found   | required    | [done]                   |
| Release binary size                                | < 10 MiB    | **1.69 MiB** [done]      |
| Default build links no inference library           | required    | libc/libm/libgcc only [done] |
| `explain` latency, cold process                    | < 10 ms     | **2.0 ms** [done]        |
| `search` latency, cold process                     | < 10 ms     | **2.35 ms** [done]       |
| Test suite                                         | all green   | 39 passing [done]        |

The latency figures include process start, opening SQLite, and the query. This is the
measurement that retires the daemon question: a Rust binary invoked per keystroke-batch is
comfortably inside the Tab budget, so there is nothing resident to build, supervise, or
explain to a user.

### P1 — full UX on a placeholder model

- Description search finds `-L, --location` from "follow redirect" — currently **failing**,
  see section 4. Fixed by tldr ingestion and cross-reference expansion.
- Full `PATH` sweep completes behind the allowlist gate with no binary run outside it.
- `ollama`, `kubectl`, `cargo`, `uv` indexed via the `--help` crawler, subcommands included.
- End to end: "get me the 5 most recently installed models in ollama" produces a command
  whose every segment verifies.
- Router: labelled cases, `exact_flag` never fires on prose, `generate` never fires on a
  well-formed command line.
- No-fight check: with `shelliq.zsh` sourced, `git che<TAB>`, `docker run --rm<TAB>`, and
  autosuggestions behave exactly as before.
- Model path < 2 s warm on CPU, < 500 ms on GPU.
- **Decision point.** If P1 is good enough, P2–P4 are a JAX learning exercise, not a
  requirement. This is a legitimate stopping place, and saying so now prevents sunk cost
  from making the decision later.

### P2 — nnx port

- Same prompt through HF `transformers` and the nnx port agree within **1e-3** on logits.
  Hard gate; nothing downstream starts until it passes.

### P3 — fine-tune

- Held-out NL→command set: fine-tuned 0.5B beats base 0.5B on exact match and on
  verifier-clean rate.
- Fine-tuned 0.5B within reach of `qwen2.5-coder:1.5b` on verifier-clean rate.
- Every synthetic training pair passed the verifier before entering the set.
- **Scrubber gate, blocking.** The scrubber's test suite includes a planted secret of every
  class in the privacy table and catches all of them. An audited random sample of the
  scrubbed set shows zero surviving secrets. Training does not start until both pass.
- No `.zsh_history` command text appears anywhere in the dataset — only its derived
  frequency counts, and only in the index.

### P4 — ship the model

- GGUF loads in `llama-server` and answers correctly at Q6_K or Q8_0.
- Benchmarked against the P1 placeholder on the same held-out set, reusing harness patterns
  from `~/code/local-llm/benchmarks/`.

### P5 — macOS

- Harvester runs on the M1 and parses mdoc pages; `bsd` and `gnu` divergences show as
  distinct platform-tagged rows.
- Tier 0 works with no Python present.
- Real M1 latency measured rather than extrapolated.

## Verification

- **Parser** — [done] done. Ground truth is each tool's own `--help`, not a hand-counted
  figure. `curl` matches `curl --help all` exactly (258, zero difference either way); `ls`
  matches exactly at 44. `grep --colour` and `tar --show-snapshot-field-ranges` appear only
  in `--help` and are absent from the man page — a genuine divergence, asserted as such.
- **Multi-section** — [done] done. `man -w` returns only the default section, which loses
  content: `signal` defaults to the 84-line section 2 syscall page while the 378-line
  section 7 overview is the one usually wanted, and `time` has pages in 1, 2, and 7.
  `harvest_all` indexes every section; `SECTION_PREFERENCE` makes bare `kill` resolve to
  the section 1 command rather than the section 2 syscall.
- **Bundle splitter** — [done] done. `-sirn` and `-xzf` decompose; `grep -sirN` flags `-N` and
  suggests `-n`; `grep -A5` yields `-A` with argument `5` rather than the flag `-5`,
  because the index says `-A` takes an argument.
- **Router** — labeled cases; `exact_flag` must never fire on prose, `generate` must never
  fire on a well-formed command line.
- **Port parity** — nnx vs HF logits within 1e-3. Hard gate before P3.
- **Verifier units** — `grep -R` → suggests `-r`; fabricated `--frobnicate` → unverified;
  `tar -czf` → clean.
- **End-to-end** — "get me the 5 most recently installed models in ollama" produces a
  verified command. `ollama` has no man page, so this exercises the `--help` crawler.
- **Model eval** — held-out NL→command set. Exact match, verifier-clean rate, and
  containerized functional equivalence on a safe subset. Compare base 0.5B / fine-tuned
  0.5B / `qwen2.5-coder:1.5b` / `qwen3-coder:30b`, reusing harness patterns in
  `~/code/local-llm/benchmarks/`.
- **Latency** — index lookups <10ms (Tab path); model path <500ms warm on GPU, <2s on CPU.
- **Leanness** — Tier 0 binary under 10MB; assert the default build links no inference
  library and opens no socket.
- **No-fight check** — with `shelliq.zsh` sourced, confirm `git che<TAB>`, `docker run
--rm<TAB>`, and autosuggestions behave exactly as before.

## Decisions taken during P0

**No TUI framework — not Ratatui, not Bubbletea.** Both are full-screen frameworks that
claim the alternate screen and run an event loop. shelliq's output is *inline*: it prints
beneath the prompt while zsh's line editor still owns the terminal, and the Tab dropdown is
rendered by zsh's own `menu select`. Taking over the screen would fight the line editor,
which is precisely the constraint the shell integration is built around. Bubbletea would
additionally reintroduce Go, undoing the single-runtime decision.

What is actually needed is colour, and that is `anstyle` — already in clap's dependency
tree, so it costs nothing. Syntax highlighting comes free from a source a generic
highlighter does not have: the verifier has already parsed the line into command,
subcommand, flags, and arguments, so colouring by *verification status* (green verified,
yellow wrong-case, red unknown) is both cheaper and more informative than lexical
highlighting. zsh-syntax-highlighting continues to colour the buffer itself.

Ratatui earns its place only in an optional full-screen `shelliq browse curl` explorer.
That is a later feature behind a Cargo feature flag, never a dependency of the hot path.

**`info` — yes, but narrowly, and for examples only.** Checked: 62 texinfo manuals are
installed here against ~5,200 man pages, and coverage is GNU-only, so it is absent for
`ollama`, `kubectl`, and `cargo`, and largely absent on macOS. It adds nothing to flag
facts, which the man parser already extracts exactly. Its value is the worked examples and
explanatory prose that GNU keeps in info rather than man — which feeds `examples_fts`, and
therefore feeds the vocabulary-bridge problem in section 4 above. Worth harvesting on Linux
as a supplement to tldr; not a P0 index input, and never a flag source.

**Training environment is uv with Python 3.14.** `training/` is a uv project pinned to
3.14, dependencies added with `uv add` so constraints come from real resolution rather than
guessed floors. Resolves clean: jax 0.11.0, flax 0.12.8, optax 0.2.8, orbax-checkpoint
0.12.1, grain 0.2.18, transformers 5.14.1, datasets 5.0.1, numpy 2.5.1. `uv sync` is
deferred to P2 because `jax[cuda12]` pulls ~3GB of wheels that P0 has no use for. Ruff is
configured for single quotes to match the house style.

## Privacy of local data

Two rules, and the first one does most of the work.

**Local-only by default.** Nothing derived from this machine leaves it. The index is built
here and never shipped. Shell history is read here and never shipped. There is no
telemetry. The model, when it exists, is trained here and only the *weights* are
distributable — and only after the gate below.

**`.zsh_history` never becomes training text.** It is used for exactly one thing: counting
how often a command and flag token appear, to order completions. A frequency table is not
command text, so the private content of a command line never enters the pipeline at all.
This is a stronger guarantee than scrubbing, because there is nothing to scrub.

**Everything else is scrubbed before training, and the scrubber fails closed.** Claude Code
transcripts are the case that needs this: they contain real project contents, paths, and
whatever was pasted into a terminal. Scrubbing runs as a mandatory stage, not an optional
one, and the dataset build aborts rather than emitting an unscrubbed pair.

What the scrubber removes, in order:

| Class                | Handling                                              |
| -------------------- | ----------------------------------------------------- |
| Home and user paths  | `/home/<user>/…` → `~/…`, usernames → `<user>`         |
| Hostnames, private IPs | → `<host>`, `<ip>`                                   |
| Email addresses      | → `<email>`                                            |
| Known key formats    | AWS, GitHub `ghp_`, Slack, JWT, `-----BEGIN … KEY-----` → drop the pair |
| URL credentials      | `https://user:pass@…` → drop the pair                  |
| Env assignments of secret-shaped names | `*_TOKEN=`, `*_SECRET=`, `*_KEY=`, `PASSWORD=` → drop the pair |
| High-entropy strings | Base64/hex runs over a length and entropy threshold → drop the pair |
| Anything unrecognized but suspicious | drop the pair                          |

Dropping beats redacting for the credential classes. A redacted secret still teaches the
model the *shape* of the command that carried it, and a model can memorize and later emit
what it was trained on; discarding a few thousand pairs from a 30–50k set costs nothing
worth protecting.

**Gate.** Before P3 training begins, an audited random sample of the scrubbed set must show
zero surviving secrets, and the scrubber's own test suite must include planted secrets of
every class above. This is listed in the P3 acceptance criteria. If the audit fails,
training does not start.

## External command corpora — two uses, not one

Public shell corpora (dotfile repos containing `.bash_history`, training-lab transcripts,
asciinema recordings, troubleshooting writeups) are worth mining, but they divide into two
categories with very different value and very different risk.

**As training pairs — only where intent is attached.** The task is English → command, so a
bare command list supplies the *output* side and nothing else. Raw `.bash_history` has no
intent attached; turning it into pairs means synthesizing the English with a model, which
is already what the index-driven synthetic pipeline does, from cleaner input. The exception
is the third category, and it is the best of the three: **troubleshooting writeups**, where
the prose around a command states the intent. Those are real pairs. Extraction is messy but
the signal is genuine, and the wording is task-oriented, which is exactly the vocabulary
gap that broke "follow redirect" in section 4.

**As a frequency prior — all of them, cheaply and safely.** A fresh install has an empty
`rank_personal`, so bare `curl -<TAB>` has nothing to rank by. Aggregating flag frequency
across public histories fixes that cold start: how often `-xzf` follows `tar`, that `curl`
in the wild means `-fsSL` far more often than `--proto-redir`. This extracts a **frequency
table of command and flag tokens, not text** — which sidesteps both problems below, since
counts are neither copyrightable expression nor a vector for leaking a secret. Stored as
`rank_corpus` beside `rank_personal`, and overridden by local history as it accumulates.

Three cautions that apply to the training-pair use and largely dissolve for the frequency
use:

1. **Licensing.** Dotfile repos are frequently unlicensed, which means no grant to
   redistribute a derivative. tldr-pages is CC-BY-4.0, which is precisely why it is the
   backbone rather than one source among equals.
2. **Secrets.** Published `.bash_history` files are a well-known source of leaked
   credentials — enough so that scanning for them is its own security-research genre.
   Anything ingested must be scrubbed before use, and a model can memorize and later emit
   what it was trained on. Frequency counts cannot.
3. **Staleness.** Public dotfiles skew old: Python 2, `ifconfig`, pre-systemd. This one is
   self-correcting — see below.

**The verifier makes noisy sources safe.** Every candidate pair is checked against the
local index and dropped unless every flag resolves. A pair using `ifconfig -a` on a machine
that has only `ip` simply fails and is discarded. That turns a data-quality problem into a
throughput problem, and it is the same mechanism already required for synthetic data —
extended to cover every external source.

Value ranking for the dataset:

| Source                    | Pre-paired | Licence   | Use              |
| ------------------------- | ---------- | --------- | ---------------- |
| tldr-pages                | yes        | CC-BY-4.0 | backbone         |
| Claude Code transcripts   | yes        | local     | 2,231 measured   |
| Troubleshooting writeups  | yes, messy | varies    | vocabulary       |
| Raw public histories      | no         | unclear   | frequency prior  |

## Open items

- The `--help` crawler executes binaries. The allowlist gate covers it, but it deserves a
  second look before the first full `PATH` sweep.
- `mandoc -T markdown` output quality is unverified — mandoc isn't installed here and
  installing it needs a sudo prompt. This matters less than expected: the rendered-text
  parser now matches `--help` exactly on curl and ls, so mandoc is an alternative rather
  than an upgrade, and its main remaining value is macOS mdoc pages in P5.
- Description search cannot find `-L, --location` from "follow redirect" (section 4). tldr
  ingestion and cross-reference expansion are the P1 fixes.
- `tldr` is not installed here; ingesting its pages means vendoring the CC-BY-4.0
  `tldr-pages` repo or fetching the assets archive.
- Transcript-derived training data needs a rephrasing pass (terse imperative → natural
  question) before it earns its place beside tldr.
