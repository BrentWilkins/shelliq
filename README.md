# ShellIQ

A local CLI assistant. It answers "is it `-r` or `-R`?" from the man pages installed on
_this_ machine, and cites the line it got the answer from.

Current release line: `0.1.0-alpha.1`. Trust hardening and the separate
custom-model experiment are active; the shipping CLI remains model-free.

## Install

Installing a prebuilt release does **not** require Rust or Cargo.

### Shell installer

The recommended method is the checksum-verifying shell installer:

```console
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/brentwilkins/shelliq/releases/download/v0.1.0-alpha.1/shelliq-installer.sh | sh
```

It selects the archive for the current machine, verifies it, and installs the `shelliq`
executable in `~/.local/bin`. It creates that directory when necessary, writes a small
`PATH` setup file, and attempts to load it from the shell profile. Make the command
available immediately with:

```sh
source "$HOME/.local/bin/env"
```

If a future shell still cannot find `shelliq`, add this to its startup file:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

### Download an archive manually

Download the archive and its matching `.sha256` file from the
[GitHub release](https://github.com/brentwilkins/shelliq/releases/tag/v0.1.0-alpha.1).
Choose the target matching the machine:

| System                        | Architecture  | Release target              |
| ----------------------------- | ------------- | --------------------------- |
| macOS                         | Apple silicon | `aarch64-apple-darwin`      |
| macOS                         | Intel         | `x86_64-apple-darwin`       |
| Linux (glibc), including WSL2 | ARM64         | `aarch64-unknown-linux-gnu` |
| Linux (glibc), including WSL2 | x86-64        | `x86_64-unknown-linux-gnu`  |

For example, on x86-64 Linux or WSL2:

```console
curl -LO https://github.com/brentwilkins/shelliq/releases/download/v0.1.0-alpha.1/shelliq-x86_64-unknown-linux-gnu.tar.xz
curl -LO https://github.com/brentwilkins/shelliq/releases/download/v0.1.0-alpha.1/shelliq-x86_64-unknown-linux-gnu.tar.xz.sha256
sha256sum -c shelliq-x86_64-unknown-linux-gnu.tar.xz.sha256
tar -xJf shelliq-x86_64-unknown-linux-gnu.tar.xz
install -d "$HOME/.local/bin"
install -m 0755 shelliq-x86_64-unknown-linux-gnu/shelliq "$HOME/.local/bin/shelliq"
```

On macOS, use this checksum command (the generated file contains a harmless blank line
that macOS `shasum` otherwise warns about):

```console
grep -v '^$' FILE.sha256 | shasum -a 256 -c -
```

Ensure `~/.local/bin` is on `PATH` as shown above.

### Optional shell integration

Release archives contain `shelliq.bash` and `shelliq.zsh` alongside the executable in
the extracted target directory. The shell installer intentionally installs only the
executable. To enable an integration from a manually extracted archive, replace
`TARGET` below with the release target selected above:

```console
install -d "$HOME/.local/share/shelliq"
install -m 0644 shelliq-TARGET/shelliq.bash shelliq-TARGET/shelliq.zsh \
  "$HOME/.local/share/shelliq/"
```

Then add one matching line to the shell startup file:

```sh
# ~/.zshrc
source "$HOME/.local/share/shelliq/shelliq.zsh"

# ~/.bashrc
source "$HOME/.local/share/shelliq/shelliq.bash"
```

### Build from source

This is the only installation method that requires a Rust toolchain. From a ShellIQ
source checkout:

```console
cargo install --locked --path crates/shelliq --root "$HOME/.local"
```

Other Unix-like systems, including BSDs, are best-effort source builds. Native Windows
is not an alpha target because ShellIQ depends on Unix man pages and shell semantics;
use the Linux build inside WSL2.

### First run

Build the local documentation index before asking ShellIQ about commands:

```console
shelliq --version
shelliq index scan
shelliq index stats
shelliq explain grep -r
```

For a smaller initial index, replace `index scan` with:

```console
shelliq index build grep tar curl ls find
```

## Try it

```console
$ cargo build --release
$ ./target/release/shelliq index build grep tar curl ls find
  grep(1)  47 flags
  tar(1)  156 flags
  curl(1)  258 flags
  ls(1)  59 flags
  find(1)  80 flags

$ shelliq explain grep -r
  ✓ -r  Read all files under each directory, recursively, following…  grep(1):168

$ shelliq explain grep -sirN pattern .
  ✓ -s  Suppress error messages about nonexistent or unreadable fil…  grep(1):85
  ✓ -i  Ignore case distinctions in patterns and input data, so tha…  grep(1):45
  ✓ -r  Read all files under each directory, recursively, following…  grep(1):168
  ~ -N  not valid for grep; did you mean -n?  grep(1):101
      Prefix each line of output with the 1-based line number within its in…

  1 of 4 need attention
```

That last example is the reason the project exists. A wrong case hides invisibly inside a
bundled short flag — `-sirN` looks fine — so the checker decomposes bundles before
checking anything.

```console
shelliq search curl maximum redirects     # find a flag by what it does
shelliq flags tar                         # every flag, ranked
shelliq index scan                        # index PATH commands with installed man pages
shelliq index stats
```

## How it works

Facts and fluency are kept apart:

| Concern                             | Owner                        |
| ----------------------------------- | ---------------------------- |
| Flag facts, case, arguments         | SQLite index, built locally  |
| English → command shape             | A small model (later phases) |
| Whether each option spelling exists | Option checker, index-backed |

A local small model on its own is _less_ reliable than a cloud one. What makes ShellIQ
useful is not that it runs locally — it is that every option it reports is looked up in the
man page on your disk and carries a citation. Running locally is the privacy story, not the
accuracy story.

**What the checker does and does not tell you.** It tells you an option spelling exists for
that command on this machine, with roughly the right argument shape. It does not tell you
the command does what you asked, that the operands are right, that the flags make sense
together, or that running it is safe. A checked command is not a verified command; see
the claim ladder in `PLAN.md`.

The index is built per machine and never shipped, so macOS needs no data transfer. It can
still go stale — it reflects the pages as of the last build, so `shelliq index refresh`
after an upgrade is what keeps it honest.

## Experimental model alpha

The current 0.5B GGUF is a development baseline, not a release candidate. It
failed the frozen P1B production-path gate after unsafe and unsupported requests
crossed the ready boundary; see
[`p1b-runtime-v1-results.md`](training/experiments/p1b-runtime-v1-results.md).
The existing Salesforce CodeT5 checkpoint also failed its fresh production-path
gate: 0/4 supported unseen commands, 0/4 correct poisoned-context responses, and
10.16 s CPU p95. Deterministic grounding safely rejected its invented operands,
but did not make it useful; see
[`codet5-runtime-v1-results.md`](training/experiments/codet5-runtime-v1-results.md).
A replacement documentation-conditioned CodeT5 ranker has now passed a fresh
command-disjoint gate: 104/128 unseen commands compiled ready, 128/128
insufficient-documentation cases abstained, and CPU p95 was 215 ms. It ranks
local Rust-valid recipes instead of generating command words. The frozen recipe
index and ranker head are included in the source tree; setup downloads the exact
pinned Salesforce CodeT5 base revision. The evaluation is documented in
[`documentation-cross-encoder-v2-results.md`](training/experiments/documentation-cross-encoder-v2-results.md);
the model runtime remains optional and is not embedded in the Rust binary.

`shelliq suggest` can query an OpenAI-compatible model server bound to numeric
loopback, deserialize compact SemanticDocumentV2 JSON, validate its Rust
render/reparse round trip, and check recognized flags against the local index.
It prints an editable command and never executes it:

### Run the documentation model

With [uv](https://docs.astral.sh/uv/) installed, the release binary extracts its
hash-checked runtime bundle on demand:

```console
shelliq model setup --scan
shelliq model run --json 'Show information about all CPUs'
```

`setup` verifies and installs the frozen assets, downloads only the pinned
CodeT5 revision, and optionally builds the ordinary per-machine man-page index.
`run` starts a temporary loopback adapter, asks the real ShellIQ client for one
suggestion, and shuts the adapter down. The locked runtime supports x86-64 and
ARM64 Linux (including WSL2) and Apple-silicon macOS; inference is CPU-only.

For shell widgets or repeated requests, keep the warmed adapter running:

```console
shelliq model serve
shelliq suggest --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --timeout-ms 10000 copy a tree while preserving its attributes
```

The first setup downloads an isolated CPU-only Python runtime and roughly 250 MB
of CodeT5 weights (about 500 MB of downloads in total). Neither setup nor
suggestion executes a generated command.
From a source checkout, `cargo build --release` produces the same model-enabled
binary; `./shelliq-model` remains a direct development entry point.

For any compatible loopback adapter, the lower-level client remains:

```console
$ shelliq suggest --endpoint http://127.0.0.1:8080/v1/chat/completions \
    copy a tree while preserving its attributes
experimental model suggestion; flags checked, operand semantics unverified; inspect and edit before running
cp -a SOURCE/DEST/
```

Run `shelliq index scan` first. It discovers executable filenames on `PATH`
without starting them and indexes only their installed man pages. In automatic
mode, `suggest` searches that documentation for a small installed-command
shortlist, requires a draft model pass to choose from it, retrieves cited flags
for the selected command, and runs a final model pass with that evidence. If no
documentation matches, it abstains and asks you to scan.

`--context` remains an explicit override for supplying evidence yourself; a
non-empty value uses the existing single model pass. Targeted
`shelliq index build NAME` may use its guarded `--help` crawler when no man page
exists, but broad `index scan` never executes discovered programs.

For zsh, source `shell/shelliq.zsh`, type an English request at an empty prompt,
and press `Ctrl-X Ctrl-G`. The widget asks focused questions for
missing documentation values and pins every answer to the initially reported
recipe.
Cancellation, empty answers, invalid responses, and changed sources leave the
original request untouched. The completed suggestion is placed in the editable
command buffer; it is never executed automatically, and Enter remains an
explicit choice. Alternatively, `shelliq-suggest find large files` preloads
the next input line. VS Code may intercept terminal chords; on this development
machine, `Ctrl-Alt-G` is configured to send the widget sequence to zsh.

For Bash, source `shell/shelliq.bash` from an interactive shell and use the same
`Ctrl-X Ctrl-G` binding. It provides the same source-pinned clarification and
buffer-preservation behavior through Readline; completed commands remain
editable and require an explicit Enter. `Ctrl-X Ctrl-H` explains the current
buffer against the local index without changing it.

On Bash 4 or newer, Tab completion offers indexed flags only for commands
without a native completion specification and only while completing a
`-`-prefixed word. If the user already has a default completion policy, ShellIQ
leaves it untouched; otherwise ordinary Bash and filename completion remain
enabled as fallbacks. The suggestion and explanation widgets also support the
Bash 3.2 version bundled with macOS.

`suggest` defaults to the versioned `context-authoritative-v1` prompt contract.
Use `--prompt-contract legacy-user-v1` only with an older adapter trained on
that exact prompt shape; the prompt contract must match the served model.

This is an integration preview, not a correctness claim. Command-level operand
arity and intent coverage are not yet validated, so suggestions can be
syntactically valid while still being incomplete or wrong. The client refuses
non-loopback endpoints, credentials, redirects, ambient proxies, oversized
responses, invalid semantic JSON, and semantic documents that do not round trip.

Pipeline data compatibility is also not validated. For example, a suggestion
that feeds `find -print0` directly to line-oriented `sort` can contain only
real options and still be functionally wrong. Treat the editable-buffer step as
a required review boundary, especially for pipelines.

### Documentation fallback

If a model pass or its local option check fails, `suggest` tries a deterministic
fallback compiled from the vendored TLDR recipe for the retrieved, installed
command. It binds only typed values visible in the request, lowers through the
same `SemanticDocumentV2` round trip, and runs the same local command/option
verifier. It emits only a complete command. Missing values return `needs_input`
with one focused question, a nonzero exit, and no standard output, so the Zsh
widget keeps the original request and never puts an abstention in the editable
buffer. The `needs_input` envelope includes a versioned `continuation.source`.
Pass that source back with `--continue-from SOURCE --answer VALUE` to pin
recompilation to the reported recipe and the same local verification path. For
recipes needing more than one value, repeat `--answer` in question order on each
continuation. If no documentation matches, it returns `no_documentation`.

`shelliq suggest --json` prints a versioned envelope for all three outcomes:
`ready`, `needs_input`, and `no_documentation`. A ready envelope contains the
rendered command, semantic document, and provenance. Abstention envelopes never
contain a command or semantic document. Documentation-backed envelopes identify
the selected command family, recipe source, and intent so clients can associate
a clarification with the recipe that produced it.

On two independent literal-clean sealed sets, the unchanged runtime delivered
81/96 complete requests exactly with 100% ready precision and local/semantic
validity, and abstained with nonzero exit plus empty stdout on 32/32 incomplete
requests. The final v4 preregistered gate still failed its per-partition coverage
floor (37/48 versus 39 required), so this is a conservative fallback rather than
a standalone high-coverage generator. Full evidence is in
`training/experiments/documentation-fallback-e2e-v4-results.md`.

## Footprint

The Rust binary links no inference library, opens no socket until `suggest` is explicitly
invoked, and needs no GPU. The optional `shelliq-model` process owns PyTorch and the pinned
CodeT5 weights and is reached only over numeric loopback.

## Layout

```text
crates/shelliq/   CLI entrypoint
crates/harvest/   man page parser, --help crawler
crates/index/     schema, FTS5, ranking
crates/verify/    tokenizer, bundle decomposition, option checking
shell/            Zsh and Bash interactive integrations
training/         JAX/Flax fine-tuning (uv, Python 3.14, development only)
```

`PLAN.md` has the full design, the measurements behind it, and the things that turned out
not to work.

## Accuracy

The parser is checked against each tool's own `--help` rather than a hand-counted figure.
It reproduces the **long** flags of `curl --help all` exactly at 258 and of `ls --help`
exactly at 44, with no divergence in either direction. Short flags are not covered by that
comparison. Run `cargo test` to check against the pages on your machine.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
