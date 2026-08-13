# ShellIQ

A local CLI assistant. It answers "is it `-r` or `-R`?" from the man pages installed on
*this* machine, and cites the line it got the answer from.

Status: P0 is shipped. Trust hardening and the separate custom-model experiment are
active; the shipping CLI remains model-free.

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

| Concern                              | Owner                        |
| ------------------------------------ | ---------------------------- |
| Flag facts, case, arguments          | SQLite index, built locally  |
| English → command shape              | A small model (later phases) |
| Whether each option spelling exists  | Option checker, index-backed |

A local small model on its own is *less* reliable than a cloud one. What makes ShellIQ
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

An opt-in `cargo build --release --features model` build adds
`shelliq suggest`. It can query an OpenAI-compatible model server bound to numeric
loopback, deserialize compact SemanticDocumentV2 JSON, validate its Rust
render/reparse round trip, and check recognized flags against the local index.
It prints an editable command and never executes it:

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

For zsh, source `shell/shelliq.zsh`, type an English request at an empty
prompt, and press `Ctrl-X Ctrl-G`. The suggestion is placed in the editable
command buffer; it is never executed automatically, and Enter remains an
explicit choice. Alternatively, `shelliq-suggest find large files` preloads
the next input line.

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

## Footprint

The default build links no inference library, opens no socket, and needs no GPU. A model is
optional, downloaded on demand, and reached over HTTP through whatever you already run
(`ollama` or `llama-server`) — which is also how Metal, Vulkan, and CUDA support arrive
without ShellIQ containing any backend code.

## Layout

```text
crates/shelliq/   CLI entrypoint
crates/harvest/   man page parser, --help crawler
crates/index/     schema, FTS5, ranking
crates/verify/    tokenizer, bundle decomposition, option checking
shell/            zsh and bash integration (not written yet)
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
