# shelliq

A local CLI assistant. It answers "is it `-r` or `-R`?" from the man pages installed on
*this* machine, and cites the line it got the answer from.

Status: **P0**. The index, man-page harvester, and option checker work. There is no model
yet, and most of the point is that the common cases never need one.

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
$ shelliq search curl maximum redirects     # find a flag by what it does
$ shelliq flags tar                         # every flag, ranked
$ shelliq index stats
```

## How it works

Facts and fluency are kept apart:

| Concern                              | Owner                        |
| ------------------------------------ | ---------------------------- |
| Flag facts, case, arguments          | SQLite index, built locally  |
| English → command shape              | A small model (later phases) |
| Whether each option spelling exists  | Option checker, index-backed |

A local small model on its own is *less* reliable than a cloud one. What makes shelliq
useful is not that it runs locally — it is that every option it reports is looked up in the
man page on your disk and carries a citation. Running locally is the privacy story, not the
accuracy story.

**What the checker does and does not tell you.** It tells you an option spelling exists for
that command on this machine, with roughly the right argument shape. It does not tell you
the command does what you asked, that the operands are right, that the flags make sense
together, or that running it is safe. A checked command is not a verified command; see
the claim ladder in `PLAN.md`.

The index is built per machine and never shipped, so macOS needs no data transfer. It can
still go stale — it reflects the pages as of the last build, so `shelliq index --refresh`
after an upgrade is what keeps it honest.

## Footprint

The default build links no inference library, opens no socket, and needs no GPU. A model is
optional, downloaded on demand, and reached over HTTP through whatever you already run
(`ollama` or `llama-server`) — which is also how Metal, Vulkan, and CUDA support arrive
without shelliq containing any backend code.

## Layout

```
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

MIT
