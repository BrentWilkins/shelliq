# Curated corpus

First-party, hand-authored schema-version-1 rows. Unlike `artifacts/`, this
directory is **git-tracked**: every row is reviewable, diffable, and editable by
hand.

The corpus currently contains 786 rows across 17 command-family files.

Files named `reviewed-preference-*.jsonl` share this reviewable directory but
form a separate, strict preference schema: `pair_id`, `chosen`, and `rejected`
replace the supervised corpus's `record_id` and `response`. Supervised corpus
loaders and syntax ratchets must exclude those files; preference consumers use
`shelliq_training.preference_data.load_preference_jsonl` explicitly.

`targeted-finishing.jsonl` and `targeted-finishing-v2.jsonl` form a standalone 100-row finishing family.
Their commands are deliberately disjoint
from the 35-row release benchmark; `scripts/build_targeted_semantic_dataset.py` enforces that boundary and
requires independent train and test command hashes before semantic conversion.

These rows exist to cover what tldr structurally cannot teach. tldr examples are
single-clause, placeholder-stripped, and overwhelmingly GNU-flavored. The gaps
this corpus targets are the ones `PLAN.md` names: flag-case traps (`-r` vs
`-R`), bundled short flags (`-fsSL`, `-xzf`, `-LsSf`), GNU-versus-BSD
divergence, compound pipelines, Zsh-native syntax, and the modern tools that
ship no man page at all (`kubectl`, `cargo`, `uv`, `ruff`, `just`).

## Row conventions

| field        | value                                                        |
| ------------ | ------------------------------------------------------------ |
| `record_id`  | `curated:<file-stem>:<platform>:<slug>`                      |
| `corpus`     | `distributable`                                              |
| `source`     | `shelliq-curated`                                            |
| `license`    | `MIT OR Apache-2.0`                                          |
| `provenance` | `corpus/<file>#<slug>@2026-08-07:model-authored`             |
| `command`    | must equal `shell_command_name(response)`                    |
| `context`    | index-shaped flag facts, **not** a restatement of the answer |

`context` is the field that carries the teaching signal. Write the flag facts a
lookup index would hold — what `-R` means, which platform spells it that way —
rather than paraphrasing the instruction.

No `{{ }}` placeholders: `smoke_finetune.automatic_preflight` rejects them.

## Validation tiers

Every response must pass, in order:

1. `zsh -f -n` — native Zsh syntax
2. `SyntaxDocumentV1::parse` — lossless CST parse, render equals input
3. `SyntaxDocumentV1::validate`

Rows also pass `SemanticDocumentV2::lower` + semantic round-trip **unless** they
are listed in [`semantic-exempt.txt`](semantic-exempt.txt).

Semantic AST v2 covers every curated row that clears the CST boundary, including
here-string redirects and `typeset` scalar and array declarations. The semantic
exemption list remains a **ratchet, not an escape hatch**: the Rust test asserts
every exempt row _fails_ to lower. When the semantic slice grows to cover a
shape, its exemptions go stale and the test fails until they are deleted.

## Shell dialect

The long-term target is Zsh **plus POSIX `sh` plus Bash**. The corpus is
Zsh-first today, and `docs/AI_PIPELINE.md` requires other shells to arrive as
explicit dialect profiles rather than being mixed into Zsh rows — so the split
is drawn at the **file** level, not row by row:

| | rows | zsh | bash | sh |
| ------------------- | ---: | ---: | ---: | ---: |
| `zsh-native.jsonl`  |   47 |   47 |   26 |   25 |
| everything else     |  639 |  639 |  639 |  634 |

`zsh-native.jsonl` is dialect-locked by design: glob qualifiers, globbing flags,
and `print -N` have no Bash or `sh` equivalent, and teaching them is the point.
A future `sh` profile skips that file wholesale. The remaining 627 rows are
ordinary command usage that already parses in all three shells, apart from five
`pipelines` rows using process substitution or a here-string — Zsh and Bash, but
not POSIX `sh`.

`schema_version: 1` has a closed field set with no dialect field, so this lives
in [`dialect-exempt.txt`](dialect-exempt.txt) instead, which accepts a bare
filename for a whole family or a `record_id` for one row. It is the third
ratchet: `tests/test_corpus.py` asserts every unlisted row parses under `zsh -n`,
`bash -n`, **and** `sh -n`, and that every listed one still genuinely fails.
Adding a non-portable row to a portable family breaks the build until it is
fixed or recorded with a reason. When a dialect field does land, this file is the
migration input.

## What does not belong here

Derived and merged datasets go in gitignored `artifacts/`. Build one with
`scripts/merge_corpus.py`; its manifest records the exact input hashes and
corpus composition. Never commit a merge output to this directory. Personal
rows never appear here at all — this corpus is `distributable` only.
