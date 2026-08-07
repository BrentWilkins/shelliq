# Zsh-first shell syntax and model output contract

## Decision

Shelliq targets Zsh first. A shell command is a shell *program*, not merely a
program name followed by flags and operands. The eventual semantic AST must be
able to represent the supported Zsh grammar: lists, pipelines, compound
commands, redirections, assignments, quoting, expansions, functions, loops,
conditionals, and Zsh-specific constructs. A raw-shell escape node is not an
acceptable substitute because it would bypass structural verification.

`tree-sitter-zsh` is the structural parser. Native `zsh -f -n -c` is the
development and corpus validity oracle. The two checks serve different jobs:

- Tree-sitter produces a walkable tree for verification and lowering.
- Native Zsh decides whether Zsh itself accepts the source without executing it
  or loading user startup files.

`crates/syntax` contains schema-version-1 `SyntaxDocumentV1`. It is a lossless,
Serde-compatible concrete syntax tree (CST), not yet the compact semantic AST
the model should learn. It preserves every accepted source byte, renders it
exactly, reparses the rendered source, and rejects a serialized tree whose node
structure does not match its text. Compact serialized keys are intentional,
but a parser CST is still too verbose to make the final 0.5B-model target.

The next lowering layer must define stable project-owned semantic nodes rather
than exposing Tree-sitter's 213 grammar node types as the learned API. The CST
is the lossless reference and validation boundary for that work.

## Parser bake-off

Measured on 2026-08-07 against `artifacts/tldr-v2.3.jsonl` (30,351 rows), using
Zsh 5.9, `brush-parser` 0.4.0, `tree-sitter-bash` 0.25.1, and
`tree-sitter-zsh` 0.63.4:

| Parser or gate | Accepted | Relevant result |
| --- | ---: | --- |
| `brush-parser` | 29,986 | All accepted trees rendered and reparsed; 29,954 reached a one-pass render fixed point. Bash/POSIX AST, not Zsh. |
| `tree-sitter-bash` | 29,983 | Error-free CSTs; wrong dialect for the primary shell. |
| `tree-sitter-zsh` | 30,002 | Best Zsh structural coverage. |
| Native `zsh -f -n -c` | 29,988 | Syntax oracle. |

Tree-sitter Zsh and native Zsh agree on 30,295 rows. Their safe intersection is
29,967 rows, or 99.93% of the native-valid corpus. Tree-sitter rejects 21 rows
accepted by native Zsh and accepts 35 rows native Zsh rejects. Both reject 328
rows; inspection shows that many are interactive keystrokes such as `<p>`, not
shell commands. This is a corpus defect the earlier `shlex` preflight could not
detect.

`yash-syntax` was rejected because it is POSIX-only and GPL-3.0-or-later, which
is not compatible with the workspace's MIT-or-Apache-2.0 distribution.

## Mandatory training gate

Before a response can become an AST training target:

1. Native `zsh -f -n -c` must accept it.
2. `SyntaxDocumentV1::parse` must accept it without Tree-sitter error or missing
   nodes.
3. Its lossless CST must render byte-for-byte to the original response.
4. The serialized CST must pass render/reparse structural validation.
5. Semantic lowering must support every CST node present; unsupported nodes are
   a hard, audited rejection, never a raw-text fallback.

Run the repeatable corpus comparison from the repository root:

```sh
cargo run --release -p shelliq-syntax --bin audit-zsh-corpus -- \
  training/artifacts/tldr-v2.3.jsonl
```

Convert the checked-in curated corpus into validated semantic targets and a
coverage manifest:

```sh
cargo run -p shelliq-syntax --bin convert-semantic-corpus -- \
  training/corpus \
  training/artifacts/curated-semantic-v2.jsonl \
  training/artifacts/curated-semantic-v2.manifest.json
```

The manifest makes CST failures, semantic failures, and normalized semantic
renders explicit. Unknown failures and stale exemptions abort conversion.

The JSON report contains counts and at most 20 record IDs per disagreement
class. It never writes response text, so the same audit shape can safely be used
for a personal corpus.

## Semantic AST requirements

The project-owned semantic AST should be versioned independently of the CST and
cover at least these node families before a quality run:

- complete programs, comments, separators, and `&&`/`||` lists;
- pipelines, negation, background execution, and coprocesses;
- simple commands with ordered assignments, words, and redirections;
- words composed from literal, quoted, parameter, command, arithmetic, brace,
  glob, and process-substitution parts;
- subshell and brace groups;
- `if`, `case`, `for`, C-style `for`, `while`, `until`, `repeat`, and functions;
- Zsh arrays, subscript and parameter-expansion flags, glob qualifiers, and
  redirection forms represented by the selected grammar.

The renderer must be deterministic. The release gate is semantic round-trip:
render the semantic AST, pass native Zsh syntax validation, parse it into the
lossless CST, lower it again, and compare semantic ASTs. Formatting differences
may normalize; syntax and expansion behavior may not.

### Lowering implementation status

`shelliq-syntax::semantic` owns semantic schema version 1. The first lowering
slice covers sequential simple commands, ordered arguments, `|` and `|&`
pipelines, and common file redirects. It normalizes inter-statement whitespace
while validating every decoded document by render, Zsh reparse, re-lower, and
semantic equality.

Words are intentionally retained as validated, lossless lexical units in this
slice. They are not yet the final model vocabulary. Before semantic ASTs become
training targets, words must be lowered into the literal, quoting, expansion,
glob, and substitution parts listed above, and the remaining statement families
must be implemented.
