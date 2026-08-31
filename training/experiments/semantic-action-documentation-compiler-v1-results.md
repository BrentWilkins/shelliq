# Semantic-action documentation compiler v1 results

Status: failed the target-blind gate; stop top-one template selection.

The compiler returned 55 `ready`, 32 `needs_input`, and five
`no_documentation` results on the open inner split. All 55 ready documents passed
the Rust action round trip and 50 retained the expected first command, but none
matched a conservative reference. Ready precision was therefore 0%, below the
50% floor, with zero accepted outputs against the required five.

The abstention mechanism correctly catches explicit TLDR placeholders, but a
top-ranked example can contain concrete-looking operands, subcommands, or options
that are neither requested nor relevant. Those values evade placeholder
detection and create false confidence.

The next selector must score complete composed candidates against the request,
not select a base template first. It must also track provenance for every emitted
word and treat non-option operands as unresolved unless they are request-visible
or explicitly classified as static structure by the typed template. Until that
exists, the candidate generator is useful research infrastructure but not a safe
command-producing compiler.

Raw report SHA-256:
`a2180ff8a81efd0ea30c87c352eb88aa232d1e4f52a71950db586c0a85282d29`.
