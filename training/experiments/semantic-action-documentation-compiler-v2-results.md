# Semantic-action documentation compiler v2 results

Status: failed the precision gate, while recovering two correct commands.

Whole-candidate scoring produced two conservative acceptances, improving on v1's
zero, and every one of 71 ready documents passed the Rust round trip. Ready
precision was only 2.82%, however, far below the required 25%. The compiler
abstained on 21 records.

The output attribution exposed an index-integrity defect: some TLDR page examples
do not invoke the command under which the page was indexed, such as meta-examples
that invoke `tldr` itself. The scorer also preferred incomplete candidates that
omitted explicit request literals because omission was not yet penalized.

V3 may filter index records unless their typed document contains the indexed
command, require every extracted request literal to appear in the selected
document, and emit per-word provenance. These are generic integrity and safety
corrections, not command-specific tuning.

Raw report SHA-256:
`3b1c7747bfba3e2ef9165c4f890defe21e85dfdaac3857503b8df712a1f8403f`.
