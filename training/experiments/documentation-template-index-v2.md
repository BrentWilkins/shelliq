# Documentation template index v2

Status: preregistered integrity correction.

V2 retains the converted TLDR v2.3-only source, provenance, platform metadata,
and Rust encode/decode validation from v1. It excludes a record unless at least
one command node in its typed document equals the indexed command, allowing only
generic removal of a trailing numeric version suffix on both names. This removes
meta-examples and cross-command recipes without a command-specific denylist.

The output manifest records the resulting count, command coverage, source hash,
index hash, and Rust round-trip count. All retained templates must round-trip.
