# Semantic-action authoritative-context fragments v1 results

Status: failed the feasibility gate, with material generic progress.

The compiler generated 8,064 candidates and every candidate passed the Rust
action round trip. Skeleton oracle coverage improved from 6/92 across five
families to 11/92 across eight families; three exact references appeared in the
bounded candidate sets. The frozen gate required 19 records across ten families.

Two representation-level failures are visible without adding command-specific
knowledge. Context options were merged before documented subcommand prefixes,
producing shapes such as options before `docker logs`. Exact command scoping also
prevented version-suffixed executables such as `python3` from reusing otherwise
applicable `python` documentation. V2 may correct only those generic interface
facts; literal-slot abstention and all other controls remain unchanged.

Raw report SHA-256:
`361b4fedafd92a7183652ff999116fe5a56c1820638be8e1c851bb86de78807d`.
