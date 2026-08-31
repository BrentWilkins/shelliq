# Documentation compiler input-complete benchmark v3

Status: preregistered selector-evidence correction.

V3 retains the exact v2 benchmark construction, filtered index, candidate
generation, completeness checks, provenance audit, abstention behavior, and
80%-coverage/95%-precision gate.

The only selector correction is generic: when the normalized documentation
instruction is an exact prefix of the request instruction, add a fixed unit
bonus to that source's instruction-only cosine similarity. Explicit values are
appended after the documentation instruction by the benchmark, so this prefers
the directly matching recipe without consulting record IDs or references.

No command-specific rule or outer/test access is allowed.
