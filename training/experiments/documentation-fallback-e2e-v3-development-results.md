# Documentation fallback end-to-end v3 development results

Status: passed open development; runtime frozen for sealed v3 test.

- Complete exact and emitted: 21/24.
- Ready precision: 21/21 (100%).
- Safe incomplete `needs_input` abstentions: 8/8.
- Emitted semantic/local-verifier validity: 21/21.
- Mean latency after one warm-up: 40.21 ms.
- Maximum latency: 44.88 ms.
- Dataset SHA-256:
  `7187d30c5b54c5bc5a475670bc9fc517739c78dd94ed3cc00703957e60d00094`.
- Raw report SHA-256:
  `8e38540669768fca1f5fef4d86b4541c9203e559ac750b451ef112b2b119c019`.

The only runtime adjustment made on open development was status-preserving:
an unbound marked TLDR exemplar now contributes an unresolved slot and therefore
`needs_input`, instead of erasing the best documentation intent and returning
`no_documentation`. It cannot make an incomplete candidate ready or emit a new
command. The v3 sealed partition remained unopened.
