# Documentation clarification flow v1 — sealed results

Status: rejected by the preregistered sealed gate.

The frozen 48-case partition was opened once against runtime `ef76b30`.

- safe default abstentions: 48/48;
- commandless structured abstentions: 48/48;
- focused questions: 48/48;
- exact generic slot kinds: 44/48 (91.7%; required at least 95%);
- answered exact commands: 43/48 (89.6%; required at least 80%);
- answered ready precision: 100%;
- semantic and local validity: 43/43 emitted commands;
- mean latency: 15.67 ms;
- maximum latency: 18.23 ms.

The four kind disagreements expose a measurement mismatch: the frozen oracle
uses broad substring matching while the runtime uses token boundaries. For
example, the oracle labels `path/to/file.hurl` as a URL because `hurl` contains
the substring `url`, while the runtime labels the explicit path as a path. No
post-sealed runtime tuning or case replacement was performed.

The frozen v3 and v4 sealed regression audit retained 100% ready precision:
81/81 emitted complete commands were exact and locally/semantically valid.
All 32 incomplete requests remained nonzero commandless abstentions. V3 now
labels one abstention `no_documentation` rather than `needs_input`, so its old
label-specific gate reports failure even though the clarification
preregistration's no-command regression invariant passed. V4 retains its known
37/48 coverage-gate failure.

The next experiment should correct only the measurement contract: identify the
missing slot by its exact documentation-placeholder label, with kind retained as
descriptive metadata. It must use a fresh command-disjoint partition and the
unchanged runtime.
