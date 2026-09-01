# Documentation clarification source-aligned v1 — sealed results

Status: passed all preregistered gates.

The frozen 48-case sealed partition was opened once against runtime `5e31b04`.

- safe default abstentions: 48/48;
- commandless structured abstentions: 48/48;
- focused questions: 48/48;
- source-aligned clarification labels: 48/48;
- answered exact commands: 45/48 (93.75%; required at least 80%);
- answered ready precision: 100%;
- semantic and local validity: 45/45 emitted commands;
- mean latency: 15.86 ms;
- maximum latency: 17.77 ms.

The frozen v3 and v4 regression audit retained 100% ready precision: all 81
emitted commands were exact and locally/semantically valid. All 32 incomplete
requests remained nonzero commandless abstentions. The legacy per-experiment
scripts still report their known historical label/coverage gate failures, which
are outside this experiment's regression invariant.

This result resolves the earlier oracle problem: clarification identity is now
adjudicated against the documentation recipe actually selected and reported by
the runtime, rather than an assumed benchmark source row.
