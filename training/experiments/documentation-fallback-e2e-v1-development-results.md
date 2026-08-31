# Documentation fallback end-to-end v1 development results

Status: open development passed; implementation frozen before sealed test.

The production `shelliq suggest` path was exercised on the 32-record open
partition with its loopback model endpoint deliberately unavailable. A fresh
fake executable and SQLite index were built through the public CLI for every
case.

- Complete requests: 21/24 exact commands, with 21 emitted.
- Ready precision: 21/21 (100%).
- Incomplete requests: 8/8 returned nonzero `needs_input` with empty stdout.
- Emitted semantic/local-verifier validity: 21/21.
- Mean complete-request latency after one warm-up: 41.49 ms.
- Maximum measured latency: 57.10 ms.
- Frozen development dataset SHA-256:
  `41997c67ee6d51b1229c3a8e7759d4b9812d20cdbfd3a9f3b385f659eab07a36`.
- Raw development report SHA-256:
  `c537987104259b9faec4a4399302ea8457bccdde816c5be1350e2ba31b6dd0a6`.

The three non-deliveries were conservative abstentions for documented values
that were not safely supported by request-visible literals. No incorrect
command was emitted. The runtime contains no command-family production rules;
the sole named-command test fixture covers the generic compiler API.
