# Documentation fallback end-to-end v4 development results

Status: passed open development; runtime and harness frozen for sealed test.

- Complete exact and emitted: 22/24.
- Ready precision: 22/22 (100%).
- Explicit fail-closed incomplete abstentions: 8/8.
- Of those, `needs_input`: 8/8.
- Emitted semantic/local-verifier validity: 22/22.
- Mean latency after one warm-up: 39.97 ms; maximum 43.44 ms.
- Dataset SHA-256:
  `6b29ff93d2a2be570f087e092ff3042c20ba2f89a423517e556378e17769afd9`.
- Raw report SHA-256:
  `fe3b00418605695d104174b5070d3ceb92511433626b1d81d632d4ed1aa15605`.
- Frozen runtime binary SHA-256:
  `8977ce96dd44810768ffe1941970fc93631065f8654e6f232d9494bd626e62ff`.

V4 changed no runtime code. Its harness implements the preregistered product
invariant by accepting either explicit abstention status only when exit is
nonzero and stdout is empty.
