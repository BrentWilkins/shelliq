# Documentation fallback end-to-end v4 results

Status: failed frozen coverage gate; production fallback integration and
end-to-end evaluation complete.

Runtime commit `8e227df`, harness commit `90cd0f7`, and the preregistered v4
partition were evaluated once.

- Complete exact and emitted: 37/48 (77.08%; frozen floor 39/48, failed).
- Ready precision: 37/37 (100%; floor 95%).
- Emitted semantic/local-verifier validity: 37/37.
- Explicit fail-closed incomplete abstentions: 16/16.
- Narrower `needs_input` classification also happened on 16/16.
- Mean latency after one warm-up: 36.72 ms; maximum 50.02 ms.
- Dataset SHA-256:
  `1b4bd3e26d3c041a82b46e731f480cdf9e5105408c00056936f8ba711b0b94ec`.
- Raw report SHA-256:
  `6aa8a6915665348fa4719beeb8e8f0f8ef86533ad90cb6558a468bfc9d084f02`.
- Binary SHA-256:
  `8977ce96dd44810768ffe1941970fc93631065f8654e6f232d9494bd626e62ff`.

The gate is failed and is not reinterpreted. Descriptively, the two independent
literal-clean sealed partitions v3 and v4 together contain 96 complete and 32
incomplete command-disjoint requests. The unchanged runtime delivered 81/96
complete commands exactly (84.38% coverage), with 81/81 ready precision and
validity, and abstained fail-closed on 32/32 incomplete requests. This aggregate
was not a preregistered promotion gate; it is reported only as robustness
context.

The result supports shipping the compiler as a conservative fallback, not as a
standalone high-coverage command generator. No wrong command crossed the ready
boundary on either literal-clean sealed partition.
