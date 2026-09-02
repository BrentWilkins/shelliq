# Documentation-conditioned cross-encoder v2 results

Status: passed sealed gate; runtime integration authorized and verified.

The unchanged v1 CodeT5 encoder/head was evaluated at the preregistered zero
logit boundary on 128 fresh commands. None appeared in supervised training,
development, or the opened v1 test.

| Sealed gate                           |  Required |           Result |
| ------------------------------------- | --------: | ---------------: |
| Sufficient unseen-command ready       |   ≥90/128 | 104/128 (81.25%) |
| Insufficient-documentation abstention |   128/128 |          128/128 |
| CPU warm p95                          | ≤5,000 ms |        215.10 ms |
| Pair truncation                       |  reported |                2 |

The exact source documentation record ranked first on 124/128 sufficient cases.
Twenty additional cases abstained because their correct top record scored below
zero or its request-visible binding failed Rust validation. No failed case
produced a ready command.

The first v2 report used a harness version where an unresolved binding returned
Python `False`, which the boolean evaluator could misclassify as a compiled
document. A production smoke exposed the type error because ShellIQ correctly
rejected JSON `false`. The report was excluded, `False` was corrected to `None`,
and the unchanged checkpoint, threshold, cases, scores, and candidate outputs
were rescored. The corrected result retained the same categorical totals; only
measured latency changed.

## Production-path verification

The passing checkpoint is served by
`training/scripts/serve_documentation_cross_encoder.py`. It accepts only
loopback chat completions, extracts ShellIQ's bounded shortlist or final selected
command, ranks at most 32 local templates per command, binds request-visible
values, and returns only structures drawn from the frozen Rust-round-tripped
recipe index. A score below zero, missing documentation, or unresolved slot
returns HTTP 422. The real ShellIQ client remains authoritative: it parses,
renders, reparses, verifies, and applies deterministic policy to every response.

The release `shelliq suggest` two-pass path was exercised with `lscpu`, a fresh
v2 command absent from supervised training:

```json
{
  "status": "ready",
  "command": "lscpu",
  "source": "model"
}
```

Local command/flag verification and the deterministic safety/operand-grounding
policy ran after the adapter. A low-confidence `pwd` smoke returned 422 and
ShellIQ safely selected its deterministic documentation fallback instead.

## Running the experimental adapter

From the repository root:

```sh
./shelliq-model setup
./shelliq-model serve --device cpu
```

The release-style interface is `shelliq model setup|serve|run`; it extracts the
same locked runtime and frozen assets embedded in the model-enabled binary. The
pinned CodeT5 base weights are downloaded on first setup rather than embedded.

## Frozen identities

- V2 checkpoint SHA-256:
  `8cdaf55e4df0670452d1eb2bfc32c68bb1ebbbe1ac20cdda3fb3905d32436ee8`
- V2 test SHA-256:
  `96992cca0137dea165cdb2733200930553525eb53cff5a517f73ca34c407884d`
- Corrected raw report SHA-256:
  `bb72ceb403833990aea5c6963ae0c3c3af2622b397fa8c5e9607e075a0ac0595`
- Excluded invalid report SHA-256:
  `b918e403ec310e2d92b3f78ce4be50557703a7bb57c9ef1192cd0f0527d175df`
