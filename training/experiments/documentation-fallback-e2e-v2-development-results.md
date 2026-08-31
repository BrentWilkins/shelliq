# Documentation fallback end-to-end v2 development results

Status: runtime change passed open development; benchmark construction invalid;
sealed v2 test remains unopened.

The preregistered intent lock and request-literal completeness rules produced:

- Complete exact: 21/24.
- Complete emitted: 22/24.
- Ready precision: 21/22 (95.45%).
- Safe incomplete abstentions: 8/8.
- Emitted semantic/local-verifier validity: 22/22.
- Mean latency after one warm-up: 40.50 ms; maximum 42.97 ms.
- Development dataset SHA-256:
  `860c06afc194f5201b19c03f82738bb66baaa41eb8c4d96b7442b195b9edc8ac`.
- Raw report SHA-256:
  `f4cb9d008639dfa924ce43d832cf58517a33d032cd309d332ea9e530b814e379`.

Open-case inspection found that the sole emitted mismatch was a bad expected
target: the generic Python benchmark binder interpreted the prose fragment
`BIN/CUE` as a `/CUE` request path, consumed it, and ignored one of the two
explicitly supplied `/tmp` paths. The production compiler consumed both supplied
paths. This invalidates exact precision as a measurement of runtime correctness.

V2 test is not opened. The next dataset must use fresh command families and
exclude any source instruction from which the generic literal extractor finds a
path, URL, remote path, integer, or quoted text before synthetic values are added.
