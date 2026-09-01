# Documentation clarification continuation v3

Status: preregistered before runtime changes, v3 dataset generation, harness
changes, or evaluation.

V2 open development isolated a verifier-interface failure. The index can contain
an exact multi-character single-dash option such as `-type`, and `split_bundle`
already gives exact indexed spellings precedence over POSIX bundle splitting.
However, the outer verifier removes inline `=value` only for `--long=value`, so
`-type=AXFR` never reaches that exact lookup.

V3 may change only this generic verifier behavior: when a token has `=`, its
prefix starts with one dash, has more than two characters, and is an exact local
flag, treat the suffix as an inline argument before bundle processing. Unknown
prefixes, ordinary short bundles, wrong-case spellings, and argument checks keep
their existing behavior. A unit regression must cover accepted exact spelling,
unknown spelling, and wrong case.

The continuation runtime, binder, response contract, v2 hermetic fixture, and
all ranking/rendering behavior remain frozen. The builder selects the next 64
eligible distinct Linux command families after exclusions through continuation
v2, split 16 open and 48 sealed. Hashes are committed before either run.

Passing gates remain: every incomplete/intermediate and invalid continuation
fails closed; at least 95% complete within eight distinct questions; ready
precision and local/semantic validity are 100%; source-aligned v1 and fallback
v3/v4 retain 100% ready precision and commandless incomplete cases; mean flow
latency is at most 500 ms and maximum at most two seconds.

The sealed partition is opened once after runtime, harness, hashes, and open
behavior are frozen. Failure rejects v3 without tuning or case replacement.
