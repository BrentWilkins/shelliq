# Documentation compiler input-complete v5 results

Status: passed with perfect ready precision.

The generic documentation-retrieval and typed-template compiler evaluated 100
distinct command families absent from the 610-record neural-training split. It
delivered 90 exact typed commands and explicitly abstained on ten ambiguous or
unresolved requests. Every candidate set contained the exact template. All 90
ready outputs passed the project-owned Rust semantic-action round trip, yielding
90% exact coverage and 100% ready precision against frozen floors of 80% and 95%.

The compiler uses no command-specific rules. It retrieves from a 26,463-record
command-integrity-filtered TLDR v2.3 index, composes typed candidates, binds only
request-visible values, scores whole candidates, tracks provenance for every
word, and withholds rendering whenever a required or conflicting value remains
unresolved.

This establishes the requested input-complete unseen-family capability. It does
not open the previously protected 98-record comparison test or constitute a
release decision.

Benchmark SHA-256:
`d023fb742fbe7878a124d50c861a09296bda79dd7ee47cf171d45e1a1311d1db`.

Raw report SHA-256:
`6f2cad469971462026ca7dcf8c298c2c378b73f2f72db39f1a5ce9d319f0bf68`.

Documentation index SHA-256:
`17b3b2e9feb215f858e893ba4f761088f9edfbd674ec89633c26d4332a6ab64b`.
