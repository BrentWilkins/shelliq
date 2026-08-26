# Semantic-action compiler rules v1 results

Status: failed the inner gate.

Only the `ss -s` summary rule matched, producing one exact, valid, and accepted rule output. Overall metrics therefore remained 1/92 accepted, 92/92 Rust-valid, and 79/92 first-command match.

The failure is a rule-interface error. Loaded contexts begin with the immutable `Output contract:` prefix, so line-specific `ss:` and `procps pmap:` matchers anchored at the beginning of the whole string rejected their intended records. Inspection also showed the authoritative kernel-detail context documents `-XX`, not the draft's `-X`. No unsupported output was emitted. The next version may correct only those prompt-visible mismatches and must explicitly decline unsupported `ss` port-filter requests.
