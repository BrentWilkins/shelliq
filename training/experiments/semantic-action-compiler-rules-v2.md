# Semantic-action compiler rules v2

Status: preregistered before evaluation.

V1 matched only the summary rule because its context marker was anchored before ShellIQ's immutable output-contract prefix. V2 retains the same rule families and changes only prompt-interface facts established from loaded source fields.

## Frozen corrections

1. Match `ss:`, `iproute2 ss:`, and `procps pmap:` at a line boundary anywhere after the immutable prompt prefix, rather than requiring the marker at byte zero.
2. Use the exact pmap mode documented by context: `-x` for extended details and `-XX` for every kernel-exposed field.
3. Decline `ss` requests containing a decimal port or the phrase `specific port`; this rule version does not compile socket filter expressions.
4. Retain v1's summary logic, documented short-option order, protocol selection, numeric listing behavior, single-PID extraction, Rust encoding, and neural fallback unchanged.

## Gate

Run the same locked 92-record hybrid evaluation. Require at least 83 Rust-valid outputs, 74 first-command matches, and 5 reference-accepted outputs. Outer validation and test remain sealed.
