# Semantic-action authoritative-context fragments v2

Status: preregistered for open inner development; outer validation and test
remain sealed.

V2 retains v1's TLDR-only index, top-eight retrieval, generic context option
extraction, six-option bound, subset enumeration, 512-candidate bound, literal
binding, Rust validation, and no-command-specific-rule constraint.

## Frozen corrections

1. Preserve the leading non-option, non-placeholder argument prefix from each
   typed base before overlaying context options. This treats documented leading
   arguments as subcommand structure without naming any command family.
2. When exact command documentation is absent, remove only a trailing numeric
   version suffix and look up that documentation key on the same platform. If it
   exists, instantiate the retrieved typed recipe with the originally requested
   command name. No general alias table or platform fallback is allowed.

## Gate

Run the same 92-record open-inner oracle. Require every candidate to pass the
Rust action round trip and at least 19 skeleton-covered records across ten
command families. Passing authorizes target-blind selection; failure stops this
context-fragment representation.
