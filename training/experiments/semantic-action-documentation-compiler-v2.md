# Semantic-action documentation compiler v2

Status: preregistered for target-blind open-inner evaluation.

V1 proved that selecting the top retrieved example before composition creates
unsafe false confidence. V2 retains the same TLDR index, context fragment
extraction, literal binder, Rust codec, and abstention statuses, with these
generic corrections only:

1. Generate the existing bounded contextual candidate set before selection.
2. Score complete candidates by missing requested options, unsupported operands,
   extra options, instruction-only similarity to source documentation, and the
   existing retrieval score, in that order.
3. Treat at most one leading non-option argument as static subcommand structure.
   Every other non-option word must be request-visible, context-fragment-visible,
   or produced by an explicit placeholder binding; otherwise report it as an
   unresolved slot and abstain.
4. Audit redirection and other non-argument words with the same request-visible
   provenance rule.

No reference, record identifier, command-specific rule, option arity table, or
outer outcome participates in generation, scoring, or readiness.

## Gate

Every ready document must pass the Rust round trip. V2 must improve on v1's zero
accepted and zero-percent ready precision, reaching at least two conservative
acceptances and at least 25% precision among ready outputs on the open 92-record
inner diagnostic. This is a progress gate, not promotion.
