# Semantic-action documentation compiler v3

Status: preregistered for target-blind open-inner evaluation.

V3 retains v2's bounded whole-candidate generation and scoring, with three
generic integrity corrections:

1. Use the command-integrity-filtered documentation-template index v2.
2. Extract literal paths, remote paths, URLs, quoted values, and integers from
   the instruction. Penalize candidates missing any such literal before operand
   and similarity scoring, and report missing literals as unresolved.
3. Emit provenance for every word as command, request binding, request literal,
   context fragment, documentation option, static subcommand, unresolved
   placeholder, or unsupported documentation operand. `ready` still requires no
   unresolved or unsupported word.

No command-specific rules or reference-derived features are allowed. Every ready
document must pass the Rust round trip. The open-inner progress gate remains at
two conservative acceptances and 25% ready precision; passing would authorize a
broader input-complete evaluation, not promotion.
