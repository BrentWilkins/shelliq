# Semantic-action documentation compiler v1

Status: preregistered for target-blind open-inner evaluation.

## Frozen compiler

1. Retrieve the top eight same-command, same-platform TLDR typed templates with
   the existing TF-IDF index and generic numeric-version fallback.
2. Select the top-ranked template without reference access.
3. Split authoritative context into generic clauses. Select an option fragment
   only when its clause shares a non-stopword lexical stem with the instruction,
   or when the option spelling occurs explicitly in the instruction. Attach only
   an immediately following numeric or quoted value.
4. Preserve the template's leading subcommand prefix, overlay selected options,
   and bind compatible literals explicitly present in the instruction.
5. Return `ready` only when no generic placeholder remains. Otherwise return
   `needs_input` with the unresolved slot names. Return `no_documentation` when
   retrieval is empty. Never execute the document.

No command-specific rules, aliases, arity tables, evaluation identifiers, or
reference-derived features are allowed.

## Gate

Every `ready` result must pass the Rust action round trip. On the open 92-record
inner diagnostic, require at least five conservative reference acceptances and
at least 50% conservative precision among `ready` results. Report abstentions,
unresolved slots, command accuracy, and exact acceptance. Failure stops this
selector; it does not authorize relaxing abstention or inspecting outer data.
