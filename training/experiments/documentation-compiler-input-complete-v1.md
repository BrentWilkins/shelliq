# Documentation compiler input-complete benchmark v1

Status: preregistered capability benchmark; no outer-validation or test access.

## Purpose

The legacy curated prompts frequently omit literal operands that their exact
references require. This benchmark isolates the requested capability: retrieve
independent documentation for command families absent from neural training,
instantiate typed slots from values explicitly supplied by the request, and
produce the exact typed command without command-specific rules.

## Frozen construction

1. Use command-integrity-filtered TLDR v2.3 index v2.
2. Exclude every command appearing in the 610-record outer neural-training split.
3. Consider records in stable record-ID order and select at most one record per
   command, up to 100 commands.
4. Require one plain statement and command, no redirects or complex nodes, at
   least one generic placeholder, at most one leading static subcommand, and only
   options or placeholders thereafter.
5. Instantiate distinct slots in document order with deterministic, type-shaped
   values: absolute paths, integers, remote paths, URLs, or quoted text. Append
   those explicit values and slot names to the original documentation
   instruction. Produce the expected document with the same target-blind generic
   binder used by the compiler.
6. Keep the original TLDR context, source record, platform, and expected typed
   document in the benchmark manifest. No curated reference contributes content.

## Gate

The benchmark must contain at least 50 distinct unseen command families. Every
ready output must pass the Rust action round trip. Require at least 80% exact
typed-document coverage over all records and at least 95% precision among ready
outputs. Report abstentions and provenance. No generated command is executed.
