# Documentation clarification flow v1

Status: preregistered before dataset generation, runtime changes, or evaluation.

The production documentation fallback already fails closed on incomplete
requests and retained 100% ready precision in the combined sealed v3 and v4
evaluation. This experiment changes only the abstention interface: a safe
abstention becomes a structured, focused clarification that can be answered
and recompiled through the same semantic lowering and local verification path.

## Frozen contract

`shelliq suggest --json` returns a versioned response envelope for every
outcome. Version 1 has exactly one of these statuses:

- `ready`: includes the rendered command, semantic document, and provenance;
- `needs_input`: includes no command or semantic document and includes one
  clarification with a generic slot kind, human-readable label, and question;
- `no_documentation`: includes no command or semantic document and a reason.

The default human interface remains command-only on stdout when ready. Both
abstention statuses return nonzero with empty stdout. A `needs_input` question
is written to stderr. The Zsh widget preserves the original request and never
places a question or partial command in `BUFFER`. Supplying `--answer VALUE`
adds that value to the original request and recompiles from documentation; it
does not splice the value into shell text or bypass semantic/local validation.

Clarification kinds and questions are derived only from generic placeholder
syntax and request binding state. No command-family clarification rules are
allowed.

## Command-disjoint evaluation

The builder selects the next 64 eligible, distinct Linux command families in
the stable documentation-template order after excluding:

- the neural training partition;
- the earlier documentation compiler benchmark;
- every command used by documentation fallback end-to-end v1 through v4;
- each other command in this clarification experiment.

Every source instruction must be free of typed request literals and every
recipe must contain exactly one bindable placeholder. Each case freezes the
incomplete instruction, expected generic slot kind, synthetic answer, completed
instruction, and exact expected command. The first 16 cases are open
development; the remaining 48 are sealed test. Dataset hashes are committed
before either partition is run. The sealed partition is opened once, only after
the builder, manifest, harness, runtime revision, and open-development behavior
are frozen.

## Passing gates

On the 48 sealed cases:

1. All incomplete requests return nonzero, write empty stdout, emit no command
   in the JSON envelope, and ask exactly one non-empty question.
2. At least 95% of incomplete responses identify the frozen generic slot kind.
3. At least 80% of answered requests emit the exact expected command.
4. Ready precision after clarification is 100%: every emitted command is the
   frozen expected command.
5. Every emitted command passes semantic round-trip and local command/flag
   verification.
6. The frozen v3 plus v4 sealed ready cases retain 100% ready precision, and
   their incomplete cases continue to emit no command.
7. After one warm-up, mean end-to-end latency is at most 250 ms and maximum
   latency is at most one second.

Failure of any gate rejects the clarification flow. Sealed outcomes do not
authorize command-specific tuning or case replacement.
