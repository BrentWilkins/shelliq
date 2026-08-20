# Verifier-backed preference data

Preference training targets the case where the model can produce a plausible
semantic document but ranks a subtly wrong one above the correct one. It is not
a replacement for supervised examples that teach missing commands or concepts.

## Record lifecycle

1. Select prompts only from training partitions or newly authored command
   families. Never source prompts from release, retention, pipeline, or shadow
   evaluation suites.
2. Generate several candidates from a frozen candidate model and optionally a
   stronger teacher. Preserve model IDs, checkpoint hashes, decoding settings,
   and raw outputs.
3. Run deterministic checks that apply to every candidate: JSON/envelope shape,
   SemanticDocumentV2 conversion, shell AST rendering, literal grounding, known
   option arity, and safe fixture outcomes where available.
4. A reviewer confirms the intended task and chooses a correct response. A
   rejected response must remain plausible enough to teach a distinction; do
   not use malformed JSON as the bulk of preference data.
5. Record why the rejected response loses using one or more closed failure
   labels. Keep raw candidates in an audit artifact and only checked pairs in
   the distributable preference corpus.
6. Split by command family before training so variants of one command cannot
   cross train/validation boundaries.

## Proposed checked-pair contract

Each JSONL record is self-contained and contains:

- `schema_version`, `pair_id`, `source`, `license`, and `provenance`;
- `platform`, `instruction`, and authoritative `context`;
- canonical SemanticDocumentV2 objects `chosen` and `rejected`;
- one or more closed `failure_modes`;
- verifier evidence and explicit review status;
- hashes of the candidate model/checkpoint and generation settings when the
  rejected answer came from a model.

Initial failure labels should cover:

- wrong command or subcommand;
- missing, extra, or wrong flag;
- wrong option value or option/value binding;
- changed, missing, or invented operand;
- wrong ordering or separator;
- wrong redirection or pipeline structure;
- producer/consumer framing mismatch;
- unsupported platform spelling;
- ungrounded literal.

The loader must reject identical chosen/rejected documents, unknown fields,
unknown failure labels, incomplete provenance, unreviewed records, duplicate
pair IDs, and records whose chosen document fails the deterministic semantic
checks. A rejected document may be structurally valid while intentionally
failing task intent; verifier evidence must demonstrate the recorded failure.

## First training experiments

Start with rejection-sampling SFT: keep reviewed verified winners and mix them
with the complete supervised corpus. This tests whether the generated prompts
and chosen answers add useful signal without introducing a preference optimizer.

After enough useful near misses exist, train offline DPO on checked
`(prompt, chosen, rejected)` pairs. Keep the supervised objective through replay
or use a frozen-reference/KL constraint. Select checkpoints using preference
validation plus the existing supervised held-out loss; release and retention
remain finalist gates, and shadow remains a single final gate.

Compare against an SFT-only control containing the same chosen answers. That
control separates the value of preference ordering from the value of simply
adding more correct examples.

## Reporting

Every preference run records pair-corpus hashes, command-family splits, failure-
mode counts, teacher/candidate provenance, SFT replay ratio, preference beta or
equivalent strength, reference checkpoint, loss curves, and verifier pass rates.
Plots must show the SFT-only control beside preference training and must not use
shadow results as a tuning series.
