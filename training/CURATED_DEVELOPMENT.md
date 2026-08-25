# Curated development benchmark

`curated-development-v1` is the checkpoint-selection benchmark for fully
specified command requests. It is development data, not training, release,
retention, pipeline confirmation, or shadow data.

The previous full-corpus checkpoint selection used a 256-row validation sample
containing 251 TLDR rows and five curated rows. Its token loss improved while
release and retention behavior regressed, so it is retained only as the broad
coverage signal.

## Freeze contract

The suite remains `draft` until it has at least 100 reviewed records across at
least 20 command families. Every record must:

- be fully specified and require at least two independently checkable
  constraints;
- use the dedicated `shelliq-curated-development` source;
- belong to a command family absent from curated SFT data and from release,
  retention, pipeline, and shadow suites;
- have no exact normalized instruction or target duplicate in those datasets;
- have a closed grounding annotation and metadata entry;
- exercise reviewed command evidence rather than an unspecified literal;
- remain permanently excluded from SFT and preference training.

Required coverage dimensions are flag selection, option values, operand
binding, ordering, and pipeline semantics. Clarification cases use a separate
product-contract evaluation and cannot enter this suite.

Run the deterministic audit with:

```sh
uv run --frozen python scripts/audit_curated_development.py
```

After the manifest is frozen, build the ignored semantic artifact and score all
records with the existing evaluator:

```sh
../target/release/convert-semantic-dataset \
  evaluation/curated-development-v1.jsonl \
  artifacts/curated-development-v1.jsonl \
  artifacts/curated-development-v1.semantic-manifest.json

uv run --frozen python scripts/evaluate_semantic_checkpoint.py \
  --semantic-dataset artifacts/curated-development-v1.jsonl \
  --selection all \
  --sequence-length 448 \
  --checkpoint CHECKPOINT \
  --grounding-audit evaluation/curated-development-grounding-v1.json \
  --output REPORT \
  --rank RANK \
  --alpha ALPHA
```

## Checkpoint selection

For every checkpoint interval, record broad held-out loss and this suite's
semantic metrics. A finalist must be Pareto-competitive on both signals. Broad
loss may be at most `0.005` above the best broad checkpoint in the run. Within
that set, maximize grounded-document exact match and use exact command/flag
sequence as the tie-breaker. Require an improvement of at least five grounded
records out of the frozen minimum of 100 before calling the change material.

Release and retention are behavioral gates applied only to preregistered
finalists. Pipeline confirmation and shadow remain untouched until one
candidate passes those gates. DPO is out of scope until an SFT checkpoint
improves this benchmark without harming retention.

## Manual adjudication

Strict matching is not a substitute for task adjudication. Prepare a closed worksheet
for the strongest retained 0.5B and 1.5B reports, review every output, then summarize it:

```sh
uv run --frozen python scripts/adjudicate_curated_development.py --prepare \
  --report 0.5b=artifacts/curated-development-v1-evaluation/incumbent.eval.json \
  --report 1.5b=artifacts/curated-development-v1-evaluation/sft-v3-qwen1p5b-allcurated-cont-step500.eval.json \
  --worksheet artifacts/curated-development-v1-adjudication-v1.json

# After recording an explicit decision for every non-exact row:
uv run --frozen python scripts/adjudicate_curated_development.py --summarize \
  --worksheet artifacts/curated-development-v1-adjudication-v1.json \
  --decisions evaluation/curated-development-adjudication-decisions-v1.json \
  --output artifacts/curated-development-v1-adjudication-v1.summary.json
```

Allowed failure modes are semantic alternative, wrong command, missing/wrong flags,
operand binding, ordering, pipeline structure, JSON contract, and truncation. Corrections
found during review must go into a separately versioned v2 suite; never edit frozen v1.

## Results

The first completed checkpoint-selection and model-capacity study is documented
in `experiments/curated-development-capacity-study-v1.md`. It records the 0.5B
plateau, the reason for testing 1.5B, all comparable development and gate scores,
and the current no-promotion decision.
