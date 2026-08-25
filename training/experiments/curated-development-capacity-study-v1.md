# Curated-development model-capacity study v1

Date: 2026-08-22

Status: Bottleneck diagnosis complete; 1.5B/rank 32 confirmed, release untouched.

## Why we ran this study

Earlier training often improved TLDR-heavy held-out loss without passing the
behavioral release and retention gates. We changed the selection process before
changing model size:

1. Freeze a command-family-disjoint development suite for fully specified,
   multi-constraint behavior.
2. Re-score retained 0.5B checkpoints before training anything new.
3. Rebuild the broad corpus and increase diverse, grounded curated supervision.
4. Select SFT checkpoints with both broad loss and curated-development behavior.
5. Use retention and release only as behavioral gates.
6. Test a larger model only if aligned 0.5B experiments plateau.

The clarification/result-state contract remains useful parallel product work.
It protects training data from ambiguous requests, but cannot by itself fix
fully specified flag and pipeline failures.

## What changed

### Frozen curated-development suite

`training/evaluation/curated-development-v1.jsonl` contains 104 cases across 34
command families. It is development-only: it is not release, retention, shadow,
SFT, pipeline-confirmation, or preference data. Its manifest and audit tools
verify provenance and prevent prompt leakage into training data.

### Rebuilt v3 SFT corpus

The raw merge contains 31,137 examples: 30,351 broad TLDR and 786 curated.
Semantic filtering accepted 30,572 examples covering 5,085 commands; 349
CST-invalid and 216 semantic-invalid examples were excluded.

Corpus SHA-256:
`284f67e77913d2676598d80b323876a2fb0f15ce93ce3ae36fb5a4e3f371f1b4`

The merge was fixed to exclude `reviewed-preference-*.jsonl`, keeping DPO
preferences out of SFT data.

### Training and evaluation support

The tooling now supports:

- initializing a fresh optimizer from an adapter checkpoint;
- multiple non-overlapping `PREFIX=WEIGHT` rehearsal slices;
- retaining checkpoints at evaluation intervals;
- choosing the 0.5B or 1.5B Qwen2.5-Coder model by model ID;
- loading architecture values from the official Qwen2 configuration; and
- reusing exact, model-matched base generations during semantic evaluation.

Interval retention matters because GPU training was not bitwise reproducible
across separate processes with the same nominal seed. A discarded intermediate
checkpoint cannot be reconstructed reliably afterward.

## Why we switched to 1.5B

The switch was not based on broad loss alone. We first gave the 0.5B model the
aligned selector and current data it had been missing.

Retained 0.5B checkpoints scored 8-15 exact grounded documents on the new suite.
Targeted, intent-only, mixed, broad-curated, low-rate continuation, and
checkpoint-interpolation experiments then plateaued. The best balanced 0.5B
result reached 17/104 grounded documents and 29/104 exact flag sequences, below
the declared material-improvement threshold of 19 grounded documents. It also
regressed on release from 12 to 11 exact flag sequences and from 10 to 9 grounded
documents.

Only after that plateau did capacity become the next justified variable. The
official 1.5B base loaded successfully and passed a bounded GPU training smoke
test before SFT runs began.

## Results

Counts are exact cases, not percentages. `JSON/env` means parseable JSON and a
valid `SemanticDocumentV2` envelope.

### Curated development (104 cases)

| Checkpoint                                   | Full v3 loss | First command |  Flags | Grounded | JSON/env |
| -------------------------------------------- | -----------: | ------------: | -----: | -------: | -------: |
| 0.5B retained source                         |     0.260057 |            89 |     22 |       14 |    98/98 |
| 0.5B selected continuation                   |     0.260727 |            89 |     29 |       17 |  101/101 |
| 1.5B, curated 8x, step 2,000                 |     0.222561 |            90 |     39 |       26 |  104/103 |
| 1.5B, 8x plus 500-step low-rate continuation |     0.224978 |        **98** | **43** |   **28** |  104/103 |
| 1.5B, curated 8x replay, step 1,500          |     0.225764 |            95 |     41 |       27 |  101/100 |
| 1.5B, curated 4x, step 2,000                 | **0.221062** |            86 |     35 |       22 |  103/103 |
| 1.5B, curated 2x, step 2,000                 |     0.221838 |            95 |     36 |       24 |  101/101 |

The strongest 1.5B checkpoint doubles grounded matches from 14 to 28, raises
exact flag sequences from 22 to 43, and raises first-command accuracy from 89 to 98. The best broad-loss checkpoint lowers loss from 0.260057 to 0.221062,
approximately 15%.

Replay dose changes the tradeoff rather than producing one monotonic winner: 8x
gives the strongest curated behavior, while 4x and 2x give slightly better broad
loss but weaker curated behavior.

### Retention gate (20 cases)

The no-harm reference is the retained 0.5B source: 20 first commands, 14 exact
flag sequences, and 10 grounded documents.

| Checkpoint                          | First command |  Flags | Grounded | JSON/env | Result    |
| ----------------------------------- | ------------: | -----: | -------: | -------: | --------- |
| 0.5B retained source                |            20 |     14 |       10 |    20/20 | Reference |
| 0.5B selected continuation          |            20 | **16** |       10 |    20/20 | Pass      |
| 1.5B, curated 8x, step 2,000        |            19 |     13 |       10 |    20/20 | Fail      |
| 1.5B, 8x plus 500-step continuation |            19 |     13 |       10 |    20/20 | Fail      |
| 1.5B, curated 8x replay, step 1,500 |            19 |     13 |       10 |    20/20 | Fail      |
| 1.5B, curated 4x, step 2,000        |            18 |     13 |       10 |    19/19 | Fail      |
| 1.5B, curated 2x, step 2,000        |            19 |     11 |        9 |    20/20 | Fail      |

Every evaluated 1.5B finalist harms retention. Stopping earlier, adding a short
low-rate continuation, or reducing curated replay did not repair the regression.

### Release gate (35 cases)

Release was evaluated only for the original selected 1.5B finalist. Later
candidates stopped at retention because that gate failed first.

| Checkpoint                   | First command |  Flags | Grounded |  JSON/env | Result |
| ---------------------------- | ------------: | -----: | -------: | --------: | ------ |
| 0.5B retained source         |            32 |     12 |       10 |     33/33 | Fail   |
| 0.5B selected continuation   |            32 |     11 |        9 |     33/33 | Fail   |
| 1.5B, curated 8x, step 2,000 |        **34** | **13** |   **12** | **34/34** | Fail   |

The promotion gate requires 35/35 JSON envelopes, at least 31 first commands,
more than 13 exact flag sequences, and more than 7 grounded documents. The 1.5B
checkpoint improved three behavioral counts, but one max-length `cargo bench`
repetition made an output invalid and 13 flags did not clear the strict floor.

Pipeline confirmation and shadow were not run because no candidate passed the
preceding gates.

## Is anything improving anything?

Yes, in three meaningful ways:

1. Command-family-disjoint generalization improves materially: the best grounded
   score rises from 14/104 to 28/104.
2. Broad modeling improves: the best full-v3 validation loss is about 15% lower.
3. Release behavior moves in the right direction: the tested 1.5B finalist moves
   from 32/12/10 to 34/13/12 on first command, flags, and grounding.

The problem is composition. No checkpoint combines those gains with the
retention no-harm requirement and perfect output-contract reliability. This is
real progress in diagnosis and capability, but not yet a deployable model.

## Current decision

- Do not promote, export, or serve a checkpoint from this study.
- Do not revisit DPO yet; no SFT checkpoint both improves curated development
  and passes retention and release.
- Do not continue the same replay-ratio sweep; it has shown the tradeoff.
- The next experiment should add training-only, command-disjoint development
  proxies for common-behavior retention and JSON/envelope reliability.
- Keep release and retention prompts excluded from training and selection.
- Treat constrained semantic decoding as a separate product decision, not as a
  substitute for grounded model behavior.

## Preregistered diagnosis follow-up

The earlier 2,000-step runs saw only 4,000 presentations from an effective 28,248-row
epoch. They do not establish a training plateau. The immediate follow-up therefore holds
the 1.5B base, rank 8, `alpha/r = 0.5`, split, seed, target modules, and curated 8x mixture
fixed while extending exposure to exactly 14,124 batch-2 steps.

`scripts/run_complete_exposure_study.py` creates an immutable manifest, asserts the
expected 23,790 unique training rows, 637 unique curated training rows, and 28,248
effective presentations before training, and retains steps 2,000, 4,000, 8,000, 12,000,
and 14,124. Each checkpoint receives curated-development evaluation, paired per-example
analysis, full validation loss, and the tiered common-shell retention gate. Release and
shadow suites are excluded.

```sh
uv run --frozen python scripts/run_complete_exposure_study.py
uv run --frozen python scripts/run_complete_exposure_study.py --execute
```

The dry run prints the full preregistration without writing. Execution is intentionally
separate because the run requires a local cached 1.5B model and a CUDA JAX device.

## Complete-exposure result

The preregistered run completed all 14,124 steps (28,248 batch-aligned presentations)
without early stopping. The fixed 256-row training probe fell from 2.08095 to 0.09572.
The broad full-validation loss reached its minimum at step 12,000 and rose slightly by
epoch completion.

| Step | Full validation loss | First command | Flags | Grounded | JSON/env | Retention first/flags/grounded | Retention gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2,000 | 0.2253776 | 79 | 39 | 27 | 103/102 | 18/12/8 | Fail |
| 4,000 | 0.2139987 | 93 | 33 | 21 | 104/104 | 19/14/12 | Pass |
| 8,000 | 0.2074246 | 91 | 39 | 25 | 103/103 | 20/14/11 | Pass |
| 12,000 | **0.2067038** | **100** | **40** | **28** | **104/104** | 19/13/10 | Pass |
| 14,124 | 0.2079367 | **100** | 37 | 24 | **104/104** | 20/14/11 | Pass |

Step 12,000 is the exposure-study diagnostic winner. Relative to this run's step 2,000,
it has 24 paired improvements and 7 regressions. Relative to the earlier strongest 1.5B
low-rate continuation, it ties grounded exact match at 28/104, improves first command
from 98 to 100 and the contract from 104/103 to 104/104, but reduces exact flags from
43 to 40; paired comparison finds 10 improvements and 12 regressions.

The experiment rejects a simple claim that training behavior had fully plateaued at
2,000 steps: command selection, contract reliability, retention, and broad loss all
improved with further exposure. It does not show further grounded-task improvement over
the prior best 1.5B result, and the epoch endpoint regresses from step 12,000. Additional
exposure alone is therefore not the next optimization axis. Proceed to the preregistered
unique-data scaling curve before a LoRA-rank or larger-base experiment.

This checkpoint is not a release candidate, the result has one seed, and release/shadow
evaluation remains untouched. The closed adjudication is complete: the 0.5B checkpoint
is task-correct on 39/104 cases (17 strict grounded), while the 1.5B step-12,000
checkpoint is task-correct on 59/104 (28 strict grounded). All 163 non-exact outputs
were explicitly reviewed; semantic alternatives account for 22 additional correct 0.5B
outputs and 31 additional correct 1.5B outputs. The immutable worksheet, versioned
decisions, and generated summary are:

- `artifacts/curated-development-exposure-adjudication-v1.json`
- `evaluation/curated-development-exposure-adjudication-decisions-v1.json`
- `artifacts/curated-development-exposure-adjudication-summary-v1.json`

## Unique-data scaling preregistration

`scripts/run_unique_data_scaling.py` holds total presentations (28,248), curated
presentations (5,096), schedule, order, seed, model, rank, and evaluation checkpoints
constant while varying unique command-family-stratified curated rows. The 25% point has
159 curated rows (selection SHA-256
`5e0f6935470b13de3931ecb6e5b9430fe487588bbd6af075337d638696831dbf`); the 50% point
has 319 (`6ca95c26ebff9e69c29043079161df724c5b72abe4cb289bc05a193fe80623ca`).
The completed 637-row exposure study supplies the 100% point.

```sh
uv run --frozen python scripts/run_unique_data_scaling.py --execute --resume
```

The curve must be interpreted on command-disjoint curated development, not TLDR-heavy
loss alone. If the behavioral slope is still positive at 100%, add the preregistered 720
reviewed diverse examples before running a 200% point. Otherwise proceed directly to the
0.5B/1.5B by rank-8/rank-32 capacity matrix.

## Authoritative local artifacts

- Curated-development reports:
  `training/artifacts/curated-development-v1-evaluation/`
- Initial 1.5B 8x run:
  `training/artifacts/sft-v3-qwen1p5b-rank8-allcurated-w8-cosine-s2000-v1/`
- Low-rate 1.5B continuation:
  `training/artifacts/sft-v3-qwen1p5b-rank8-allcurated-w8-cont-lr1e5-s500-v1/`
- Interval-retained 8x replay:
  `training/artifacts/sft-v3-qwen1p5b-rank8-allcurated-w8-cosine-s2000-intervals-v1/`
- Interval-retained 4x run:
  `training/artifacts/sft-v3-qwen1p5b-rank8-allcurated-w4-cosine-s2000-intervals-v1/`
- Interval-retained 2x run:
  `training/artifacts/sft-v3-qwen1p5b-rank8-allcurated-w2-cosine-s2000-intervals-v1/`

Artifact directories are local ignored experiment output. The frozen suite,
manifests, audit code, training support, and this report are the version-controlled
deliverables.

## Completed bottleneck diagnosis

The preregistered follow-up is complete. It separates exposure, unique-data count, base
size, and LoRA rank while preserving the same corpus, presentation count, order,
schedule, target modules, selection checkpoints, and `alpha/r = 0.5` where required.

### Unique-data curve

The curve held 5,096 curated and 28,248 total presentations fixed while varying unique
curated rows.

| Unique curated rows | Selected step | First command | Flags | Grounded | Retention | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 159 (25%) | 8,000 | 98 | 40 | 28 | 19/13/10 | Pass |
| 319 (50%) | - | - | - | - | - | No checkpoint passed every gate |
| 637 (100%) | 12,000 | 100 | 40 | 28 | 19/13/10 | Pass |

The selectable endpoints are flat on exact flags and grounded documents. The 50% run's
best-looking development checkpoint reached 103/43/28 but failed retention at 19/12/7.
Broad validation loss improved with more unique rows, but command-disjoint behavior did
not. The conditional reviewed 720-example/200% point was therefore not triggered. This
rules out simple scaling of the existing pool as the evidenced next move; it does not
rule out genuinely new constraint and command-family coverage.

### Exact capacity matrix

All four cells used 28,248 presentations, batch size 2, 14,124 steps, the same training
and split seeds, the same example order and schedule, and the same corpus hash. Selection
required 104/104 development contract, passing tiered retention, and validation loss
within 0.005 of the cell minimum.

| Configuration | Step | First command | Flags | Grounded | Retention | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5B/rank 8 | - | - | - | - | - | No selectable checkpoint |
| 0.5B/rank 32 | 12,000 | 100 | 45 | 27 | 20/16/11 | Pass |
| 1.5B/rank 8 | 12,000 | 100 | 40 | 28 | 19/13/10 | Pass |
| 1.5B/rank 32 | 8,000 | 96 | 49 | 33 | 20/16/14 | Pass |

At rank 32, moving from 0.5B to 1.5B adds six grounded cases and four exact-flag cases
while losing four first-command cases. Moving the 1.5B model from rank 8 to rank 32 adds
five grounded and nine exact-flag cases while losing four first-command cases. Both
grounded differences meet the five-case materiality threshold. The 0.5B/rank-8 gate
failure prevents a complete base-size conclusion at both ranks.

Training fit does not support a simple adapter-underfit story. On the fixed probe, the
selected 1.5B/rank-32 checkpoint has train/held-out losses 0.14856/0.22795, versus
0.14345/0.22874 for 1.5B/rank 8. Rank 32 materially changes command-disjoint behavior,
but does not clearly lower training-fit loss.

### Seed confirmation

The two 1.5B leaders were compared on three total training seeds. Every selected
checkpoint independently passes 104/104 development contract and the 20-case retention
gate.

| Seed | Rank | Step | First command | Flags | Grounded | Retention |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026 | 8 | 12,000 | 100 | 40 | 28 | 19/13/10 |
| 2026 | 32 | 8,000 | 96 | 49 | 33 | 20/16/14 |
| 2027 | 8 | 12,000 | 101 | 49 | 33 | 19/14/11 |
| 2027 | 32 | 12,000 | 102 | 49 | 34 | 19/15/11 |
| 2028 | 8 | 8,000 | 99 | 42 | 27 | 20/14/12 |
| 2028 | 32 | 12,000 | 102 | 49 | 33 | 19/14/11 |

Using rank 8 as the baseline, rank 32 changes first-command/flags/grounded counts by
`-4/+9/+5`, `+1/0/+1`, and `+3/+7/+6` on seeds 2026, 2027, and 2028. It wins the strict
grounded-first ordering on all three seeds and has material wins on 2026 and 2028. Paired
document comparison finds rank-32 improvements/regressions of 13/9, 7/7, and 15/6.
The registered two-of-three confirmation rule therefore selects 1.5B/rank 32.

### Adjudicated correctness

The closed 208-output adjudication covers the preregistered best 0.5B and full-exposure
1.5B/rank-8 checkpoints. Human task correctness rises from 39/104 to 59/104 while strict
grounded match rises from 17/104 to 28/104. All 163 non-exact outputs have explicit
decisions. The rank-32 seed comparison was not retrospectively human-labelled; its
conclusion uses the frozen strict metrics and paired per-record evidence specified for
seed confirmation.

### Diagnosis and stopping decision

The replicated evidence supports LoRA rank as a material performance lever for the 1.5B
recipe. It does not prove that rank 8 failed to fit training data, so the narrow claim is
a confirmed rank effect rather than a pure adapter-fitting ceiling. Base-model capacity
also matters at rank 32 on the controlled seed, but was not established at both ranks.
Neither longer exposure nor more unique rows from the current curated pool produced a
grounded scaling slope.

Stop this diagnosis. Do not add another rank, model size, replay ratio, data fraction, or
DPO run. The evidence-backed next action is a separate preregistered finalist decision
for 1.5B/rank 32, with seed-2027 step 12,000 as the strongest observed checkpoint at
102/49/34 and passing 19/15/11 retention. Only that separate decision may authorize
release evaluation. Shadow and pipeline-confirmation remain downstream of their existing
gates. Despite the confirmed rank effect, 34/104 strict grounded match is not by itself a
compelling product result.

Authoritative diagnosis artifacts:

- `artifacts/unique-data-scaling-v1/analysis.json`
- `artifacts/capacity-matrix-v1/analysis.json`
- `artifacts/capacity-seed-confirmation-v1/analysis.json`
- `artifacts/curated-development-exposure-adjudication-summary-v1.json`
