# ShellIQ bottleneck diagnosis preregistration v1

Date: 2026-08-23

Status: Complete; diagnosis stopped at the registered seed-confirmation boundary.

## Objective

Identify whether current command-disjoint behavior is limited primarily by unique-data
diversity, LoRA capacity, base-model capacity, or optimization/target format. This is a
diagnostic study, not a release-selection exercise.

Release, semantic-shadow-release, and pipeline-confirmation suites remain untouched.
DPO, replay-ratio sweeps, full-parameter tuning, and unregistered model sizes are outside
this study.

## Fixed primary measurements

- manually adjudicated task correctness on the frozen 104-case curated-development suite;
- strict grounded-document exact match;
- exact command flag sequence;
- first command;
- paired per-record gains and regressions;
- full validation loss only as a secondary coverage/optimization measurement.

Every selectable checkpoint must have 104/104 parseable JSON and valid semantic-document
envelopes. Tiered retention must have 20/20 JSON and envelopes, at least 19/20 first
commands, 13/20 flag sequences, and 9/20 grounded documents.

A behavioral difference is material at five cases out of 104. Aggregate loss alone cannot
establish a winner.

## Stage 1: exposure

The 1.5B/rank-8/curated-8x run used 28,248 presentations and retained steps 2,000, 4,000,
8,000, 12,000, and 14,124. Step 12,000 was selected within the 0.005 loss tolerance:
100 first commands, 40 flag sequences, 28 grounded documents, 104/104 contract, and a
passing 19/13/10 retention result. Epoch completion regressed behavior. Exposure is now a
controlled selection variable, not an open-ended tuning axis.

## Stage 2: unique-data scaling

Compare 159, 319, and 637 unique curated rows. Hold the following fixed:

- 5,096 curated presentations and 28,248 total presentations;
- 1.5B base, rank 8, alpha 4;
- sequence length, target modules, seed, split, shuffle policy, schedule, and checkpoints.

For each point, restrict selection to checkpoints within 0.005 of that point's minimum
full validation loss which also pass contract and retention. Select by adjudicated
correctness, grounded match, flags, first command, then loss.

Add the reviewed 720-example/200% point only if 100% beats 50% by at least five adjudicated
correct or strict-grounded cases. Otherwise stop data expansion and advance to capacity.

## Stage 3: controlled capacity matrix

On the selected data recipe, compare exactly:

| Base | Rank | Alpha |
| --- | ---: | ---: |
| Qwen2.5-Coder 0.5B | 8 | 4 |
| Qwen2.5-Coder 0.5B | 32 | 16 |
| Qwen2.5-Coder 1.5B | 8 | 4 |
| Qwen2.5-Coder 1.5B | 32 | 16 |

Hold unique data, presentation count, order, schedule, target modules, evaluation
checkpoints, and `alpha/r = 0.5` fixed. Do not expand the matrix.

Interpretation:

- rank 32 improves training fit and command-disjoint behavior materially: LoRA ceiling;
- training fit is already good and rank 32 does not improve development: data/generalization;
- 1.5B materially leads 0.5B at both ranks: base capacity;
- neither model/rank fits training behavior: optimization, target format, or truncation.

## Stage 4: seed confirmation

Advance at most two configurations. Each must have three total seeds. A conclusion must
hold on at least two seeds, retain paired per-record evidence, and pass contract and
retention independently. Do not average a contract or retention failure away.

## Terminal stopping rules

Stop the diagnosis when evidence supports one of data diversity, LoRA capacity,
base-model capacity, optimization/format, or no material improvement within the registered
matrix. If no candidate clears all gates, report that no promotable recipe was found.
Do not respond by adding another sweep.

Even a successful diagnosis stops before release evaluation. A separately preregistered
finalist decision is required before touching release or shadow data.

## Result

The registered stages are complete. The existing-pool unique-data curve did not have a
positive behavioral slope: the selectable 25% and 100% points both reached 28/104 strict
grounded documents and 40/104 exact flag sequences. The 50% point had no checkpoint that
passed every contract, loss, and retention gate, so the conditional 200% point was not
triggered.

The exact four-cell capacity matrix advanced 1.5B/rank 8 and 1.5B/rank 32 to seed
confirmation. Across seeds 2026, 2027, and 2028, rank 32 won the strict grounded-first
ordering three times and met the preregistered five-case materiality threshold twice.
Every selected checkpoint passed 104/104 development contract and its independent
20-case retention gate. The confirmed configuration is therefore 1.5B/rank 32 with
alpha 16.

This confirms a replicated rank effect, but not a pure training-fit adapter ceiling:
rank 32 did not clearly improve the fixed training probe over rank 8. The controlled
matrix also found a material 1.5B advantage over 0.5B at rank 32, but the 0.5B/rank-8
cell had no selectable checkpoint, so a base-size effect at both ranks is not established.

The diagnosis stops here. No release, shadow, pipeline-confirmation, DPO, replay-ratio,
full-parameter, larger-base, or additional rank experiment is authorized by this study.
