# Historical model-development handoff

> This August 2026 checkpoint records superseded generative-model experiments.
> It is retained for research provenance, not as current setup guidance. For the
> supported optional ranker, use `shelliq model doctor` and see
> [docs/ALPHA_RELEASE.md](docs/ALPHA_RELEASE.md).

## Current checkpoint — 2026-08-21 preference pilot

### Expanded preference corpus v2 pilot complete

The next data pass evaluated the rank-8 baseline on all 280 eligible curated
training records allowed by the four-per-command diversity cap. Rank 8 emitted
valid SemanticDocumentV2 for 271/280, selected the correct first command for
246/280, and exactly grounded 47/280. Static harvesting retained 190 valid,
correct-command, non-equivalent near misses.

Manual Codex review reused 59 prior decisions and adjudicated 131 new cases.
The complete result is 99 accepted preference pairs: 72 train and 27 command-
family-disjoint validation pairs. Of 190 reviewed candidates, 93 were excluded:
46 task-equivalent, 41 dependent on underspecified operands, 4 ambiguous, and
2 bad existing targets. The exporter explicitly carried forward two accepted
v1 pairs that fell outside the expanded pool's command cap, so v2 is a superset.

Local Qwen was used only to order review. Calibration against 64 completed
manual decisions gave 82.8% precision and 75% recall for its `chosen` label; it
missed 8 useful pairs and falsely preferred 5 hidden-operand or ambiguous pairs.
No Qwen prescreen label was accepted or rejected automatically.

`training/corpus/reviewed-preference-v2.jsonl` and the complete new decision
ledger are committed in `e35815c`. The fixed v1 recipe completed on
v2: rank 8 / alpha 4, 3 epochs (216 updates), LR `1e-5`, beta `0.1`, replay
weight `0.2`, 128
curated replay examples, matched chosen-SFT and cached-reference DPO conditions.
The expanded DPO run produced the first positive command-family-held-out
preference signal: 15/27 reference-relative wins and mean reference-margin
delta `+1.470`. It did not improve behavior. Release-development preserved
32/35 first command and 12/35 flags but fell from 10/35 to 9/35 grounded.
Retention fell from 20/20 to 19/20 first command, 14/20 to 13/20 flags, and
10/20 to 9/20 grounded. It fails the retention gate and is not promotable.

Chosen-only SFT again overfit, falling to 28/35 first command, 9/35 flags, and
6/35 grounded on release and 14/20, 10/20, and 5/20 on retention. Exact results
and case-level changes are in `training/experiments/preference-pilot-v2.md`.
Shadow was not evaluated.

Verifier-backed preference plumbing is implemented. The first manually
reviewed corpus contains 32 accepted rank-8 near-miss pairs from 64 candidates;
the other half were excluded rather than teaching arbitrary operands or
equivalent spellings. See `training/experiments/preference-pilot-v1.md` and
`training/VERIFIED_TEACHER_DATA.md`.

The matched pilot used the rank-8 step-10,000 baseline, cached reference scores
(no second resident model), 128-example curated replay, 25 command-family-
disjoint train pairs, and 7 validation pairs. DPO exactly retained 14/20 flag
and 10/20 grounded retention, but did not improve release (12/35 flags, 10/35
grounded) and introduced one `readlink-canonical` wrong-command regression.
Chosen-answer SFT collapsed behavior and is rejected. The shadow suite was not
evaluated and nothing is promotable.

Do not tune DPO hyperparameters on 32 pairs. The next task is expanding and
manually reviewing diverse rank-8 near misses, then repeating the same matched
SFT/DPO control. The Qwen generation cascade is diagnostic only: exact-match
failures largely reflect generic prompts with hidden concrete target operands.
Use teachers to adjudicate supplied pairs or author fully grounded triples.

Updated 2026-08-20.

### One-epoch SFT coverage result

Rank-8 full-partition SFT at a lower `1e-4` peak LR reached every 23,646 frozen
training row once and improved fixed corpus validation loss from `0.26980` to
`0.26548`. It regressed behavior: release 33/35 first command but 9/35 flags
and 6/35 grounded; retention 20/20 first command, 13/20 flags, 8/20 grounded.
It is not promotable. This rules out the simple explanation that the incumbent
only needed the remaining 15% of its training-partition rows. Details:
`training/experiments/lora-rank8-coverage-lr1e-4-v1.md`.

### Preference pilot v3 stop decision

The next 28 manually reviewed, explicit-target contrasts were added to v2,
creating the 127-pair `training/corpus/reviewed-preference-v3.jsonl`. Matched
SFT and cached-reference DPO used fresh broad-corpus replay at every update
rather than cycling 128 examples. DPO again improved held-out preference
margins (18/31, `+1.910`) but not command behavior: release remained 32/35
first command and 12/35 flags while grounding fell to 9/35 and JSON/envelope
fell to 32/35; retention was 20/20 first command, 13/20 flags, and 9/20
grounded. SFT regressed much further. Neither is promotable; shadow was not
evaluated. See `training/experiments/preference-pilot-v3-stream.md`.

Stop DPO/SFT preference hyperparameter, rank, replay-shape, and more-of-the-
same harvested-pair sweeps. They are improving the pairwise surrogate without
a retention-safe behavioral gain. The next model decision is architectural:
clarification/result state to remove inherently unanswerable commands, then
reconsider the target representation or base-model capacity for fully specified
multi-constraint commands.

## Planned clarification and execution-state contract

The current `context-authoritative-v1` contract requires an executable
`SemanticDocumentV2`, so it cannot correctly answer an underspecified request
with a question. This is a product and data-quality gap, not evidence that the
model should guess operands. Add a versioned response envelope with two safe
outcomes: a command document or one focused clarification question. The shell
widget must display a clarification and never put it in the editable command
buffer.

Follow-ups should use compact, typed execution provenance rather than an
unbounded chat or shell-history scrape. After a user-run command successfully
produces one recognized path, a later request such as "how many lines are in
that file?" may use that path. A suggested-but-not-run command, a failed
command, or multiple candidate paths must cause clarification instead.

This work improves safety and lets us retain otherwise underspecified examples
as clarification data, but it cannot fix wrong flags or invalid pipelines in a
fully specified request. Do not make it the next LoRA tuning axis. First build
and evaluate the fully grounded controlled-contrast pack and conservative
full-corpus mixture described in `training/experiments/preference-pilot-v2.md`.
Then implement the clarification contract as a small vertical slice with its
own held-out evaluation; do not mix its results into command-quality metrics.

## Current goal

Improve the model so it does not produce functionally invalid pipelines such as:

```sh
find . -type f -size +1G -print0 | sort -nr
```

This command passed option-spelling validation, but `find -print0` emits
NUL-delimited filenames with no size column while `sort -nr` expects
newline-delimited numeric input.

The attempted fix that hard-coded `find | sort` compatibility rules in the Rust
verifier was rejected and reverted. The verifier deliberately checks indexed
command and option facts; it does not claim that operands or combinations of
valid flags implement the user's intent. Fix this failure through model training
and evaluation rather than a command-pair special case. Do not loosen fail-closed
option validation to make model output pass.

## Installed local alpha

- `~/.local/bin/shelliq` points to the repo-owned `shell/shelliq-alpha`, which
  runs `target/release/shelliq` and pins the focused alpha index.
- Focused index: `~/.local/share/shelliq/alpha-index.sqlite`, containing `cp`,
  `grep`, `find`, `sort`, `tar`, `curl`, and `ls`.
- Comprehensive index: `~/.local/share/shelliq/index.sqlite` (2,443 commands,
  41,748 flags at the last check).
- Model: ignored `training/artifacts/shelliq-semantic-alpha-q8_0.gguf`;
  `training/artifacts/current.gguf` points to it.
- User service: `shelliq-model.service`, enabled and bound only to
  `127.0.0.1:8080`.
- Zsh integration is sourced from `~/.zshrc`.
- Normal terminal widget: `Ctrl-X Ctrl-G`.
- VS Code terminal sends the widget sequence with `Ctrl-Alt-G` through the user
  `keybindings.json`.
- The widget's read-only `status` bug was fixed by renaming it `exit_status`.

## Relevant commits

- `005f50f` — grounded two-pass alpha suggestions
- `69cbf39` — local alpha runtime wiring
- `80f36f9` — preload suggestions into the zsh buffer
- `9eb25e5` — portable zsh widget fix

## Verified alpha baseline

The formal focused-index smoke gate passed:

- shortlist recall: 5/5
- locally validated end-to-end suggestions: 5/5
- unsupported-task abstention: passed
- RTX 4090 latency: 164 ms mean, 239 ms maximum
- editable widget: works in VS Code

The prompt `find files larger than one gigabyte under the current directory`
produced and successfully ran:

```sh
find . -type f -size +1G -print
```

The current release checkpoint is documented in `docs/ALPHA_RELEASE.md` as
`training/artifacts/semantic-authoritative-curated-v3/checkpoint`, step 2500,
using prompt contract `context-authoritative-v1`.

## Pipeline tuning completed 2026-08-20

- Added 12 reviewed training rows in `training/corpus/pipeline-contracts.jsonl`.
- Added an eight-row frozen held-out suite in
  `training/evaluation/pipeline-compatibility-v1.jsonl`, with a strict grounding
  audit and tests preventing exact instruction or target leakage.
- The incumbent checkpoint scored 1/8 exact command/flag sequences and 0/8
  strict documents on the pipeline suite.
- A 100-step candidate at learning rate `2e-6`, with the new rows weighted 4x
  among 600 broad rehearsal rows, corrected the original failure and improved
  pipeline results to 3/8 flag sequences and 1/8 strict documents. It also
  improved retention from 14/20 to 15/20 flag sequences and from 8/20 to 9/20
  strict documents, with no retention regression.
- Continuing the candidate to 200 steps improved pipeline flag sequences to
  4/8, but both checkpoints plateaued on the release suite at 12/35 exact
  command/flag sequences and 7/35 grounded documents.
- The fixed promotion gate requires more than 13/35 flag sequences and more
  than 7/35 grounded documents. Both candidates are therefore rejected and
  must not be exported, linked as `current.gguf`, or served as the alpha model.

Artifacts and reports are under:

```text
training/artifacts/pipeline-compatibility-incumbent-v1.eval.json
training/artifacts/pipeline-refine-lr2e6-s100-w4-v1/
training/artifacts/pipeline-refine-lr2e6-s100-w4-v1.*.json
training/artifacts/pipeline-refine-lr2e6-s200-w4-v1/
training/artifacts/pipeline-refine-lr2e6-s200-w4-v1.*.json
```

## Broader curriculum result

- Added 48 balanced rows in `training/corpus/intent-fidelity.jsonl`: four
  contrasts for each of 12 command families disjoint from release and retention.
- The candidate used `curated-semantic-v6`. A post-run local-help audit corrected
  two unsupported zstd long-option spellings to portable `-o` and `-T0`; use the
  regenerated `curated-semantic-v7` for future work: 786/786 rows, zero
  exemptions. The rejected candidate is not being re-scored after this data fix.
- A 100-step, `2e-6` candidate used 636 broad rows plus 2x exposure for the 36
  new rows in its training split. Release stayed at 12/35 exact flag sequences
  and 7/35 grounded documents, so it was rejected without consulting the
  pipeline holdout.
- Retention passed at 15/20 flag sequences and 9/20 grounded documents versus
  incumbent 14/20 and 8/20. The broader data is useful and should stay, but a
  low-rate continuation still does not cross the release plateau.
- Candidate artifacts are under
  `training/artifacts/intent-pipeline-refine-lr2e6-s100-w2-v1*`. Do not export,
  link as `current.gguf`, or serve them.

## Next work

The completed full-corpus SFT phase selected rank 16 step 10,000 at validation
`0.2664` and rank 8 step 10,000 at `0.2698`. Both degraded after the 11,823-step
epoch boundary while training loss continued falling, so this is overfitting,
not an unfinished run. Rank 16's small margin and 89.9% clipping rate do not
establish a winner. Exact curves are committed in
`training/experiments/lora-full-corpus-finalists-v1.{json,md}`. The active next
phase is `training/scripts/run_full_corpus_evaluation.py`: corpus test loss,
then unchanged release-development and retention gates for both checkpoints.
It is resumable and must not evaluate pipeline or shadow suites.

The retained step-10,000 checkpoints consumed 20,000 rows (84.6% of the first
random permutation). The one-epoch early-stopping floor prevented termination,
not selection of an earlier checkpoint. If behavioral gates are promising,
follow with an exact-step-11,823 coverage check using a gentler LR tail and two
shuffle seeds rather than several ordinary extra epochs.

Behavioral evaluation is complete. Rank 8 scored test loss `0.2591`, release
flags/grounded `12/35` and `10/35`, and retention `14/20` and `10/20`; it passes
retention but fails release. Rank 16 scored test loss `0.2564`, release `13/35`
and `8/35`, and retention `11/20` and `8/20`; it fails both gates. Select rank 8
as the non-promotable SFT starting point. Exact results are committed in
`training/experiments/lora-full-corpus-finalists-v1-evaluation.{json,md}`.
Do not evaluate shadow or pipeline. The next work is verified teacher/rejection
data for missing flags, extra flags, and invalid envelopes, retaining curriculum
replay and the rank-8 retention gate. The initial release analysis files said
`passed: true` only because the analyzer incorrectly skipped gate calculation
without CLI exit mode; the saved metrics were correct and the gate was
recomputed. Gate calculation is now unconditional.

The detailed next-phase plan is `training/VERIFIED_TEACHER_DATA.md`. Pilot with
Codex, use Luna as the default low-cost bulk draft generator, and optionally use
a local model for diversity. Reserve the available Fireworks account and a
stronger hosted reviewer for ambiguous cases, new risky command families,
disagreements, and a fixed random audit sample. All sources use a
provider-neutral JSONL artifact, no generator approves its own records, and
deterministic verification plus human corpus acceptance remain authoritative.
Most rejected candidates should be actual rank-8 samples. Train an SFT-only
chosen-answer control before DPO so data value is separated from
preference-optimizer value.

No generated command is currently authorized for execution. Docker is available
in the user's normal terminal and is the intended backend; only the restricted
Codex process is denied daemon access. The required Docker-backed harness has
not been built. Functional tests eventually require a fresh `zsh -f` inside a
fresh disposable container per candidate, no network/host mounts, Docker socket,
or credentials, one synthetic writable fixture, a pinned image, strict resource
limits, and hostile containment fixtures. Unsafe commands remain static-only.

The just-completed phase was full-corpus SFT, not another subset sweep. It used
`training/scripts/run_full_corpus_finalists.py` and advanced the tied rank-8
(`alpha=4`, peak LR `2e-4`) and rank-16
(`alpha=16`, peak LR `5e-5`) recipes across all 23,646 usable training rows.
Each was allowed at most two reshuffled epochs (23,646 batch-2 steps total), 500-step
warmup plus cosine decay, corpus-validation checks every 1,000 steps with the
corpus test split untouched, and at least one full epoch before early stopping.
The trainer retained one actual best checkpoint per run and removed superseded
generated checkpoints only after the replacement was successfully saved. That
phase forbade release, retention, pipeline, and shadow evaluation.

The short rank/alpha/LR optimization screen has now completed all 30 runs.
Rank 8, `alpha / rank = 1`, learning rate `1e-4` led at held-out loss `0.3371`,
but ranks 4/8/16 were within `0.0037`, so this is not a defensible rank winner.
Run the committed deeper study from `training/` in the user's interactive
terminal so Rich progress is visible:

```sh
UV_CACHE_DIR=/tmp/shelliq-uv-cache \
  uv run --frozen python scripts/run_convergence_study.py --execute
```

It runs nine configurations: ranks 4/8/16 at `5e-5` and `1e-4` with
`alpha / rank = 1`, plus rank-8 ratio/LR diagnostics `(0.5, 2e-4)`,
`(2, 5e-5)`, and `(2, 1e-4)`. Each gets at most
3,000 steps, held-out checks every 250 steps, patience 5/minimum delta `0.002`,
and pre-clipping gradient diagnostics. It is exact-manifest resumable with
`--resume`, saves reports rather than adapters, and forbids release, retention,
pipeline, and shadow evaluation. Afterward, repeat only finalists across seeds
and retrain the selected best-step recipe with a checkpoint. The obsolete
ignored `training/artifacts/lora-rank-sweep-v1` adapters were removed after
their exact results were committed, reclaiming 1.3 GB; incumbents were kept.

The fixed-recipe GPU rank sweep is complete. Exact results and plots are in
`training/experiments/lora-rank-fixed-recipe-v1.{json,md}`. Rank 8 led the tested
8/16/32/64 recipes at release 12/35 flags, 8/35 grounded and retention 14/20
flags, 8/20 grounded, but missed the 13/35 release flag floor. Nothing was
promoted and the shadow holdout was not evaluated. Larger ranks also had worse
training-corpus held-out loss, so do not claim rank 8 is intrinsically optimal:
the experiment held learning rates and step counts fixed instead of tuning each
rank to convergence.

Do not start another broad hyperparameter sweep. The checkpoint-level
rank-4/8/16 convergence study above is the active experiment; select only by
corpus held-out loss and clipping diagnostics. More importantly, build the
verifier-backed teacher-data loop: reviewed rejection-sampling winners become
SFT data; verified useful near misses become `(prompt, chosen, rejected)` pairs
for later offline DPO with supervised replay/reference regularization.

Rank-sweep infrastructure is committed and has completed the fixed-recipe
comparison. `training/scripts/run_rank_sweep.py` dry-runs by default and uses
constant `alpha / rank = 2`, identical broad then curated schedules,
release-development evaluation, and retention evaluation. It fingerprints every
input in an immutable manifest, supports exact-manifest resume, and deliberately
never evaluates the pipeline or final shadow holdouts. Train/evaluation scripts
accept rank and alpha explicitly.

`training/evaluation/semantic-shadow-release-v1.jsonl` is the frozen final
20-case gate: two examples in each of ten command families absent from all
training corpus files. Its grounding audit is closed and strict. Do not run it
for individual rank, ordering, seed, or curriculum-weight variants; consult it
once after development plus retention select a winner.

Long train/evaluation commands show Rich progress on an interactive terminal
and retain sparse plain-text logs when redirected. A real CUDA rank-32 plumbing
smoke completed 20 steps in 36.38 seconds, reduced training loss from 2.3671 to
0.2182 and held-out loss from 2.4600 to 1.5132, and wrote a resumable checkpoint.
This proves GPU/rank/progress plumbing only, not model quality.

Keep broad -> curated order and the seed fixed during the rank comparison. Once
a rank wins, compare that baseline against interleaved/replay training, then
consider curriculum-weight sweeps. A simple curated -> broad reversal is a weak
primary alternative because the broad final pass may erase specialization. If
the apparent improvement is small, repeat rank 16 and the winner with extra
seeds before promotion.

Codex is currently the stronger teacher in a manual distillation loop: it
proposes examples and contrasts, but only reviewed, deterministically checked
records enter the corpus. Conversation text is not ingested automatically. Build
the first automated version as teacher candidate generation plus rejection by
general syntax/semantic/task checks, not as trust in teacher output.

Do not keep tuning schedules against `pipeline-compatibility-v1`; it has already
served as a model-selection holdout. Parameterize and sweep 0.5B LoRA capacity,
then try verifier-guided rejection sampling and only later offline preference
optimization while retaining the 786-row curriculum. The user explicitly wants
to avoid a 1.5B resident laptop model; reserve it as an offline-teacher diagnostic
only after 0.5B approaches saturate, and do not deploy it without a separate
decision. Choose candidates with unchanged development and retention gates,
freeze a fresh shadow release suite before the sweep, and use the pipeline suite
only once for final confirmation. Export or update the served GGUF only after
every gate passes.

Before starting, inspect `git status` and `git diff`; preserve unrelated user
work. The rejected pipeline-verifier implementation should not reappear.
