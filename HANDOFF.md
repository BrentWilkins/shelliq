# Current development handoff

Updated 2026-08-20.

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
