# Verified teacher and contrastive data plan

## 2026-08-21 implementation checkpoint

### Expanded reviewed corpus

The second pass used the complete eligible curated inventory under a four-row
per-command cap: 280 rank-8 evaluations yielded 190 structurally valid,
correct-command near misses. Manual review retained 99 preference pairs with a
72/27 command-family-disjoint train/validation split. It excluded 46 task-
equivalent answers, 41 hidden-operand comparisons, 4 ambiguous requirements,
and 2 bad existing targets. Two accepted v1 pairs outside the new pool were
carried forward explicitly, making `corpus/reviewed-preference-v2.jsonl` a
strict superset rather than replacing evidence accidentally.

Qwen prescreening is advisory only. On the already-reviewed calibration set,
its `chosen` classification achieved 82.8% precision and 75% recall, missing 8
useful pairs and falsely selecting 5 ambiguous or hidden-operand comparisons.
This is adequate for review ordering, not corpus admission or exclusion.

The fixed-recipe v2 SFT/DPO comparison completed with all v1 hyperparameters
unchanged, isolating the effect of corpus expansion. Across 72 train and 27
command-family-disjoint validation pairs, DPO reached 71/72 reference-relative
train wins and 15/27 validation wins. Its held-out mean reference-margin delta
was `+1.470`, the first positive held-out preference signal; replay probe loss
was `0.00721`. The matched chosen-only SFT control was nearly flat on validation
(`0.5185`, margin delta `-0.0799`) and regressed release-development behavior.

DPO itself preserved release first-command and flag metrics but lost one
grounded match. It also lost one first-command, flag, and grounded match on
retention, so it fails the gate and is not promotable. This is evidence that
the larger reviewed corpus helps DPO learn distinctions, not that it yet
improves generated commands. Exact behavioral results are in
`experiments/preference-pilot-v2.md`. Shadow remains forbidden.

The first end-to-end pilot is complete. A balanced 128-case rank-8 failure
pool produced 64 structurally valid, correct-command near misses. Manual review
accepted 32 preference pairs and rejected 32: 24 task-equivalent, 6 dependent
on hidden or underspecified operands, and 2 ambiguous. The accepted corpus is
`corpus/reviewed-preference-v1.jsonl`; the complete audit ledger is
`reviews/student-failure-v2.decisions.jsonl`.

The local Qwen teacher cascade was useful diagnostically but is not training
data. Generic corpus prompts often omit concrete sample operands present in the
reviewed target, so asking a teacher to recreate the exact target tests hidden-
operand guessing. Future teachers should either adjudicate supplied
chosen/rejected answers or author the grounded instruction/context/answer trio
together.

A matched 75-step pilot compared chosen-answer SFT plus replay against cached-
reference DPO plus the same replay. DPO preserved existing release and
retention metrics but made no release improvement and added one wrong-command
regression. It fit training preferences without improving the seven held-out
command families. Chosen-only SFT catastrophically overfit. Exact results and
the stop decision are in `experiments/preference-pilot-v1.md`.

Next: expand diverse reviewed rank-8 failures before tuning the optimizer. Do
not sweep beta, learning rate, or epochs against the current 32 pairs. Keep the
same SFT control, cached-reference DPO, curated replay, command-family split,
development release, and retention comparison when the corpus is larger.

### Targeted contrast pass v1

The leak-checked 100-row `targeted-finishing` family was used as a second
candidate source rather than repeated as ordinary SFT. It is command-family
disjoint from every frozen evaluation suite. The rank-8 checkpoint produced 40
valid, correct-command semantic near misses from a 48-row eligible pool; manual
review retained 28 and rejected 12 (8 still hid material operands or endpoints,
4 were task-equivalent). `corpus/reviewed-preference-v3.jsonl` therefore carries
forward all 99 v2 pairs and adds only those 28 reviewed, explicit-target pairs.

The pilot runner now supports `--replay-mode stream`: each preference update
uses a fresh deterministic `shelliq-curated` training record instead of cycling
a fixed 128-example replay subset. This is a preservation experiment, not a
claim that a sampled replay loss is a retention gate. Run the matched SFT/DPO
comparison on v3 with this mode before any beta, learning-rate, rank, or
base-model sweep; release and retention remain the only behavioral gates.

### V3 result and stop condition

The v3 stream-replay matched run reached the strongest held-out preference
margin so far for DPO: 18/31 reference-relative wins and `+1.910`. It still
failed behavior gates: release was unchanged on first-command and flag matches
(32/35 and 12/35), lost one grounded match and three JSON/envelope matches;
retention preserved first-command accuracy but fell to 13/20 flags and 9/20
grounded. The same chosen-answer SFT control regressed much more severely.

This repeats the v2 pattern after both additional explicit-target data and
fresh broad-corpus replay. Stop preference beta/LR/rank/replay sweeps and do
not treat a rising preference margin as a promotion signal. Exact results are
in `experiments/preference-pilot-v3-stream.md`; shadow remains forbidden.

## Why this is the next phase

The full-corpus rank-8 checkpoint is the best supervised starting point, but it
is not promotable. It passes the retention gate and improves grounded release
behavior, yet the 35-row release-development suite still contains 16
missing-flag labels, 13 extra-flag labels, 3 wrong-command labels, and 2 invalid
JSON/envelope labels. Rank 16 has slightly lower corpus test loss but worse
grounding and retention.

More ordinary epochs would continue optimizing token loss without supplying the
missing distinctions. The next phase should add information that the model does
not currently have:

- varied correct examples of multi-constraint flag selection;
- actual high-probability rank-8 mistakes paired with verified corrections;
- controlled contrasts that isolate one error, such as a missing flag, extra
  flag, wrong option binding, or changed operand;
- a small amount of response-envelope reinforcement, without filling the
  preference set with trivial malformed-JSON negatives.

Do not copy release, retention, pipeline, or shadow prompts into training. Those
suites identify failure families only. New prompts must be authored for training
and remain command-family-disjoint from validation.

## Teacher source and cost-controlled scaling path

Use a stronger model as a proposal generator, never as the authority that
accepts a record.

1. **Run a small Codex-assisted pilot under human direction.** Author and review
   the first batch in the repository. Conversation text is not training data;
   only explicit, auditable artifacts that pass the checks below can enter the
   corpus. This pilot calibrates the schema, verifier, failure taxonomy, and
   acceptance-rate target.
2. **Use Luna as the default bulk draft generator.** A Luna agent can cheaply
   draft new training prompts, canonical targets, evidence references, and a few
   controlled contrasts. Run it through the same provider-neutral artifact
   boundary and record its exact model/version and prompt-template hash. Luna is
   a proposer, not a reviewer of its own records.
3. **Optionally use a local model for additional drafts.** Offline inference does
   not increase the deployed model's steady-state RAM requirement. A local model
   can add prompt diversity or proposed targets, but correlated errors are likely,
   especially if it is another Qwen Coder checkpoint. Treat all such output as
   untrusted and measure its verified acceptance rate before scaling it.
4. **Escalate selectively through Fireworks.** The available Fireworks account
   makes a larger hosted reasoning/coding model practical without local VRAM.
   Send it only records that fail to reach deterministic agreement, have
   ambiguous command semantics, cover a new risky command family, or belong to a
   fixed random audit sample. Benchmark suitable hosted models on the pilot and
   select on verified acceptance rate and error coverage rather than reputation.
   Fireworks supports JSON Schema structured outputs and asynchronous JSONL batch
   inference; verify that the selected model reports batch support before a job.
5. **Sample the rank-8 student for most rejected candidates.** Its real errors
   are more useful than a teacher guessing what the student might do wrong. Let
   the teacher construct controlled negatives only where student samples do not
   cover an important failure mode.
6. **Keep generation and approval independent.** Neither Luna nor a local model
   may approve its own proposal. Deterministic checks run first; a human approves
   accepted records. A stronger hosted reviewer can advise on ambiguous records,
   but it still does not replace verifier evidence or human corpus acceptance.

Structured output constrains the teacher's proposal record, not the student's
raw response. Student sampling must remain unconstrained enough to expose the
invalid-envelope behavior that evaluation needs to measure.

Provider references:

- [Fireworks structured outputs](https://docs.fireworks.ai/structured-responses/structured-response-formatting)
- [Fireworks Batch API](https://docs.fireworks.ai/guides/batch-inference)
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [OpenAI Batch API](https://developers.openai.com/api/reference/resources/batches)
- [OpenAI DPO data format](https://developers.openai.com/api/docs/guides/direct-preference-optimization)

Keep generation provider-neutral: the checked-in request manifest, raw response,
and normalized candidate records must not depend on a provider SDK. Do not add
an API dependency until the exporter is implemented; if one is needed, add it
with `uv add` rather than editing `pyproject.toml` by hand. Never check an API key
into the repository or put it in a command-execution container.

The initial teacher tournament uses the frozen
`evaluation/teacher-selection-v1.jsonl` set. It excludes TLDR-derived questions:
their public availability makes pretraining contamination likely and therefore
makes them poor evidence for teacher quality. This set may choose a teacher but may
never become teacher-authored training data or a student release benchmark.

Calibration on 2026-08-21 found that v1 is suitable for exercising the harness
but not yet for ranking teachers. Its inherited generic instructions omit some
literal operands present in the frozen expected documents (for example,
`access.log`, `ubuntu.iso`, and the rsync endpoints), and a few references demand
options not supported by their supplied context. Quarantine exact-match scores
from this set. Before a full tournament, author v2 requests whose instruction and
context jointly determine every expected word, then have a human review each
request/reference pair. Do not repair v1 in place after recording results.

`evaluation/teacher-selection-v2.jsonl` is the resulting 20-case calibration
set. Its builder requires concrete expected words to be visible in the prompt,
allows only quoted shell-language expressions to be derived, and applies four
reviewed instruction corrections for implications the lexical audit cannot see.
It contains 8 precise, 8 multi-constraint, 2 pipeline, and 2 compositional cases,
including 4 Darwin cases. The smaller size is intentional: the existing corpus
cannot supply 48 trustworthy complex references without weakening the audit.

Teacher candidates receive a versioned, provider-neutral description of the
project-private `SemanticDocumentV2` wire format plus neutral formatting
examples. Merely naming the schema is insufficient: an external teacher cannot
reasonably infer abbreviated keys such as `s[].c[].n.s`. The prompt is hashed
with every result so a schema-prompt change creates a distinct tournament run.

### Tournament runbook

Run from `training/`. Start each local candidate with eight prompts and a ten-minute
ceiling, then run all 20 v2 cases only if the smoke result is healthy:

The local runner defaults to `reasoning_effort=none`. This task needs precise
schema translation, and unbounded hidden reasoning can consume the entire output
budget before emitting JSON. Override it only as a separately labeled experiment.
Local requests also use the server's JSON-object constraint. This prevents an
early stop with unbalanced braces while preserving the server's exact output;
the pipeline never repairs malformed candidate text after generation.

The Qwen 3.8 v2 sweep supports a verifier-directed cascade, not global reasoning:
run `none` first, retry only rejected outputs with `medium`, and escalate remaining
failures. Medium rescued hard `brew` and `jq` cases but regressed a simple required
flag; high added no strict accuracy, cost almost four times as much as none, and
produced one malformed response.

```sh
uv run python scripts/run_teacher_tournament.py run-local \
  --challenges evaluation/teacher-selection-v2.jsonl \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --model local-candidate-exact-name \
  --output artifacts/teacher-local-smoke.jsonl \
  --limit 8 \
  --max-total-seconds 600

uv run python scripts/run_teacher_tournament.py run-local \
  --challenges evaluation/teacher-selection-v2.jsonl \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --model local-candidate-exact-name \
  --output artifacts/teacher-local-full.jsonl \
  --max-total-seconds 3600

uv run python scripts/run_teacher_tournament.py score \
  --challenges evaluation/teacher-selection-v2.jsonl \
  --results artifacts/teacher-local-full.jsonl \
  --output artifacts/teacher-local-full.score.json
```

The local runner accepts only an uncredentialed loopback HTTP endpoint. It shows
Rich progress, records per-row latency and token usage, preserves row-level errors,
and aborts when projected total runtime exceeds the stated ceiling. Errors remain
in the metric denominator.

For hosted candidates, `export-batch` requires explicit current input/output prices
and rejects the request artifact when its worst-case estimate exceeds the default
$5 cap. It writes a sidecar manifest containing exact model, provider, prompt hashes,
provider-safe request IDs, original challenge IDs, and the estimate. Anthropic output
is a submit-ready Message Batches JSON body; OpenAI and Fireworks output their
documented JSONL dataset formats. `import-batch` converts downloaded JSONL results
back to the same normalized candidate schema used for local models. The script does
not submit jobs or read API keys, so billable action remains explicit in the provider
CLI or dashboard.

Always re-check the provider model identifier, batch support, and prices immediately
before export. Do not copy example prices from a prior report into a new run.

Track cost and quality per generator: requested rows, valid envelopes, static
verifier pass rate, human acceptance rate, failure-mode coverage, input/output
tokens, hosted cost, and reviewer time. Start with a small stratified batch from
each proposed generator. Scale the cheapest source only if its *verified accepted
records per dollar and reviewer-hour* are competitive. Re-audit a fixed random
sample of accepted records with the strongest reviewer to estimate silent-error
rate; do not review only failures.

## Candidate construction

For each new prompt:

1. Freeze the instruction, authoritative public context, platform, command
   family, and intended postcondition before viewing generated answers.
2. Ask the teacher for one canonical target with evidence references and, at
   most, a few single-fault contrasts. A rationale helps review but is not
   verifier evidence.
3. Sample multiple rank-8 candidates using recorded seeds and decoding settings.
   Preserve raw text, including invalid JSON.
4. Run deterministic checks on every candidate and attach closed failure labels.
5. Have a human approve the intended task, canonical target, evidence, and any
   chosen/rejected pair.
6. Put every approved canonical answer in the rejection-sampling SFT candidate
   set. Add a preference pair only when the rejected answer is plausible and a
   verifier or reviewer can state exactly why it loses.

Prefer a rejected answer that the student assigns meaningful probability and
that differs from the chosen answer along one or two labeled dimensions. Do not
prefer one valid equivalent command merely because it differs from the reference.
Do not use malformed JSON for the bulk of DPO pairs; teach the response envelope
primarily through correct SFT examples and report format negatives separately.

## Verification ladder

Every candidate receives a machine-readable disposition. “The teacher said so”
and “the process exited zero” are never sufficient dispositions.

### 1. Provenance and partition checks

- Record the distributable source and license.
- Reject prompts, targets, and near-duplicates from frozen evaluation suites.
- Reject prompt variables containing secrets or private host data.
- Assign command-family grouping before train/validation splitting.
- Retain immutable hashes of raw generations and accepted records.

### 2. Static checks for every candidate

- strict JSON and exact SemanticDocumentV2 envelope;
- complete AST lowering, rendering, and parse/round-trip stability;
- intended platform, command, and subcommand;
- command-local option existence, case, scope, arity, and value binding;
- ordered flags, operands, separators, redirects, and pipeline framing;
- literal grounding, rejecting invented paths, hosts, values, and credentials;
- risk classification before any possible execution.

Static checks can reject a candidate but cannot always prove that it accomplishes
the task. Authoritative command context and human review remain required when no
functional oracle exists.

### 3. Functional checks for an allowlisted safe subset

Define a synthetic fixture and its expected observable result before execution.
Check the exit status, bounded stdout/stderr, post-execution filesystem tree, and
the task-specific postcondition. Exit status zero alone is not success.

Reasonable initial fixtures include read-only text processing, archive
inspection, and file transformations confined to synthetic `/work`. Commands
requiring credentials, host services, package management, privilege, devices,
process control, or an external network remain static-only.

### 4. Human acceptance

A reviewer confirms that the chosen answer satisfies the instruction and that
the rejected answer fails for the recorded reason. Review explicitly allows
valid alternative commands. Only records with a named reviewer, review date, and
non-empty verifier evidence may enter the checked corpus.

## Command-execution safety boundary

No generated command is currently authorized for functional execution. The
repository has a strict preference-record loader, but the execution harness
described here has not been built.

Docker is available in the user's normal host environment and is the intended
backend. The restricted Codex process cannot access the Docker daemon; that is a
tooling restriction, not evidence that Docker is unavailable. No test image has
yet been pinned and no Docker-backed candidate harness has yet been accepted.

Before enabling functional checks, build and adversarially test a harness with
these properties:

- one fresh disposable Docker container **per candidate**;
- a fresh non-interactive `zsh -f` with no startup files when shell semantics are
  required, and direct `argv` execution for a simple command;
- a pinned image digest and pinned utility behavior;
- rootless Docker or user-namespace isolation where practical, plus a fixed
  unprivileged container UID, all capabilities dropped, `no-new-privileges`, and
  a restrictive seccomp policy;
- isolated PID, IPC, UTS, mount, and network namespaces, with networking disabled;
- a read-only root, fresh tmpfs `/tmp`, and exactly one writable synthetic
  `/work` directory;
- no host home, repository, Docker socket, SSH agent, cloud configuration,
  credentials, or API keys mounted into the container;
- a minimal environment, closed stdin, and bounded stdout/stderr;
- CPU, memory, PID, file-size, and wall-clock limits, with the full process tree
  killed on completion or timeout;
- evidence containing the harness version, image digest, fixture hash, command
  hash, exit status, output hashes, postcondition result, and rejection reason.

A fresh shell without a fresh containment boundary is insufficient. A generic
container with host mounts, Docker socket access, or networking is also
insufficient. The host-side runner must pass data as arguments or structured
input; it must never interpolate generated text into an outer host shell command.

### Never-execute default classes

These remain static-plus-human-review unless a separately designed mock harness
is later approved:

- `sudo`, `doas`, `su`, privilege changes, and user/group administration;
- block devices, mounts, partitioning, kernel controls, reboot, and shutdown;
- package managers, service managers, process signalling, and schedulers;
- external network clients, SSH, cloud CLIs, cluster control, and credential tools;
- download-to-shell, interpreters/eval, arbitrary `sh -c`, and dynamic code;
- absolute host paths, parent traversal, host configuration, and secrets;
- unbounded recursion, fork/background behavior, and resource-stress commands.

Pipelines and redirects may execute only after full AST risk classification and
only inside the same candidate container.

### Harness acceptance tests

Keep functional execution disabled until hostile fixtures prove that a candidate
cannot:

- read a planted host/home secret;
- write outside `/work`;
- reach the internet or host network;
- retain a background descendant after timeout;
- exceed PID, memory, output, CPU, or file-size limits;
- corrupt logs with terminal controls or output flooding;
- access the repository or container-engine socket.

These are containment tests, not proof that an arbitrary command is safe.

## Training and evaluation sequence

1. Build a reviewed pilot and report acceptance and rejection counts by failure
   family. Do not train if any accepted row lacks evidence.
2. Train an **SFT-only control** on approved chosen answers, the existing broad
   corpus, and curated retention replay. Compare it with the unchanged rank-8
   checkpoint. This isolates the value of additional correct data.
3. Once enough genuine high-quality near misses exist, train a DPO-style run on
   the same chosen answers and verified rejected candidates. Retain supervised
   replay and a frozen reference/KL constraint. Compare directly with the
   SFT-only control so preference optimization cannot take credit for data
   expansion.
4. Select checkpoints on command-family-disjoint teacher-data validation and the
   supervised corpus validation set. Then apply the release-development and
   retention gates. Keep the pipeline shadow suite as the single-use final gate.
5. Promote only when schema/envelope, command/flag, grounding, and retention gates
   pass. Preserve all failures and reviewer decisions in the experiment report.

The first implementation milestone is a provider-neutral proposal/candidate
artifact and static-verifier report. The Docker harness is a separate milestone
that must land before any record claims functional-execution evidence.
