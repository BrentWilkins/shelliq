# Full-corpus LoRA finalist evaluation v1

The corpus test split, release-development suite, and retention suite were
evaluated for the two step-10,000 checkpoints. Pipeline and shadow suites were
not consulted.

| rank | test loss | release flags | release grounded | release gate | retention flags | retention grounded | retention gate |
| ---: | ---: | ---: | ---: | :---: | ---: | ---: | :---: |
| 8 | 0.2591 | 12/35 | 10/35 | fail | 14/20 | 10/20 | pass |
| 16 | 0.2564 | 13/35 | 8/35 | fail | 11/20 | 8/20 | fail |

Rank 16's 0.00266 lower test loss does not predict better structured behavior.
It loses two grounded release cases and three retention flag cases relative to
rank 8. Rank 8 is the selected SFT starting point, but it is not promotable.

The real release gate rejects both checkpoints. Rank 8 emits invalid JSON for
2/35 cases and reaches only 12/35 exact command-plus-flag sequences. Rank 16
emits invalid JSON for 1/35 and reaches 13/35 flag sequences; the fixed gate
requires strictly exceeding 13/35. Rank 16 also regresses retention first-command
accuracy from 20/20 to 19/20 and flag accuracy from 14/20 to 11/20.

Rank 8's main release failure signals are missing flags (16 cases), extra flags
(13), wrong commands (3), and invalid JSON/envelopes (2). These overlapping
categories point toward verified contrastive teacher data and format-negative
examples rather than another broad rank/LR sweep. Preserve rank 8's retention
behavior through replay and the existing no-regression gate.

The initial generated release-analysis artifacts incorrectly showed a passing
gate because the runner invoked analysis without CLI exit-on-failure mode and
the analyzer treated that as “do not calculate the gate.” The metrics were
correct. Commit `d614922` introduced the runner; the subsequent analyzer fix
makes gate calculation unconditional while keeping `--gate` as exit behavior
only. The corrected gate results above were recomputed from the saved evaluation
reports.
