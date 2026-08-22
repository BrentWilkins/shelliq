# Preference pilot v3: grounded contrasts with broad replay stream

This matched experiment added 28 manually reviewed, explicit-target contrast
pairs to the 99-pair v2 corpus (127 total). The command-family-disjoint split
contained 96 training and 31 validation pairs. Both conditions started from the
rank-8 full-corpus step-10,000 checkpoint and used rank 8 / alpha 4, three
epochs (288 updates), learning rate `1e-5`, beta `0.1` for DPO, and replay
weight `0.2`.

Unlike v1/v2, replay used `--replay-mode stream`: every preference update used
one fresh deterministic record from the broad `shelliq-curated` training split,
for 288 records rather than cycling a fixed 128-record subset. This is a
preservation intervention, not a behavioral retention proxy.

| checkpoint | held-out reference-relative wins | held-out margin delta | release first command | release flags | release grounded | retention first command | retention flags | retention grounded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rank-8 baseline | — | — | 32/35 | 12/35 | 10/35 | 20/20 | 14/20 | 10/20 |
| chosen SFT | 16/31 | -0.235 | 29/35 | 7/35 | 5/35 | 15/20 | 10/20 | 4/20 |
| DPO | 18/31 | +1.910 | 32/35 | 12/35 | 9/35 | 20/20 | 13/20 | 9/20 |

DPO again improved command-family-held-out preference margins while failing to
improve generated-command behavior. It preserved retention first-command
accuracy but still lost one flag-sequence and one grounded-document match; on
release it added a JSON/envelope regression (32/35). Chosen-only SFT overfit
more severely. Neither checkpoint is promotable and shadow was not evaluated.

The repeated result across v2 and v3 is sufficient to stop preference
hyperparameter, rank, and replay-shape sweeps for this setup. More valid pairs
and broader replay improve the surrogate preference metric, but have not
produced a retention-safe behavioral gain. The next work should be the
clarification/result-state contract and a reconsideration of the training
target or base-model capacity, not another DPO continuation.
