# Preference pilot v2

This pilot repeated the v1 matched-control recipe after expanding the manually
reviewed preference corpus from 32 to 99 pairs. The command-family-disjoint
split contained 72 training and 27 validation pairs. Both conditions started
from the rank-8 full-corpus step-10,000 checkpoint and used rank 8 / alpha 4,
three epochs (216 updates), learning rate `1e-5`, 128 curated supervised replay
examples, and replay weight `0.2`. DPO used beta `0.1` and cached frozen-reference
log probabilities; the control used chosen-answer SFT.

| checkpoint      | held-out reference-relative wins | held-out margin delta | release first command | release flags | release grounded | retention first command | retention flags | retention grounded |
| --------------- | -------------------------------: | --------------------: | --------------------: | ------------: | ---------------: | ----------------------: | --------------: | -----------------: |
| rank-8 baseline |                                — |                     — |                 32/35 |         12/35 |            10/35 |                   20/20 |           14/20 |              10/20 |
| chosen SFT      |                            14/27 |                -0.080 |                 28/35 |          9/35 |             6/35 |                   14/20 |           10/20 |               5/20 |
| DPO             |                            15/27 |                +1.470 |                 32/35 |         12/35 |             9/35 |                   19/20 |           13/20 |               9/20 |

DPO fit 71/72 training preferences reference-relative and, unlike v1, moved
held-out command-family margins in the preferred direction. The effect was not
large or reliable enough to improve generated-command behavior. On release
development it gained `pidstat-process-cpu-memory` but regressed
`gzip-keep-original` and `readlink-canonical`. On retention it gained
`xargs-null-basenames` but regressed `lsof-listener-8080` and
`docker-compose-log-window`. It therefore fails retention and is not
promotable. Chosen-only SFT again overfit and is rejected.

This is a useful data-quality result, not a checkpoint win. Expanding reviewed
near misses changed held-out preference learning from negative to slightly
positive, supporting continued use of the 99-pair corpus. The next experiment
should add more controlled, fully grounded contrasts and more command-family
coverage before tuning DPO hyperparameters. Do not evaluate the shadow suite
and do not promote either v2 checkpoint.
