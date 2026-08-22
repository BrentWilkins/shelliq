# Rank-8 one-epoch SFT coverage v1

This isolated data coverage from preference optimization. Starting from the
base model, rank 8 / alpha 4 trained on all 23,646 frozen training-partition
records exactly once (11,823 batch-2 steps), with a `1e-4` peak LR, 500-step
warmup, and cosine decay to `1e-5`. Validation, release, retention, and shadow
records were excluded from gradients.

The final full-coverage checkpoint was also best on the fixed 256-row corpus
validation sample: `0.26548`, better than the incumbent rank-8 checkpoint's
`0.26980`. It did not improve command behavior:

| checkpoint | release first command | release flags | release grounded | retention first command | retention flags | retention grounded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent rank-8 | 32/35 | 12/35 | 10/35 | 20/20 | 14/20 | 10/20 |
| one-epoch lower-LR rank-8 | 33/35 | 9/35 | 6/35 | 20/20 | 13/20 | 8/20 |

This is not promotable. It shows that the prior 10,000-step selection did not
merely miss useful unique training rows: slower full coverage improves corpus
loss while still moving command flags and grounding in the wrong direction.
Shadow was not evaluated.
