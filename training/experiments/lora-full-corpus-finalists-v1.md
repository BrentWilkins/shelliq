# Full-corpus LoRA finalists v1

Both finalists were offered all 23,646 usable training rows with batch size 2,
500 warmup steps, cosine decay, deterministic reshuffling, and validation every
1,000 steps. One epoch is 11,823 optimizer steps. The fixed 256-row validation
sample selected checkpoints; corpus test and behavioral suites were not used.

| rank | alpha/rank | peak LR | best step | best validation | final validation | final train | clipped |
| ---: | ---------: | ------: | --------: | --------------: | ---------------: | ----------: | ------: |
|   16 |          1 |    5e-5 |    10,000 |          0.2664 |           0.2841 |      0.1392 |   89.9% |
|    8 |        0.5 |    2e-4 |    10,000 |          0.2698 |           0.2832 |      0.1341 |   28.0% |

Validation curve near convergence (lower is better):

|                    step |         r8 |        r16 |
| ----------------------: | ---------: | ---------: |
|                   5,000 |     0.2765 |     0.2789 |
|                   6,000 |     0.2756 |     0.2717 |
|                   7,000 |     0.2738 |     0.2723 |
|                   8,000 |     0.2711 |     0.2722 |
|                   9,000 |     0.2732 |     0.2708 |
|              **10,000** | **0.2698** | **0.2664** |
|                  11,000 |     0.2699 |     0.2682 |
| 12,000 (epoch boundary) |     0.2787 |     0.2791 |
|                  13,000 |     0.2809 |     0.2810 |
|                  14,000 |     0.2823 |     0.2821 |
|        15,000 (stopped) |     0.2832 |     0.2841 |

The selected step had consumed 20,000 rows, or 84.6% of the first deterministic
random permutation. The early-stopping floor prevented termination before one
epoch but did not require the retained best checkpoint to follow one epoch.
The matching best step and degradation at the 11,823-step epoch boundary are
not evidence that training was incomplete. Training loss continued from roughly
0.21 to 0.14 while validation worsened, which is direct overfitting evidence.
Rank 16's validation advantage is 0.00336 (about 1.25% relative), too small to
promote without corpus-test and behavioral gates. A validation loss of 0.266 is
token cross-entropy, not a 26.6% error rate; it implies geometric-mean target
token probability around 76.6% and perplexity around 1.305.

Exact observations are in `lora-full-corpus-finalists-v1.json`.
