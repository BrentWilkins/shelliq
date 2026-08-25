# Semantic compiler held-out comparison v1

Status: complete; neither contender achieved held-out semantic acceptance.

## Question

Given the same command-disjoint supervised data and record presentations, does a
roughly 60M-parameter custom encoder-decoder trained from scratch generalize to
unseen command families, and how does it compare with pretrained
`Salesforce/codet5-small`?

CodeT5 is an encoder-decoder model intended for downstream code-generation
fine-tuning. Its official model card is
<https://huggingface.co/Salesforce/codet5-small>; the original paper is
<https://aclanthology.org/2021.emnlp-main.685/>.

## Frozen data

- Input: `artifacts/curated-semantic-v7.jsonl`, SHA-256
  `d5bb17a6e13f95051a0a97971a07ddfb8a806fdf63dc3d47d160cbf47f62d7ca`.
- Split seed: `20260825`.
- Grouping key: command, using the existing stable SHA-256 split function.
- Fractions: 10% validation, 10% test, 80% train.
- Train: 610 rows, 124 commands.
- Validation: 78 rows, 15 commands.
- Test: 98 rows, 17 commands.
- The checked manifest freezes every record ID and command before training.
- Manifest SHA-256:
  `e6f661623b6be106b9419ecdce13a8f68e3130c81ca8e78accf54ed7aa2b272c`.

No command may occur in more than one split. Test targets must not be decoded or
used for selection until both contenders' validation-selected checkpoints are
locked.

## Contenders

### Custom scratch compiler

- Existing Qwen tokenizer only; no pretrained neural weights.
- Four encoder and four decoder layers.
- `d_model=320`, eight heads, feed-forward size 1280.
- Learned source and target positions, shared token/output embeddings.
- Exact parameter count: 60,207,680 with the frozen tokenizer vocabulary.

### CodeT5-small

- Official `Salesforce/codet5-small` tokenizer and pretrained weights.
- Full-model fine-tuning through the Transformers seq2seq interface.
- Record the resolved revision, configuration, and exact trainable parameter
  count before training.
- The parameter-count ratio must be within 5%; otherwise stop before training
  and resize the custom contender without consulting validation or test data.

Pre-training inspection resolved revision
`b1ee9570c289f21b5922b9c768a1ce12957bf968` at 60,492,288 trainable
parameters, 1.0047 times the custom count. The match gate passes.

This is a whole-system pretrained-versus-scratch comparison. Tokenizer and
pretraining differences are intentional and must be reported; the experiment
does not isolate architecture alone.

## Matched budget and selection

- Prompt contract: `context-authoritative-v1`.
- Source and target limits: 256 tokens, with no implicit truncation.
- Seed: `20260825`.
- Batch size: 8.
- Fixed 50 epochs: 30,500 record presentations per contender.
- AdamW, zero weight decay, gradient norm clipped to 1.0.
- Custom learning rate: `1e-3`; CodeT5 learning rate: `5e-5`.
- BF16 autocast on CUDA; FP32 optimizer state.
- Evaluate validation teacher-forced token loss after every epoch and retain the
  lowest-loss checkpoint. Both contenders still consume all 50 epochs.
- Greedy decoding only; no beam search or validator-guided retries.

Different learning rates are preregistered because scratch training and
pretrained fine-tuning have different stable optimization scales. There is no
learning-rate sweep in this experiment.

Pre-training token inspection also passes. The custom tokenizer's maximum
source/target lengths are 162/151 in train, 154/77 in validation, and 156/81 in
test. CodeT5's are 181/205, 165/95, and 168/112 respectively. No row requires
truncation, and these measurements are not model outputs.

## Metrics and decision

On the sealed test split, report:

1. conservative reference acceptance (primary semantic metric);
2. exact compact-target match;
3. Rust semantic round-trip validity;
4. paired wins, losses, and ties for both acceptance and exact match;
5. validation-selected epoch, validation loss, runtime, parameter count, and
   checkpoint hash.

Passing the custom-model generalization gate requires at least one accepted
held-out record and at least 25% Rust-valid output. A zero-acceptance result means
the overfit proof did not transfer. CodeT5 superiority is descriptive at this
stage; no release or promotion decision follows from this development split.

## Prohibitions

- Do not inspect or train on release, pipeline, shadow, or
  `curated-development-v1` evaluation suites.
- Do not execute generated commands.
- Do not serve, export, or promote either checkpoint.
- Preserve checkpoints and raw reports only as ignored local artifacts.

## Results

Both contenders completed the fixed 50 epochs and exactly 30,500 training-record
presentations. The selected checkpoints were locked before the test split was
decoded.

| Contender | Parameters | Best epoch | Validation loss | Train time |
| --- | ---: | ---: | ---: | ---: |
| Custom scratch | 60,207,680 | 4 | 1.92849 | 152.57 s |
| CodeT5-small | 60,492,288 | 8 | 0.46963 | 147.00 s |

The 98-row sealed command-disjoint test result was:

| Contender | Exact target | Rust-valid | First command | Reference accepted |
| --- | ---: | ---: | ---: | ---: |
| Custom scratch | 0/98 | 0/98 | 0/98 | 0/98 |
| CodeT5-small | 0/98 | 91/98 | 86/98 | 0/98 |

The custom model collapsed to one unterminated `git -n ...` document on every
test row. It therefore failed the preregistered generalization gate. The earlier
4/4 overfit result established local memorization only and did not transfer to
unseen command families.

CodeT5 learned the compact output grammar and usually selected the command named
by the authoritative context. Its 91 paired Rust-valid wins over custom have an
exact two-sided McNemar p-value of `8.08e-28`; its 86 first-command wins have
`p=2.58e-26`. However, every output differed materially from its reviewed
reference, most often through incorrect, duplicated, or misplaced options and
operands. Both contenders scored 0/98 on the primary semantic metric, so there
is no semantic winner.

The checked generation-free result is
`experiments/semantic-compiler-heldout-v1-results.json`. Raw generations,
training curves, and the roughly 415 MB checkpoints remain ignored under
`artifacts/semantic-compiler-heldout-v1/`. Their checkpoint SHA-256 values are:

- custom: `d006192fd8a9ddb5ddc09fa897c7e81aec8b7d0d891312e6a50df07dc61bc13b`;
- CodeT5: `7d21790f21ca6f89de2f9693267772e0025cf08db1c8aab0fe48f1ce2649251c`.

## Decision

Stop the plain scratch compiler recipe. Do not interpret CodeT5's structural
validity as task correctness and do not promote either checkpoint. Grammar
constraining alone is not the next bottleneck because CodeT5 already emits valid
documents 92.9% of the time.

The next defensible architecture experiment is a hybrid: retain a pretrained
natural-language encoder but replace free-form JSON generation with a compact,
project-owned semantic action vocabulary and grammar-masked decoder. That tests
whether reducing the 32K/152K output spaces and enforcing field structure frees
capacity for option/operand semantics. It must reuse this frozen split and
compare against the locked CodeT5 result without reopening the test set for
model selection.
