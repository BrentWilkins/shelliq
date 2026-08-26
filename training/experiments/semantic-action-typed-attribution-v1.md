# Semantic-action typed attribution v1

Status: complete; candidate ranking is the dominant bottleneck.

This read-only study uses the selected epoch-7 checkpoint from
`semantic-action-typed-slots-v1` and the already-open 92-record inner split. It
does not retrain, tune, or access outer validation or test.

## Frozen measurements

- Teacher-forced candidate top-1, top-4, top-8, and mean reciprocal rank by
  grammar word role and candidate provenance.
- Teacher-forced argument-count top-1, top-4, top-8, mean reciprocal rank, and
  top-1 mean absolute error.
- Decode all 92 records with reference argument counts but learned candidates.
- On the oracle-fully-covered records only, decode with reference words but
  learned counts, then with both reference words and reference counts.

Reference decisions remove their learned score contribution but leave all other
beam scores and structural choices unchanged. The width-8 beam, role masks,
192-action bound, checkpoint, and candidate inventory are frozen.

## Decision rule

- If gold counts materially improve complete acceptance or count top-4 recall is
  weak, prioritize the count head.
- If gold words materially improve acceptance while gold counts do not,
  prioritize role-specific candidate ranking.
- If both gold words and counts still fail on fully covered records, prioritize
  structural/beam scoring.
- If reference candidates are usually outside top eight despite adequate oracle
  coverage, use role- and position-specific selectors with explicit hard
  negatives rather than expanding the inventory.

Any follow-up architecture and gate will be preregistered after these results.

## Outcome

The 23 oracle-fully-covered references all decode exactly when both reference
words and reference argument counts are supplied. With reference words but
learned counts, 13/23 remain exact and accepted. With reference counts but
learned candidates, the full 92-record score improves only from one to two exact
and accepted references. Structure and beam composition are therefore adequate;
word selection dominates count selection.

Teacher-forced ranks reinforce that conclusion:

- Command candidates: 83/91 top-1 (91.21%), 89/91 top-4.
- Argument candidates: 84/229 top-1 (36.68%), 134/229 top-4,
  149/229 top-8.
- Local candidates: 108/158 top-1 (68.35%).
- Local-plus-global candidates: 56/82 top-1 (68.29%).
- Global-only candidates: 3/66 top-1 (4.55%).
- Derived-only candidates: 0/13 top-1 and 3/13 top-8.
- Argument counts: 16/93 top-1 (17.20%), 67/93 top-4, 89/93 top-8.

Although count top-1 accuracy is low, width-8 count expansion is sufficient for
13 exact programs once words are correct. The next intervention must improve
global/derived argument ranking. The existing candidate content representation
is a UTF-8 byte histogram, which discards character order; additionally, normal
training lets prompt context dominate candidates that are both local and global.
An evidence-supported follow-up should add order-sensitive text features,
role-specific scoring, and an auxiliary content-only loss for globally observed
targets.

The raw diagnostic report SHA-256 is
`1ef80b95bb407f2337c95e930ac3d5ea6e16bd9f7272e8b86ff8d5652b006e86`.
