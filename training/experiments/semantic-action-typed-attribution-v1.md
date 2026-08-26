# Semantic-action typed attribution v1

Status: diagnostic protocol frozen before evaluation.

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
