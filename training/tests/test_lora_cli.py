import sys

from scripts import evaluate_checkpoint, evaluate_semantic_checkpoint, smoke_finetune


def test_finetune_accepts_rank_and_alpha(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'smoke_finetune.py',
            '--semantic-dataset',
            'dataset.jsonl',
            '--rank',
            '64',
            '--alpha',
            '128',
        ],
    )
    args = smoke_finetune.parse_args()
    assert (args.rank, args.alpha) == (64, 128.0)


def test_checkpoint_evaluators_accept_rank_and_alpha(monkeypatch):
    common = ['--checkpoint', 'checkpoint', '--output', 'report.json', '--rank', '32', '--alpha', '64']

    monkeypatch.setattr(sys, 'argv', ['evaluate_checkpoint.py', '--dataset', 'dataset.jsonl', *common])
    raw_args = evaluate_checkpoint.parse_args()

    monkeypatch.setattr(
        sys,
        'argv',
        ['evaluate_semantic_checkpoint.py', '--semantic-dataset', 'dataset.jsonl', *common],
    )
    semantic_args = evaluate_semantic_checkpoint.parse_args()

    assert (raw_args.rank, raw_args.alpha) == (32, 64.0)
    assert (semantic_args.rank, semantic_args.alpha) == (32, 64.0)
