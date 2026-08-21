from scripts import evaluate_corpus_loss
from shelliq_training.data import Corpus, Split


def test_semantic_loader_is_called_with_only_its_path(monkeypatch, tmp_path):
    dataset = tmp_path / 'semantic.jsonl'
    records = [object()]
    monkeypatch.setattr(
        evaluate_corpus_loss,
        'load_semantic_jsonl',
        lambda path: records if path == dataset else None,
    )
    monkeypatch.setattr(
        evaluate_corpus_loss,
        'automatic_preflight',
        lambda loaded: (loaded, {'rejected': 'reason'}),
    )

    def fake_split(loaded, *, corpus, seed):
        assert loaded is records
        assert corpus is Corpus.DISTRIBUTABLE
        assert seed == 2026
        return {Split.TEST: records}

    monkeypatch.setattr(evaluate_corpus_loss, 'split_records', fake_split)

    selected, rejected = evaluate_corpus_loss.load_evaluation_split(
        dataset,
        split=Split.TEST,
        split_seed=2026,
    )

    assert selected is records
    assert rejected == {'rejected': 'reason'}
