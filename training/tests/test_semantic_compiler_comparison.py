import importlib.util
from pathlib import Path

import pytest

from shelliq_training.compiler_data import CompilerExample
from shelliq_training.data import Corpus

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_semantic_compiler_comparison.py'
SPEC = importlib.util.spec_from_file_location('run_semantic_compiler_comparison', SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def example(index: int) -> CompilerExample:
    return CompilerExample(str(index), Corpus.DISTRIBUTABLE, (index + 2, 1), (index + 3, 1))


def test_ordered_batches_are_seeded_complete_and_nonduplicating():
    examples = [example(index) for index in range(7)]

    first = MODULE.ordered_batches(examples, batch_size=3, seed=19)
    second = MODULE.ordered_batches(examples, batch_size=3, seed=19)

    assert [[item.record_id for item in batch] for batch in first] == [[item.record_id for item in batch] for batch in second]
    assert sorted(item.record_id for batch in first for item in batch) == [str(index) for index in range(7)]
    assert [len(batch) for batch in first] == [3, 3, 1]


def test_custom_configuration_matches_preregistered_parameter_count():
    class Tokenizer:
        pad_token_id = 151643
        eos_token_id = 151645
        bos_token_id = None

        def __len__(self):
            return 151665

    model = MODULE.build_model('custom', Tokenizer(), source_length=256, target_length=256, pretrained=False)
    identity = MODULE.model_identity('custom', model)

    assert identity['parameter_count'] == MODULE.CUSTOM_PARAMETER_TARGET
    assert identity['trainable_parameter_count'] == MODULE.CUSTOM_PARAMETER_TARGET


def test_resolve_device_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA was requested'):
        MODULE.resolve_device('cuda')
