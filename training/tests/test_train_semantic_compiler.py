import importlib.util
from pathlib import Path

import pytest
import torch

from shelliq_training.compiler_data import CompilerExample, CompilerSpecialTokens
from shelliq_training.data import Corpus, Platform, SFTRecord
from shelliq_training.semantic_compiler import CompilerConfig, SemanticCompiler

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'train_semantic_compiler.py'
SPEC = importlib.util.spec_from_file_location('train_semantic_compiler', SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(record_id: str, command: str) -> SFTRecord:
    return SFTRecord(
        record_id=record_id,
        corpus=Corpus.DISTRIBUTABLE,
        source='fixture',
        license='MIT',
        provenance='fixture',
        command=command,
        platform=Platform.LINUX,
        instruction='Do it.',
        response='{"v":2}',
        context='context',
    )


def test_deterministic_subset_is_stable_and_command_disjoint():
    records = [record('a', 'find'), record('b', 'find'), record('c', 'grep'), record('d', 'tar')]

    first = MODULE.deterministic_subset(records, count=3, seed=7)
    second = MODULE.deterministic_subset(list(reversed(records)), count=3, seed=7)

    assert [item.record_id for item in first] == [item.record_id for item in second]
    assert len({item.command for item in first}) == 3


def test_train_model_rejects_zero_steps():
    config = CompilerConfig(
        vocab_size=16,
        pad_token_id=0,
        decoder_start_token_id=1,
        eos_token_id=2,
        d_model=8,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        feedforward_size=16,
        dropout=0,
        max_source_length=4,
        max_target_length=4,
    )
    example = CompilerExample('a', Corpus.DISTRIBUTABLE, (3, 2), (4, 2))
    with pytest.raises(ValueError, match='positive'):
        MODULE.train_model(
            SemanticCompiler(config),
            [example],
            steps=0,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0,
            max_grad_norm=1,
            special_tokens=CompilerSpecialTokens(0, 2, 1),
            device=torch.device('cpu'),
            log_interval=0,
        )
