#!/usr/bin/env python3
"""Run the preregistered typed candidate-ranking v2 experiment."""

from __future__ import annotations

import run_semantic_action_typed as typed
from transformers import AutoModelForSeq2SeqLM

from shelliq_training.semantic_action_typed_ranking_model import SemanticActionTypedRankingModel

EXPERIMENT = 'semantic-action-typed-ranking-v2'
ROLE_COUNT = 8


def build_model(tokenizer, *, dropout: float = 0.1) -> SemanticActionTypedRankingModel:
    pretrained = AutoModelForSeq2SeqLM.from_pretrained(
        typed.candidate.MODEL_ID,
        revision=typed.candidate.REVISION,
        local_files_only=True,
    )
    return SemanticActionTypedRankingModel.from_codet5(
        pretrained,
        tokenizer,
        maximum_source_bytes=typed.candidate.SOURCE_BYTE_LENGTH,
        role_count=ROLE_COUNT,
        dropout=dropout,
    )


def main() -> None:
    typed.EXPERIMENT = EXPERIMENT
    typed.build_model = build_model
    typed.main()


if __name__ == '__main__':
    main()
