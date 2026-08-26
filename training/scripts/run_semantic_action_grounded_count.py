#!/usr/bin/env python3
"""Run the preregistered grounded retrieval/count hybrid."""

from __future__ import annotations

import run_semantic_action_typed as typed
from transformers import AutoModelForSeq2SeqLM

from shelliq_training.semantic_action_grounded_count_model import (
    SemanticActionGroundedCountModel,
)

EXPERIMENT = 'semantic-action-grounded-count-v1'
ROLE_COUNT = 8


def build_model(tokenizer, *, dropout: float = 0.1) -> SemanticActionGroundedCountModel:
    pretrained = AutoModelForSeq2SeqLM.from_pretrained(
        typed.candidate.MODEL_ID,
        revision=typed.candidate.REVISION,
        local_files_only=True,
    )
    return SemanticActionGroundedCountModel.from_codet5(
        pretrained,
        tokenizer,
        maximum_source_bytes=typed.candidate.SOURCE_BYTE_LENGTH,
        role_count=ROLE_COUNT,
        dropout=dropout,
    )


def main() -> None:
    typed.EXPERIMENT = EXPERIMENT
    typed.BEAM_WIDTH = 32
    typed.ARGUMENT_COUNT_TOP_K = 8
    typed.GROUND_ARGUMENTS = True
    typed.NORMALIZE_NUMBER_WORDS = True
    typed.POST_COMMAND_COUNTS = True
    typed.build_model = build_model
    typed.main()


if __name__ == '__main__':
    main()
