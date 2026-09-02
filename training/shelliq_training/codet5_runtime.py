# ruff: noqa: E402, I001
"""Local inference runtime for the Salesforce CodeT5 semantic-action model."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch
from transformers import RobertaTokenizer

_SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_semantic_action_candidate as candidate
import run_semantic_action_grounded_count_v2 as grounded_count
import run_semantic_action_typed as typed
from shelliq_training.data import Corpus, SequenceTooLongError
from shelliq_training.semantic_action_copy import AlignedCodeT5Tokenizer
from shelliq_training.semantic_action_typed import (
    TypedActionExample,
    byte_roles,
    typed_candidates,
)
from shelliq_training.semantic_actions import ActionGrammar, SemanticActionClient


class CodeT5SemanticRuntime:
    """Load one frozen checkpoint and decode requests through the Rust grammar."""

    def __init__(self, checkpoint_path: Path, actions_path: Path, device: str = 'auto') -> None:
        self.device = _device(device)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        _validate_checkpoint(checkpoint)

        vocab, merges = candidate.tokenizer_files()
        self.tokenizer = RobertaTokenizer(vocab=vocab, merges=merges)
        self.aligned = AlignedCodeT5Tokenizer(vocab, merges)
        self.client = SemanticActionClient(actions_path)
        self.grammar = ActionGrammar(self.client.manifest())
        self.roles = byte_roles(self.grammar)
        self.global_roles = {word.encode('utf-8'): tuple(roles) for word, roles in checkpoint['global_roles'].items()}

        self.model = grounded_count.build_model(self.tokenizer, dropout=0).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self._lock = threading.Lock()

    def generate(self, source: str) -> dict[str, object]:
        if not source.strip():
            raise ValueError('user message must not be empty')
        source_ids, source_bytes, byte_token_indices = self.aligned.encode_with_byte_alignment(source)
        if len(source_ids) > candidate.SOURCE_LENGTH:
            raise SequenceTooLongError(f'request has {len(source_ids)} source tokens; maximum is {candidate.SOURCE_LENGTH}')
        if len(source_bytes) > candidate.SOURCE_BYTE_LENGTH:
            raise SequenceTooLongError(f'request has {len(source_bytes)} source bytes; maximum is {candidate.SOURCE_BYTE_LENGTH}')

        candidates = typed_candidates(
            source,
            self.grammar,
            self.global_roles,
            maximum_bytes=candidate.SOURCE_BYTE_LENGTH,
            normalize_number_words=True,
        )
        example = TypedActionExample(
            record_id='runtime',
            corpus=Corpus.DISTRIBUTABLE,
            source_ids=source_ids,
            source_bytes=source_bytes,
            byte_token_indices=byte_token_indices,
            candidates=candidates,
            action_ids=(),
            candidate_labels=(),
            argument_count_labels=(),
            word_role_labels=(),
        )

        with self._lock:
            sequences = typed.generate(
                self.model,
                [example],
                self.tokenizer,
                self.grammar,
                self.device,
            )
            decoded = self.client.decode(sequences)[0]
        if not decoded.valid or decoded.document is None:
            raise ValueError(f'semantic-action decoder rejected generation: {decoded.error}')
        return decoded.document

    def chat_completion(self, request: object) -> dict[str, object]:
        content = validate_chat_request(request)
        document = self.generate(content)
        return {
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': json.dumps(document, separators=(',', ':')),
                    },
                    'finish_reason': 'stop',
                }
            ]
        }


def validate_chat_request(request: object) -> str:
    if not isinstance(request, dict):
        raise ValueError('request must be a JSON object')
    if request.get('temperature') != 0:
        raise ValueError('temperature must be zero')
    if request.get('stream') is not False:
        raise ValueError('streaming is not supported')
    max_tokens = request.get('max_tokens')
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 192:
        raise ValueError('max_tokens must be an integer from 1 through 192')
    messages = request.get('messages')
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError('exactly one message is required')
    message = messages[0]
    if not isinstance(message, dict) or message.get('role') != 'user':
        raise ValueError('the message role must be user')
    content = message.get('content')
    if not isinstance(content, str) or not content.strip():
        raise ValueError('message content must be a nonempty string')
    return content


def _device(requested: str) -> torch.device:
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    if requested not in {'cpu', 'cuda'}:
        raise ValueError('device must be auto, cpu, or cuda')
    return torch.device(requested)


def _validate_checkpoint(checkpoint: object) -> None:
    if not isinstance(checkpoint, dict) or checkpoint.get('schema_version') != 1:
        raise ValueError('unsupported checkpoint schema')
    if checkpoint.get('codet5_revision') != candidate.REVISION:
        raise ValueError('checkpoint CodeT5 revision does not match the pinned tokenizer/model revision')
    if not isinstance(checkpoint.get('global_roles'), dict):
        raise ValueError('checkpoint does not contain the typed global-role lexicon')
    if not isinstance(checkpoint.get('model_state_dict'), dict):
        raise ValueError('checkpoint does not contain model weights')
    metadata = checkpoint.get('metadata')
    if not isinstance(metadata, dict) or metadata.get('experiment') != 'semantic-action-compiler-rules-v2':
        raise ValueError('checkpoint is not the compiler-rules-v2 outer artifact')
