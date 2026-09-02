"""Documentation-conditioned CodeT5 cross-encoder utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, RobertaTokenizer
from transformers.utils import cached_file

from shelliq_training.data import load_semantic_jsonl
from shelliq_training.documentation_templates import (
    DocumentationTemplate,
    RetrievedTemplate,
    bind_template,
    documentation_templates,
    unresolved_placeholders,
)
from shelliq_training.semantic_actions import SemanticActionClient

EXPERIMENT = 'documentation-cross-encoder-v1'
EXPERIMENT_V2 = 'documentation-cross-encoder-v2'
MODEL_ID = 'Salesforce/codet5-small'
REVISION = 'b1ee9570c289f21b5922b9c768a1ce12957bf968'
MAX_LENGTH = 256
SPLIT_SEED = 20260901
SAFE_COMMAND = re.compile(r'^[A-Za-z0-9][A-Za-z0-9+._-]*$')
VERB_MAP = {
    'display': 'show',
    'print': 'show',
    'list': 'show',
    'remove': 'delete',
    'create': 'make',
    'download': 'fetch',
    'upload': 'send',
    'search': 'find',
    'execute': 'run',
    'convert': 'transform',
    'install': 'add',
}


@dataclass(frozen=True, slots=True)
class RankCase:
    command: str
    source_record_id: str
    query: str


@dataclass(frozen=True, slots=True)
class RankedPool:
    source_record_id: str
    record_ids: tuple[str, ...]
    scores: tuple[float, ...]
    top_record_id: str
    top_score: float


def ordered_commands(commands: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            commands,
            key=lambda command: (
                hashlib.sha256(f'{EXPERIMENT}\0{command}'.encode()).hexdigest(),
                command,
            ),
        )
    )


def split_commands(commands: Iterable[str]) -> dict[str, tuple[str, ...]]:
    ordered = ordered_commands(commands)
    if len(ordered) < 704:
        raise ValueError(f'need at least 704 eligible commands, found {len(ordered)}')
    return {
        'test': tuple(ordered[:128]),
        'development': tuple(ordered[128:192]),
        'train': tuple(ordered[192:704]),
    }


def load_templates(
    path: Path,
) -> tuple[dict[str, DocumentationTemplate], dict[str, dict[str, object]], dict[str, tuple[str, ...]]]:
    raw_rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    raw_by_id = {row['record_id']: row for row in raw_rows}
    typed = documentation_templates(load_semantic_jsonl(path))
    typed_by_id = {template.record_id: template for template in typed}
    grouped: dict[str, list[str]] = defaultdict(list)
    for template in typed:
        if template.platform.value == 'linux' and SAFE_COMMAND.fullmatch(template.command):
            grouped[template.command].append(template.record_id)
    eligible = {command: tuple(sorted(record_ids)) for command, record_ids in grouped.items() if len(record_ids) >= 2}
    return typed_by_id, raw_by_id, eligible


def paraphrase(instruction: str) -> str:
    words = instruction.split(maxsplit=1)
    if not words:
        return instruction
    replacement = VERB_MAP.get(words[0].lower())
    if replacement is None:
        return instruction
    if words[0][0].isupper():
        replacement = replacement.capitalize()
    return replacement if len(words) == 1 else f'{replacement} {words[1]}'


def candidate_text(raw: Mapping[str, object]) -> str:
    context = str(raw['context']).removeprefix('Output contract: compact SemanticDocumentV2 JSON only.\n')
    return (
        f'<command>{raw["command"]}</command>\n'
        f'<platform>{raw["platform"]}</platform>\n'
        f'<summary>{context}</summary>\n'
        f'<example>{raw["instruction"]}</example>\n'
        f'<shape>{raw["shell_response"]}</shape>'
    )


def pair_text(query: str, raw: Mapping[str, object]) -> str:
    return f'<query>{query}</query>\n<candidate>\n{candidate_text(raw)}\n</candidate>'


def distractor_commands(commands: Sequence[str], index: int, count: int = 7) -> tuple[str, ...]:
    if len(commands) <= count:
        raise ValueError('not enough commands for distinct distractors')
    return tuple(commands[(index + offset) % len(commands)] for offset in range(1, count + 1))


def candidate_pool(
    case: RankCase,
    split_commands_: Sequence[str],
    grouped: Mapping[str, Sequence[str]],
    *,
    sufficient: bool,
) -> tuple[str, ...]:
    command_index = split_commands_.index(case.command)
    record_ids: list[str] = []
    if sufficient:
        own = list(grouped[case.command][:32])
        if case.source_record_id not in own:
            own[-1] = case.source_record_id
        record_ids.extend(own)
    for command in distractor_commands(split_commands_, command_index):
        record_ids.append(grouped[command][0])
    return tuple(dict.fromkeys(record_ids))


class FrozenCodeT5CrossEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pretrained = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            local_files_only=True,
        )
        self.encoder = pretrained.get_encoder()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.hidden_size = int(pretrained.config.d_model)
        self.head = nn.Linear(self.hidden_size, 1)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def score_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings).squeeze(-1)


def tokenizer() -> RobertaTokenizer:
    vocab = cached_file(MODEL_ID, 'vocab.json', revision=REVISION, local_files_only=True)
    merges = cached_file(MODEL_ID, 'merges.txt', revision=REVISION, local_files_only=True)
    return RobertaTokenizer(vocab=vocab, merges=merges)


@torch.no_grad()
def embed_texts(
    model: FrozenCodeT5CrossEncoder,
    tokenizer_: RobertaTokenizer,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[torch.Tensor, int]:
    model.encoder.eval()
    chunks: list[torch.Tensor] = []
    truncated = 0
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        lengths = tokenizer_(batch_texts, add_special_tokens=True, truncation=False)['input_ids']
        truncated += sum(len(tokens) > MAX_LENGTH for tokens in lengths)
        encoded = tokenizer_(
            batch_texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt',
        )
        chunks.append(
            model.encode(
                encoded['input_ids'].to(device),
                encoded['attention_mask'].to(device),
            ).cpu()
        )
    return torch.cat(chunks) if chunks else torch.empty((0, model.hidden_size)), truncated


def rank_pool(
    model: FrozenCodeT5CrossEncoder,
    embeddings: torch.Tensor,
    record_ids: Sequence[str],
    source_record_id: str,
) -> RankedPool:
    with torch.no_grad():
        scores = model.score_embeddings(embeddings).cpu().tolist()
    top_index = max(range(len(scores)), key=lambda index: (scores[index], record_ids[index]))
    return RankedPool(
        source_record_id,
        tuple(record_ids),
        tuple(float(score) for score in scores),
        record_ids[top_index],
        float(scores[top_index]),
    )


def compile_selected(
    case: RankCase,
    selected_record_id: str,
    typed_by_id: Mapping[str, DocumentationTemplate],
    actions: SemanticActionClient,
) -> bool:
    return bind_selected(case, selected_record_id, typed_by_id, actions) is not None


def bind_selected(
    case: RankCase,
    selected_record_id: str,
    typed_by_id: Mapping[str, DocumentationTemplate],
    actions: SemanticActionClient,
) -> dict[str, object] | None:
    template = typed_by_id[selected_record_id]
    required = set(unresolved_placeholders(template.document))
    bound = bind_template(RetrievedTemplate(template, 1.0), case.query)
    if not required.issubset(dict(bound.bindings)):
        return None
    try:
        encoded = actions.encode([bound.document])
        decoded = actions.decode(encoded)[0]
    except ValueError:
        return None
    return bound.document if decoded.valid and decoded.document == bound.document else None


def threshold_for_complete_abstention(insufficient_scores: Sequence[float]) -> float:
    if not insufficient_scores:
        raise ValueError('cannot calibrate without insufficient scores')
    return math.nextafter(max(insufficient_scores), math.inf)


class DocumentationCrossEncoderRuntime:
    """Rank local documentation recipes and return only validated bound documents."""

    def __init__(
        self,
        documentation_index: Path,
        checkpoint_path: Path,
        actions_path: Path,
        device: str = 'auto',
    ) -> None:
        self.device = runtime_device(device)
        self.typed_by_id, self.raw_by_id, self.grouped = load_templates(documentation_index)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint.get('experiment') != EXPERIMENT_V2 or checkpoint.get('threshold') != 0.0:
            raise ValueError('runtime requires the frozen v2 zero-logit checkpoint')
        self.threshold = float(checkpoint['threshold'])
        self.model = FrozenCodeT5CrossEncoder().to(self.device)
        self.model.head.load_state_dict(checkpoint['head_state_dict'])
        self.model.eval()
        self.tokenizer = tokenizer()
        self.actions = SemanticActionClient(actions_path)
        self._lock = threading.Lock()

    def generate(self, source: str) -> dict[str, object]:
        context, instruction = parse_authoritative_prompt(source)
        commands = context_commands(context)
        record_ids = tuple(record_id for command in commands for record_id in self.grouped.get(command, ())[:32])
        if not record_ids:
            raise ValueError('no local documentation candidates for the bounded command context')
        texts = [pair_text(instruction, self.raw_by_id[record_id]) for record_id in record_ids]
        with self._lock:
            embeddings, _ = embed_texts(
                self.model,
                self.tokenizer,
                texts,
                device=self.device,
                batch_size=16,
            )
            ranked = rank_pool(self.model, embeddings, record_ids, '')
        if ranked.top_score < self.threshold:
            raise ValueError('local documentation is insufficient at the frozen abstention threshold')
        selected = self.typed_by_id[ranked.top_record_id]
        document = bind_selected(
            RankCase(selected.command, selected.record_id, instruction),
            selected.record_id,
            self.typed_by_id,
            self.actions,
        )
        if document is None:
            raise ValueError('selected documentation recipe has unresolved or invalid bound slots')
        return document

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


def parse_authoritative_prompt(source: str) -> tuple[str, str]:
    match = re.search(
        r'<context>\n(?P<context>.*?)\n</context>\n<instruction>\n(?P<instruction>.*?)\n</instruction>\Z',
        source,
        re.DOTALL,
    )
    if match is None:
        raise ValueError('adapter requires context-authoritative-v1 prompt regions')
    context = match.group('context').strip()
    instruction = match.group('instruction').strip()
    if not context or not instruction:
        raise ValueError('context and instruction must be nonempty')
    return context, instruction


def context_commands(context: str) -> tuple[str, ...]:
    final = re.search(r'Final pass: use exactly command `([^`]+)`\.', context)
    if final is not None:
        return (final.group(1),)
    marker = 'Commands:\n'
    if marker not in context:
        raise ValueError('adapter context does not contain a bounded command set')
    commands = tuple(
        line.removeprefix('- ').strip()
        for line in context.split(marker, maxsplit=1)[1].splitlines()
        if line.startswith('- ') and line.removeprefix('- ').strip()
    )
    if not commands or len(commands) > 6:
        raise ValueError('adapter command set must contain from one through six commands')
    return commands


def validate_chat_request(request: object) -> str:
    if not isinstance(request, dict):
        raise ValueError('request must be a JSON object')
    if request.get('temperature') != 0 or request.get('stream') is not False:
        raise ValueError('adapter requires temperature zero and stream false')
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


def runtime_device(requested: str) -> torch.device:
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if requested not in {'cpu', 'cuda'}:
        raise ValueError('device must be auto, cpu, or cuda')
    return torch.device(requested)
