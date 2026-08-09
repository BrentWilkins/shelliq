"""PEFT adapter, merged Hugging Face, and llama.cpp GGUF export."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np
from flax import nnx
from safetensors.numpy import save_file

from shelliq_training.checkpoint import TargetFormat, lora_signature
from shelliq_training.data import Corpus
from shelliq_training.lora import LoRALinear
from shelliq_training.model import Qwen2ForCausalLM
from shelliq_training.prompt import PromptContract

EXPORT_SCHEMA_VERSION = 2
SUPPORTED_QUANTIZATIONS = frozenset({'Q6_K', 'Q8_0'})
_HF_COPY_FILES = (
    'config.json',
    'generation_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'merges.txt',
    'vocab.json',
    'chat_template.jinja',
)


class ExportError(RuntimeError):
    """An export artifact is incomplete or incompatible."""


@dataclass(frozen=True, slots=True)
class ExportMetadata:
    model_id: str
    corpus: Corpus
    target_format: TargetFormat
    prompt_contract: PromptContract
    rank: int
    alpha: float
    targets: tuple[str, ...]
    adapter_parameters: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result['schema_version'] = EXPORT_SCHEMA_VERSION
        result['corpus'] = self.corpus.value
        result['target_format'] = self.target_format.value
        result['prompt_contract'] = self.prompt_contract.value
        result['publishable'] = self.corpus is Corpus.DISTRIBUTABLE
        return result


@dataclass(frozen=True, slots=True)
class LlamaCppToolchain:
    root: Path

    @property
    def convert_hf(self) -> Path:
        return self.root / 'convert_hf_to_gguf.py'

    @property
    def convert_lora(self) -> Path:
        return self.root / 'convert_lora_to_gguf.py'

    @property
    def quantize(self) -> Path:
        return self.root / 'build' / 'bin' / 'llama-quantize'

    @property
    def cli(self) -> Path:
        return self.root / 'build' / 'bin' / 'llama-cli'


CommandRunner = Callable[[Sequence[str]], None]


def export_peft_adapter(
    model: Qwen2ForCausalLM,
    output_directory: str | Path,
    *,
    model_id: str,
    corpus: Corpus,
    target_format: TargetFormat,
    prompt_contract: PromptContract,
) -> ExportMetadata:
    """Write an immutable PEFT adapter directory without any base weights."""
    output = Path(output_directory).resolve()
    _require_new_path(output)
    rank, alpha, targets = lora_signature(model)
    tensors = _adapter_tensors(model)
    parameter_count = sum(int(tensor.size) for tensor in tensors.values())
    metadata = ExportMetadata(model_id, corpus, target_format, prompt_contract, rank, alpha, targets, parameter_count)

    with _staged_directory(output) as stage:
        save_file(
            tensors,
            stage / 'adapter_model.safetensors',
            metadata={
                'format': 'pt',
                'shelliq_corpus': corpus.value,
                'shelliq_publishable': str(corpus is Corpus.DISTRIBUTABLE).lower(),
            },
        )
        adapter_config = {
            'base_model_name_or_path': model_id,
            'bias': 'none',
            'fan_in_fan_out': False,
            'inference_mode': True,
            'lora_alpha': alpha,
            'lora_dropout': 0.0,
            'peft_type': 'LORA',
            'r': rank,
            'target_modules': list(targets),
            'task_type': 'CAUSAL_LM',
        }
        _write_json(stage / 'adapter_config.json', adapter_config)
        _write_json(stage / 'shelliq_metadata.json', metadata.to_dict())
    return metadata


def export_merged_hf_model(
    model: Qwen2ForCausalLM,
    base_model_directory: str | Path,
    output_directory: str | Path,
    *,
    model_id: str,
    corpus: Corpus,
    target_format: TargetFormat,
    prompt_contract: PromptContract,
) -> ExportMetadata:
    """Merge LoRA into base kernels and write a llama.cpp-convertible HF directory."""
    base = Path(base_model_directory).resolve()
    output = Path(output_directory).resolve()
    _require_new_path(output)
    if not (base / 'config.json').is_file():
        raise ExportError(f'base model directory has no config.json: {base}')
    rank, alpha, targets = lora_signature(model)
    tensors = _merged_hf_tensors(model)
    adapter_parameters = sum(int(tensor.size) for tensor in _adapter_tensors(model).values())
    metadata = ExportMetadata(model_id, corpus, target_format, prompt_contract, rank, alpha, targets, adapter_parameters)

    with _staged_directory(output) as stage:
        copied = 0
        for filename in _HF_COPY_FILES:
            source = base / filename
            if source.is_file():
                shutil.copy2(source, stage / filename)
                copied += 1
        if copied == 0:
            raise ExportError(f'base model directory has no supported metadata files: {base}')
        save_file(
            tensors,
            stage / 'model.safetensors',
            metadata={'format': 'pt'},
        )
        _write_json(stage / 'shelliq_metadata.json', metadata.to_dict())
    return metadata


def export_adapter_gguf(
    adapter_directory: str | Path,
    base_model_directory: str | Path,
    output_path: str | Path,
    *,
    toolchain: LlamaCppToolchain,
    outtype: str = 'f16',
    runner: CommandRunner | None = None,
) -> None:
    """Convert a PEFT directory into a llama.cpp runtime LoRA adapter."""
    adapter = Path(adapter_directory).resolve()
    base = Path(base_model_directory).resolve()
    output = Path(output_path).resolve()
    _require_new_path(output)
    _require_file(toolchain.convert_lora, 'LoRA converter')
    if not (adapter / 'adapter_model.safetensors').is_file():
        raise ExportError(f'PEFT adapter weights missing: {adapter}')
    if not (base / 'config.json').is_file():
        raise ExportError(f'base model config missing: {base}')
    command_runner = runner or _subprocess_runner
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.shelliq-lora-', dir=output.parent) as temporary:
        staged_output = Path(temporary) / output.name
        command_runner(
            (
                sys.executable,
                str(toolchain.convert_lora),
                '--base',
                str(base),
                '--outfile',
                str(staged_output),
                '--outtype',
                outtype,
                str(adapter),
            )
        )
        _validate_gguf(staged_output)
        staged_output.replace(output)


def export_model_gguf(
    merged_hf_directory: str | Path,
    output_path: str | Path,
    *,
    toolchain: LlamaCppToolchain,
    quantization: str,
    runner: CommandRunner | None = None,
) -> None:
    """Convert merged HF weights and quantize only to the approved Q6_K/Q8_0 formats."""
    if quantization not in SUPPORTED_QUANTIZATIONS:
        raise ValueError(f'quantization must be one of {sorted(SUPPORTED_QUANTIZATIONS)}')
    merged = Path(merged_hf_directory).resolve()
    output = Path(output_path).resolve()
    _require_new_path(output)
    _require_file(toolchain.convert_hf, 'HF converter')
    _require_file(toolchain.quantize, 'llama-quantize')
    if not (merged / 'model.safetensors').is_file() or not (merged / 'config.json').is_file():
        raise ExportError(f'merged Hugging Face directory is incomplete: {merged}')
    command_runner = runner or _subprocess_runner
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.shelliq-gguf-', dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        f16 = temporary_path / 'model-f16.gguf'
        quantized = temporary_path / output.name
        command_runner(
            (
                sys.executable,
                str(toolchain.convert_hf),
                str(merged),
                '--outfile',
                str(f16),
                '--outtype',
                'f16',
            )
        )
        _validate_gguf(f16)
        command_runner((str(toolchain.quantize), str(f16), str(quantized), quantization))
        _validate_gguf(quantized)
        quantized.replace(output)


def smoke_test_gguf(
    model_path: str | Path,
    *,
    toolchain: LlamaCppToolchain,
    prompt: str = 'Respond with the word OK.',
    timeout_seconds: float = 120.0,
) -> str:
    """Actually load a GGUF with llama-cli and generate one token sequence."""
    model = Path(model_path).resolve()
    _validate_gguf(model)
    _require_file(toolchain.cli, 'llama-cli')
    result = subprocess.run(
        (str(toolchain.cli), '-m', str(model), '-p', prompt, '-n', '8', '--no-display-prompt'),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return result.stdout


def _adapter_tensors(model: Qwen2ForCausalLM) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for layer_number, layer in enumerate(model.model.layers):
        modules = {
            'self_attn.q_proj': layer.self_attn.q_proj,
            'self_attn.k_proj': layer.self_attn.k_proj,
            'self_attn.v_proj': layer.self_attn.v_proj,
            'self_attn.o_proj': layer.self_attn.o_proj,
            'mlp.gate_proj': layer.mlp.gate_proj,
            'mlp.up_proj': layer.mlp.up_proj,
            'mlp.down_proj': layer.mlp.down_proj,
        }
        for module_name, module in modules.items():
            if not isinstance(module, LoRALinear):
                continue
            prefix = f'base_model.model.model.layers.{layer_number}.{module_name}'
            tensors[f'{prefix}.lora_A.weight'] = _numpy(module.adapter.lora_a[...].T)
            tensors[f'{prefix}.lora_B.weight'] = _numpy(module.adapter.lora_b[...].T)
    if not tensors:
        raise ExportError('model has no LoRA adapters')
    return tensors


def _merged_hf_tensors(model: Qwen2ForCausalLM) -> dict[str, np.ndarray]:
    tensors = {
        'model.embed_tokens.weight': _numpy(model.model.embed_tokens.embedding[...]),
        'model.norm.weight': _numpy(model.model.norm.weight[...]),
    }
    for layer_number, layer in enumerate(model.model.layers):
        prefix = f'model.layers.{layer_number}.'
        modules = {
            'self_attn.q_proj': layer.self_attn.q_proj,
            'self_attn.k_proj': layer.self_attn.k_proj,
            'self_attn.v_proj': layer.self_attn.v_proj,
            'self_attn.o_proj': layer.self_attn.o_proj,
            'mlp.gate_proj': layer.mlp.gate_proj,
            'mlp.up_proj': layer.mlp.up_proj,
            'mlp.down_proj': layer.mlp.down_proj,
        }
        for module_name, module in modules.items():
            tensors[f'{prefix}{module_name}.weight'] = _merged_linear_weight(module)
            base_module = module.base_module if isinstance(module, LoRALinear) else module
            if base_module.bias is not None:
                tensors[f'{prefix}{module_name}.bias'] = _numpy(base_module.bias[...])
        tensors[f'{prefix}input_layernorm.weight'] = _numpy(layer.input_layernorm.weight[...])
        tensors[f'{prefix}post_attention_layernorm.weight'] = _numpy(layer.post_attention_layernorm.weight[...])
    return tensors


def _merged_linear_weight(module: nnx.Linear | LoRALinear) -> np.ndarray:
    if isinstance(module, LoRALinear):
        kernel = module.base_module.kernel[...]
        delta = (module.scale * (module.adapter.lora_a[...] @ module.adapter.lora_b[...])).astype(kernel.dtype)
        return _numpy((kernel + delta).T)
    return _numpy(module.kernel[...].T)


def _numpy(value: jax.Array) -> np.ndarray:
    return np.asarray(jax.device_get(value))


class _staged_directory:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.stage: Path | None = None

    def __enter__(self) -> Path:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.stage = Path(tempfile.mkdtemp(prefix=f'.{self.output.name}-', dir=self.output.parent))
        return self.stage

    def __exit__(self, error_type: object, error: object, traceback: object) -> None:
        assert self.stage is not None
        if error_type is None:
            self.stage.replace(self.output)
        else:
            shutil.rmtree(self.stage, ignore_errors=True)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _subprocess_runner(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def _require_new_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f'export path already exists: {path}')


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise ExportError(f'{description} not found: {path}')


def _validate_gguf(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 4:
        raise ExportError(f'GGUF output was not created: {path}')
    with path.open('rb') as stream:
        if stream.read(4) != b'GGUF':
            raise ExportError(f'output does not have GGUF magic: {path}')
