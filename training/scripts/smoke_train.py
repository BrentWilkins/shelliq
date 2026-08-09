"""Run one real Qwen2.5-Coder-0.5B LoRA step and report GPU memory.

Run from ``training/`` with ``uv run python scripts/smoke_train.py``.
This is a hardware/integration smoke test, not a useful training run.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import hf_hub_download
from safetensors.flax import load_file
from transformers import AutoTokenizer

from shelliq_training.config import Qwen2Config
from shelliq_training.data import Corpus, Platform, SFTRecord, collate_sft, tokenize_record, tokenizer_pad_id
from shelliq_training.lora import inject_lora, parameter_count
from shelliq_training.model import Qwen2ForCausalLM
from shelliq_training.prompt import PromptContract
from shelliq_training.training import create_lora_optimizer, train_step
from shelliq_training.weights import load_hf_state_dict

MODEL_ID = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
SEQUENCE_LENGTH = 128


def gibibytes(byte_count: int) -> float:
    return byte_count / 1024**3


def gpu_memory() -> tuple[float, float]:
    stats = jax.devices()[0].memory_stats() or {}
    current = int(stats.get('bytes_in_use', 0))
    peak = int(stats.get('peak_bytes_in_use', current))
    return gibibytes(current), gibibytes(peak)


def make_batch() -> dict[str, jax.Array]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    record = SFTRecord(
        record_id='smoke:find:largest-files',
        corpus=Corpus.DISTRIBUTABLE,
        source='smoke-test',
        license='CC0-1.0',
        provenance='training/scripts/smoke_train.py',
        command='find',
        platform=Platform.LINUX,
        instruction='List the five largest files under the current directory.',
        response="find . -type f -printf '%s %p\\n' | sort -nr | head -5",
        context='GNU find supports -type f and -printf.',
    )
    example = tokenize_record(
        record,
        tokenizer,
        max_length=SEQUENCE_LENGTH,
        prompt_contract=PromptContract.LEGACY_USER_V1,
    )
    return collate_sft(
        [example],
        sequence_length=SEQUENCE_LENGTH,
        pad_token_id=tokenizer_pad_id(tokenizer),
    )


def main() -> None:
    if jax.default_backend() != 'gpu':
        raise SystemExit('smoke_train.py requires a JAX GPU backend')

    model = Qwen2ForCausalLM(Qwen2Config(), param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    weights_path = hf_hub_download(MODEL_ID, 'model.safetensors')
    load_hf_state_dict(model, load_file(weights_path), param_dtype=jnp.bfloat16)
    inject_lora(model, rank=16, alpha=32, rngs=nnx.Rngs(1))
    optimizer = create_lora_optimizer(model)
    batch = make_batch()

    lora_parameters = parameter_count(model, nnx.LoRAParam)
    all_parameters = parameter_count(model, nnx.Param)
    print(f'device: {jax.devices()[0]}')
    print(f'parameters: {all_parameters:,} total; {lora_parameters:,} LoRA')
    print(f'batch: 1 x {SEQUENCE_LENGTH}; base dtype: bfloat16')

    started = time.perf_counter()
    first_loss = train_step(model, optimizer, batch)
    np.asarray(first_loss)
    first_seconds = time.perf_counter() - started
    current, peak = gpu_memory()
    print(
        f'compile + first step: {first_seconds:.2f}s; loss {float(first_loss):.4f}; memory {current:.2f} GiB, peak {peak:.2f} GiB'
    )

    started = time.perf_counter()
    second_loss = train_step(model, optimizer, batch)
    np.asarray(second_loss)
    second_seconds = time.perf_counter() - started
    current, peak = gpu_memory()
    print(f'second step: {second_seconds:.3f}s; loss {float(second_loss):.4f}; memory {current:.2f} GiB, peak {peak:.2f} GiB')


if __name__ == '__main__':
    main()
