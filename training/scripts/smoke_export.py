"""Export a zero-initialized real Qwen LoRA through PEFT, HF, and GGUF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
from flax import nnx
from safetensors.flax import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.config import Qwen2Config  # noqa: E402
from shelliq_training.data import Corpus  # noqa: E402
from shelliq_training.export import (  # noqa: E402
    LlamaCppToolchain,
    export_adapter_gguf,
    export_merged_hf_model,
    export_model_gguf,
    export_peft_adapter,
    smoke_test_gguf,
)
from shelliq_training.lora import inject_lora  # noqa: E402
from shelliq_training.model import Qwen2ForCausalLM  # noqa: E402
from shelliq_training.weights import load_hf_state_dict  # noqa: E402

MODEL_ID = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--llama-cpp', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quantization', choices=('Q6_K', 'Q8_0'), default='Q8_0')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'smoke output already exists: {output}')
    weights = args.base / 'model.safetensors'
    if not weights.is_file():
        raise FileNotFoundError(f'base weights not found: {weights}')
    output.mkdir(parents=True)

    model = Qwen2ForCausalLM(Qwen2Config(), param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    load_hf_state_dict(model, load_file(weights), jnp.bfloat16)
    inject_lora(model, rank=16, alpha=32, rngs=nnx.Rngs(1))
    toolchain = LlamaCppToolchain(args.llama_cpp.resolve())

    adapter = output / 'adapter'
    adapter_gguf = output / 'adapter-f16.gguf'
    merged = output / 'merged-hf'
    model_gguf = output / f'model-{args.quantization.lower()}.gguf'
    export_peft_adapter(model, adapter, model_id=MODEL_ID, corpus=Corpus.DISTRIBUTABLE)
    export_adapter_gguf(adapter, args.base, adapter_gguf, toolchain=toolchain)
    export_merged_hf_model(model, args.base, merged, model_id=MODEL_ID, corpus=Corpus.DISTRIBUTABLE)
    export_model_gguf(
        merged,
        model_gguf,
        toolchain=toolchain,
        quantization=args.quantization,
    )
    generated = smoke_test_gguf(model_gguf, toolchain=toolchain)
    print(f'PEFT adapter: {adapter}')
    print(f'GGUF adapter: {adapter_gguf}')
    print(f'merged HF: {merged}')
    print(f'quantized GGUF: {model_gguf}')
    print(f'llama-cli output: {generated.strip()}')


if __name__ == '__main__':
    main()
