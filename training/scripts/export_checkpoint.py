"""Export a shelliq Orbax LoRA checkpoint to PEFT/HF and optionally GGUF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax.numpy as jnp
from flax import nnx
from safetensors.flax import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.checkpoint import restore_checkpoint  # noqa: E402
from shelliq_training.config import Qwen2Config  # noqa: E402
from shelliq_training.data import Corpus  # noqa: E402
from shelliq_training.export import (  # noqa: E402
    LlamaCppToolchain,
    export_adapter_gguf,
    export_merged_hf_model,
    export_model_gguf,
    export_peft_adapter,
)
from shelliq_training.lora import DEFAULT_TARGETS, inject_lora  # noqa: E402
from shelliq_training.model import Qwen2ForCausalLM  # noqa: E402
from shelliq_training.training import create_lora_optimizer  # noqa: E402
from shelliq_training.weights import load_hf_state_dict  # noqa: E402

DEFAULT_MODEL_ID = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True, help='local Hugging Face snapshot directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=('adapter', 'merged'), required=True)
    parser.add_argument('--model-id', default=DEFAULT_MODEL_ID)
    parser.add_argument('--corpus', type=Corpus, required=True)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=32.0)
    parser.add_argument('--targets', default=','.join(DEFAULT_TARGETS))
    parser.add_argument('--gguf-output', type=Path)
    parser.add_argument('--llama-cpp', type=Path)
    parser.add_argument('--quantization', choices=('Q6_K', 'Q8_0'), default='Q8_0')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = args.base / 'model.safetensors'
    if not weights_path.is_file():
        raise SystemExit(f'only an unsharded local model.safetensors is currently supported: {weights_path}')
    model = Qwen2ForCausalLM(Qwen2Config(), param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    load_hf_state_dict(model, load_file(weights_path), jnp.bfloat16)
    targets = tuple(target.strip() for target in args.targets.split(',') if target.strip())
    inject_lora(model, rank=args.rank, alpha=args.alpha, targets=targets, rngs=nnx.Rngs(1))
    optimizer = create_lora_optimizer(model)
    checkpoint = restore_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        model_id=args.model_id,
        corpus=args.corpus,
    )
    if args.kind == 'adapter':
        metadata = export_peft_adapter(model, args.output, model_id=args.model_id, corpus=args.corpus)
    else:
        metadata = export_merged_hf_model(
            model,
            args.base,
            args.output,
            model_id=args.model_id,
            corpus=args.corpus,
        )
    if args.gguf_output is not None:
        if args.llama_cpp is None:
            raise SystemExit('--llama-cpp is required with --gguf-output')
        toolchain = LlamaCppToolchain(args.llama_cpp.resolve())
        if args.kind == 'adapter':
            export_adapter_gguf(args.output, args.base, args.gguf_output, toolchain=toolchain)
        else:
            export_model_gguf(
                args.output,
                args.gguf_output,
                toolchain=toolchain,
                quantization=args.quantization,
            )
    print(json.dumps({'checkpoint_step': checkpoint.step, **metadata.to_dict()}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
