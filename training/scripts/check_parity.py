"""Compare the JAX Qwen2 port with the Hugging Face PyTorch model.

This is the hard architecture gate before any fine-tuning work. Run it from
``training/`` with ``uv run python scripts/check_parity.py``.
"""

import sys
from pathlib import Path

# Direct script execution puts ``training/scripts`` on sys.path rather than
# the project root. Keep the documented command usable without an editable
# install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from huggingface_hub import hf_hub_download
from safetensors.flax import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from shelliq_training.config import Qwen2Config
from shelliq_training.model import Qwen2ForCausalLM
from shelliq_training.weights import load_hf_state_dict

MODEL_ID = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
PROMPT = 'Write a shell command that lists the five largest files.'
TOLERANCE = 1e-3


def hf_logits(prompt: str) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    with torch.no_grad():
        logits = model(input_ids).logits
    return input_ids.numpy(), logits.float().numpy()


def nnx_logits(input_ids: np.ndarray) -> np.ndarray:
    weights_path = hf_hub_download(MODEL_ID, 'model.safetensors')
    state_dict = load_file(weights_path)
    model = Qwen2ForCausalLM(Qwen2Config(), param_dtype=jnp.float32, rngs=nnx.Rngs(0))
    load_hf_state_dict(model, state_dict, param_dtype=jnp.float32)

    # NVIDIA GPUs use reduced-precision float32 dot products by default. That
    # is useful for training, but too imprecise for an architecture parity gate
    # against PyTorch CPU float32.
    with jax.default_matmul_precision('highest'):
        logits = model(jnp.asarray(input_ids))
    return np.asarray(logits)


def main() -> None:
    input_ids, hf_out = hf_logits(PROMPT)
    nnx_out = nnx_logits(input_ids)
    diff = np.abs(hf_out - nnx_out)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    print(f'JAX backend: {jax.default_backend()}')
    print(f'max abs diff: {max_diff:.6f}')
    print(f'mean abs diff: {mean_diff:.6f}')
    if max_diff < TOLERANCE:
        print(f'PASS: nnx port agrees with HF transformers within {TOLERANCE}.')
        return
    print(f'FAIL: nnx port diverges from HF transformers by at least {TOLERANCE}.')
    raise SystemExit(1)


if __name__ == '__main__':
    main()
