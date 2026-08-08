import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from safetensors.numpy import load_file

from shelliq_training.config import Qwen2Config
from shelliq_training.data import Corpus
from shelliq_training.export import (
    LlamaCppToolchain,
    export_adapter_gguf,
    export_merged_hf_model,
    export_model_gguf,
    export_peft_adapter,
)
from shelliq_training.lora import inject_lora
from shelliq_training.model import Qwen2ForCausalLM


def tiny_model(param_dtype=jnp.float32):
    config = Qwen2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = Qwen2ForCausalLM(config, param_dtype=param_dtype, rngs=nnx.Rngs(0))
    inject_lora(model, rank=2, alpha=4, rngs=nnx.Rngs(1))
    model.model.layers[0].self_attn.q_proj.adapter.lora_b[...] = 1
    return model


def base_directory(path):
    path.mkdir()
    (path / 'config.json').write_text('{"model_type":"qwen2"}\n', encoding='utf-8')
    (path / 'tokenizer.json').write_text('{}\n', encoding='utf-8')


def test_peft_adapter_has_hf_shapes_metadata_and_no_base_weights(tmp_path):
    model = tiny_model()
    output = tmp_path / 'adapter'

    metadata = export_peft_adapter(
        model,
        output,
        model_id='Qwen/tiny',
        corpus=Corpus.PERSONAL,
    )

    tensors = load_file(output / 'adapter_model.safetensors')
    key = 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'
    assert tensors[key].shape == (2, 8)
    assert all('base_layer' not in name for name in tensors)
    assert metadata.adapter_parameters == sum(value.size for value in tensors.values())
    config = json.loads((output / 'adapter_config.json').read_text())
    shelliq = json.loads((output / 'shelliq_metadata.json').read_text())
    assert config['base_model_name_or_path'] == 'Qwen/tiny'
    assert shelliq['publishable'] is False
    with pytest.raises(FileExistsError):
        export_peft_adapter(model, output, model_id='Qwen/tiny', corpus=Corpus.PERSONAL)


def test_merged_hf_export_applies_lora_delta_and_marks_publication_policy(tmp_path):
    model = tiny_model()
    base = tmp_path / 'base'
    base_directory(base)
    output = tmp_path / 'merged'
    q_proj = model.model.layers[0].self_attn.q_proj
    expected = np.asarray(
        q_proj.base_module.kernel[...] + q_proj.scale * (q_proj.adapter.lora_a[...] @ q_proj.adapter.lora_b[...])
    ).T

    export_merged_hf_model(
        model,
        base,
        output,
        model_id='Qwen/tiny',
        corpus=Corpus.DISTRIBUTABLE,
    )

    tensors = load_file(output / 'model.safetensors')
    np.testing.assert_allclose(tensors['model.layers.0.self_attn.q_proj.weight'], expected)
    assert (output / 'tokenizer.json').is_file()
    metadata = json.loads((output / 'shelliq_metadata.json').read_text())
    assert metadata['publishable'] is True


def test_merged_hf_export_supports_bfloat16_base(tmp_path):
    model = tiny_model(jnp.bfloat16)
    base = tmp_path / 'base'
    base_directory(base)

    export_merged_hf_model(
        model,
        base,
        tmp_path / 'merged',
        model_id='Qwen/tiny',
        corpus=Corpus.DISTRIBUTABLE,
    )

    tensors = load_file(tmp_path / 'merged' / 'model.safetensors')
    assert tensors['model.embed_tokens.weight'].dtype.name == 'bfloat16'
    assert tensors['model.layers.0.self_attn.q_proj.weight'].dtype.name == 'bfloat16'


def fake_toolchain(path):
    (path / 'build' / 'bin').mkdir(parents=True)
    for relative in ('convert_hf_to_gguf.py', 'convert_lora_to_gguf.py', 'build/bin/llama-quantize'):
        target = path / relative
        target.write_text('# fake\n', encoding='utf-8')
    return LlamaCppToolchain(path)


def gguf_runner(commands):
    commands = list(commands)
    output = Path(commands[commands.index('--outfile') + 1]) if '--outfile' in commands else Path(commands[2])
    output.write_bytes(b'GGUFfake')


def test_gguf_wrappers_use_local_tools_and_approved_quantization(tmp_path):
    toolchain = fake_toolchain(tmp_path / 'llama.cpp')
    base = tmp_path / 'base'
    base_directory(base)
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_model.safetensors').write_bytes(b'weights')
    adapter_output = tmp_path / 'adapter.gguf'

    export_adapter_gguf(adapter, base, adapter_output, toolchain=toolchain, runner=gguf_runner)
    assert adapter_output.read_bytes().startswith(b'GGUF')

    merged = tmp_path / 'merged'
    base_directory(merged)
    (merged / 'model.safetensors').write_bytes(b'weights')
    model_output = tmp_path / 'model-q8.gguf'
    export_model_gguf(merged, model_output, toolchain=toolchain, quantization='Q8_0', runner=gguf_runner)
    assert model_output.read_bytes().startswith(b'GGUF')
    with pytest.raises(ValueError, match='Q6_K'):
        export_model_gguf(
            merged,
            tmp_path / 'model-q4.gguf',
            toolchain=toolchain,
            quantization='Q4_K_M',
            runner=gguf_runner,
        )
