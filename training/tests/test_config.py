import json

from shelliq_training.config import Qwen2Config


def test_qwen_config_loads_architecture_fields_from_json(tmp_path):
    document = {
        'vocab_size': 100,
        'hidden_size': 48,
        'intermediate_size': 96,
        'num_hidden_layers': 3,
        'num_attention_heads': 6,
        'num_key_value_heads': 2,
        'max_position_embeddings': 1024,
        'rope_theta': 10000.0,
        'rms_norm_eps': 1e-5,
        'tie_word_embeddings': False,
        'ignored_upstream_field': 'safe',
    }
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(document))

    config = Qwen2Config.from_json(path)

    assert config.hidden_size == 48
    assert config.num_hidden_layers == 3
    assert config.head_dim == 8
    assert config.tie_word_embeddings is False
