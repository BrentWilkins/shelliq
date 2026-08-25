import torch

from shelliq_training.compiler_data import CompilerBatch
from shelliq_training.semantic_compiler import CompilerConfig, SemanticCompiler


def tiny_config() -> CompilerConfig:
    return CompilerConfig(
        vocab_size=16,
        pad_token_id=0,
        decoder_start_token_id=1,
        eos_token_id=2,
        d_model=16,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        feedforward_size=32,
        dropout=0,
        max_source_length=8,
        max_target_length=8,
    )


def tiny_batch() -> CompilerBatch:
    return CompilerBatch(
        source_ids=torch.tensor([[4, 5, 2, 0]]),
        source_attention_mask=torch.tensor([[True, True, True, False]]),
        decoder_input_ids=torch.tensor([[1, 6, 7, 8]]),
        decoder_attention_mask=torch.ones((1, 4), dtype=torch.bool),
        labels=torch.tensor([[6, 7, 8, 2]]),
    )


def test_compiler_forward_has_vocabulary_logits_and_tied_embeddings():
    torch.manual_seed(7)
    model = SemanticCompiler(tiny_config())
    batch = tiny_batch()

    logits = model(
        batch.source_ids,
        batch.source_attention_mask,
        batch.decoder_input_ids,
        batch.decoder_attention_mask,
    )

    assert logits.shape == (1, 4, 16)
    assert model.output_projection.weight is model.token_embedding.weight
    assert torch.isfinite(model.loss(batch))


def test_cpu_compiler_can_overfit_and_greedy_decode_tiny_batch():
    torch.manual_seed(11)
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model = SemanticCompiler(tiny_config())
        batch = tiny_batch()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0)
        initial_loss = model.loss(batch).item()

        model.train()
        for _ in range(120):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            loss.backward()
            optimizer.step()

        model.eval()
        generated = model.generate(batch.source_ids, batch.source_attention_mask, max_new_tokens=4)

        assert model.loss(batch).item() < initial_loss * 0.02
        assert generated.tolist() == [[6, 7, 8, 2]]
    finally:
        torch.set_num_threads(original_threads)
