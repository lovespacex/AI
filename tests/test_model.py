import torch

from minichat.config import ModelConfig
from minichat.model import MiniChatModel


def test_small_model_forward_and_backward() -> None:
    config = ModelConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=176,
        max_seq_len=32,
        gradient_checkpointing=True,
    )
    model = MiniChatModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(input_ids, input_ids)
    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.token_embeddings.weight.grad is not None


def test_200m_configuration_size() -> None:
    config = ModelConfig()
    with torch.device("meta"):
        model = MiniChatModel(config)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 195_000_000 <= parameters <= 205_000_000

