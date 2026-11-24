import pytest

from config import ModelConfig


def test_config_accepts_divisible_head_sizes() -> None:
    config = ModelConfig(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    config.validate()


def test_config_rejects_invalid_head_counts() -> None:
    config = ModelConfig(
        hidden_size=127,
        num_attention_heads=8,
        num_key_value_heads=2,
    )

    with pytest.raises(ValueError):
        config.validate()
