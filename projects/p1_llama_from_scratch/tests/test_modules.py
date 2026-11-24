import pytest
import torch

from config import ModelConfig
from modules import (
    FeedForward,
    GroupedQueryAttention,
    RMSNorm,
    RotaryEmbedding,
)


def test_rms_norm_has_a_starter_block() -> None:
    module = RMSNorm(hidden_size=4)
    hidden_states = torch.ones(2, 3, 4)

    with pytest.raises(NotImplementedError):
        module(hidden_states)


def test_rope_has_a_starter_block() -> None:
    config = ModelConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    module = RotaryEmbedding(config)
    query = torch.zeros(2, 4, 3, 4)
    key = torch.zeros(2, 2, 3, 4)
    position_ids = torch.arange(3).repeat(2, 1)

    with pytest.raises(NotImplementedError):
        module(query, key, position_ids)


def test_swiglu_has_a_starter_block() -> None:
    config = ModelConfig(
        hidden_size=16,
        intermediate_size=32,
    )
    module = FeedForward(config)
    hidden_states = torch.ones(2, 3, 16)

    with pytest.raises(NotImplementedError):
        module(hidden_states)


def test_grouped_query_attention_has_a_starter_block() -> None:
    config = ModelConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    module = GroupedQueryAttention(config)
    hidden_states = torch.ones(2, 3, 16)
    rotary_embedding = RotaryEmbedding(config)
    position_ids = torch.arange(3).repeat(2, 1)

    with pytest.raises(NotImplementedError):
        module(hidden_states, rotary_embedding, position_ids)
