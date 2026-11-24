import pytest
import torch
import torch.nn.functional as functional

from config import ModelConfig
from kv_cache import KVCache
from model import DecoderBlock, LlamaStyleModel
from modules import (
    FeedForward,
    GroupedQueryAttention,
    RMSNorm,
    RotaryEmbedding,
)
from sampling import (
    sample_greedy,
    sample_min_p,
    sample_temperature,
    sample_top_k,
    sample_top_p,
)


def make_tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )


def make_position_ids(batch_size: int, sequence_length: int) -> torch.Tensor:
    positions = torch.arange(sequence_length, dtype=torch.long)
    return positions.unsqueeze(0).expand(batch_size, sequence_length)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


def test_rms_norm_matches_pytorch_reference() -> None:
    torch.manual_seed(0)
    hidden_size = 8
    eps = 1e-6

    rms_norm = RMSNorm(hidden_size=hidden_size, eps=eps)
    rms_norm.weight.data = torch.linspace(0.5, 1.5, hidden_size)

    hidden_states = torch.randn(2, 3, hidden_size)
    actual_output = rms_norm(hidden_states)
    expected_output = functional.rms_norm(
        hidden_states,
        normalized_shape=(hidden_size,),
        weight=rms_norm.weight,
        eps=eps,
    )

    assert actual_output.shape == hidden_states.shape
    assert torch.allclose(actual_output, expected_output, atol=1e-5, rtol=1e-5)


def test_rms_norm_normalizes_each_token_independently() -> None:
    torch.manual_seed(1)
    rms_norm = RMSNorm(hidden_size=4, eps=1e-6)

    first_batch = torch.randn(1, 2, 4)
    modified_batch = first_batch.clone()
    modified_batch[0, 1] = modified_batch[0, 1] * 100.0

    first_output = rms_norm(first_batch)
    modified_output = rms_norm(modified_batch)

    assert torch.allclose(first_output[0, 0], modified_output[0, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# RotaryEmbedding (RoPE)
# ---------------------------------------------------------------------------


def test_rope_preserves_shape_and_vector_norms() -> None:
    torch.manual_seed(2)
    config = make_tiny_config()
    rotary_embedding = RotaryEmbedding(config)

    query = torch.randn(2, config.num_attention_heads, 5, config.head_dim)
    key = torch.randn(2, config.num_key_value_heads, 5, config.head_dim)
    position_ids = make_position_ids(batch_size=2, sequence_length=5)

    rotated_query, rotated_key = rotary_embedding(query, key, position_ids)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    assert torch.allclose(
        rotated_query.norm(dim=-1), query.norm(dim=-1), atol=1e-5
    )
    assert torch.allclose(
        rotated_key.norm(dim=-1), key.norm(dim=-1), atol=1e-5
    )


def test_rope_is_identity_at_position_zero_and_rotates_later_positions() -> None:
    torch.manual_seed(3)
    config = make_tiny_config()
    rotary_embedding = RotaryEmbedding(config)

    query = torch.randn(1, config.num_attention_heads, 3, config.head_dim)
    key = torch.randn(1, config.num_key_value_heads, 3, config.head_dim)
    position_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)

    rotated_query, rotated_key = rotary_embedding(query, key, position_ids)

    assert torch.allclose(rotated_query[:, :, 0, :], query[:, :, 0, :], atol=1e-5)
    assert torch.allclose(rotated_key[:, :, 0, :], key[:, :, 0, :], atol=1e-5)
    assert not torch.allclose(rotated_query[:, :, 1, :], query[:, :, 1, :])


# ---------------------------------------------------------------------------
# FeedForward (SwiGLU)
# ---------------------------------------------------------------------------


def test_feed_forward_matches_swiglu_formula() -> None:
    torch.manual_seed(4)
    config = make_tiny_config()
    feed_forward = FeedForward(config)

    hidden_states = torch.randn(2, 4, config.hidden_size)
    actual_output = feed_forward(hidden_states)

    gate_values = feed_forward.gate_projection(hidden_states)
    up_values = feed_forward.up_projection(hidden_states)
    expected_output = feed_forward.down_projection(
        functional.silu(gate_values) * up_values
    )

    assert actual_output.shape == hidden_states.shape
    assert torch.allclose(actual_output, expected_output, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# KVCache
# ---------------------------------------------------------------------------


def test_kv_cache_append_stores_and_returns_valid_prefix() -> None:
    torch.manual_seed(5)
    config = make_tiny_config()
    cache = KVCache.create(
        batch_size=2,
        max_sequence_length=8,
        config=config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    prefill_key = torch.randn(2, config.num_key_value_heads, 3, config.head_dim)
    prefill_value = torch.randn(2, config.num_key_value_heads, 3, config.head_dim)
    cached_key, cached_value = cache.append(prefill_key, prefill_value)

    assert cache.length == 3
    assert cached_key.shape == (2, config.num_key_value_heads, 3, config.head_dim)
    assert torch.allclose(cached_key, prefill_key)
    assert torch.allclose(cached_value, prefill_value)

    decode_key = torch.randn(2, config.num_key_value_heads, 1, config.head_dim)
    decode_value = torch.randn(2, config.num_key_value_heads, 1, config.head_dim)
    cached_key, cached_value = cache.append(decode_key, decode_value)

    expected_key = torch.cat([prefill_key, decode_key], dim=2)
    expected_value = torch.cat([prefill_value, decode_value], dim=2)

    assert cache.length == 4
    assert torch.allclose(cached_key, expected_key)
    assert torch.allclose(cached_value, expected_value)


def test_kv_cache_reset_restarts_from_zero() -> None:
    torch.manual_seed(6)
    config = make_tiny_config()
    cache = KVCache.create(
        batch_size=1,
        max_sequence_length=8,
        config=config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    first_key = torch.randn(1, config.num_key_value_heads, 2, config.head_dim)
    first_value = torch.randn(1, config.num_key_value_heads, 2, config.head_dim)
    cache.append(first_key, first_value)
    cache.reset()

    assert cache.length == 0

    second_key = torch.randn(1, config.num_key_value_heads, 1, config.head_dim)
    second_value = torch.randn(1, config.num_key_value_heads, 1, config.head_dim)
    cached_key, cached_value = cache.append(second_key, second_value)

    assert cache.length == 1
    assert torch.allclose(cached_key, second_key)
    assert torch.allclose(cached_value, second_value)


# ---------------------------------------------------------------------------
# GroupedQueryAttention
# ---------------------------------------------------------------------------


def test_grouped_query_attention_is_causal() -> None:
    torch.manual_seed(7)
    config = make_tiny_config()
    attention = GroupedQueryAttention(config)
    rotary_embedding = RotaryEmbedding(config)

    hidden_states = torch.randn(1, 4, config.hidden_size)
    modified_states = hidden_states.clone()
    modified_states[:, 3, :] = modified_states[:, 3, :] + 10.0
    position_ids = make_position_ids(batch_size=1, sequence_length=4)

    first_output = attention(hidden_states, rotary_embedding, position_ids)
    modified_output = attention(modified_states, rotary_embedding, position_ids)

    assert first_output.shape == hidden_states.shape
    assert torch.allclose(
        first_output[:, :3, :], modified_output[:, :3, :], atol=1e-5
    )


def test_grouped_query_attention_cached_decode_matches_full_pass() -> None:
    torch.manual_seed(8)
    config = make_tiny_config()
    attention = GroupedQueryAttention(config)
    rotary_embedding = RotaryEmbedding(config)

    hidden_states = torch.randn(1, 4, config.hidden_size)
    full_positions = make_position_ids(batch_size=1, sequence_length=4)
    full_output = attention(hidden_states, rotary_embedding, full_positions)

    cache = KVCache.create(
        batch_size=1,
        max_sequence_length=8,
        config=config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    prefill_positions = full_positions[:, :3]
    attention(hidden_states[:, :3, :], rotary_embedding, prefill_positions, kv_cache=cache)

    decode_positions = full_positions[:, 3:4]
    decode_output = attention(
        hidden_states[:, 3:4, :],
        rotary_embedding,
        decode_positions,
        kv_cache=cache,
    )

    assert torch.allclose(decode_output, full_output[:, 3:4, :], atol=1e-5)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sampling_greedy_and_temperature() -> None:
    logits = torch.tensor([[0.1, 5.0, 0.2], [4.0, 0.3, -1.0]])

    greedy_tokens = sample_greedy(logits)
    assert torch.equal(greedy_tokens, torch.tensor([1, 0]))

    low_temp_tokens = sample_temperature(logits, temperature=1e-4)
    assert torch.equal(low_temp_tokens, torch.tensor([1, 0]))

    with pytest.raises(ValueError):
        sample_temperature(logits, temperature=0.0)


def test_sampling_top_k_top_p_and_min_p() -> None:
    torch.manual_seed(9)
    logits = torch.tensor([[0.0, 20.0, -5.0, -10.0]])

    assert sample_top_k(logits, top_k=1).item() == 1
    assert sample_top_p(logits, top_p=0.5).item() == 1
    assert sample_min_p(logits, min_p=0.5).item() == 1


# ---------------------------------------------------------------------------
# DecoderBlock & LlamaStyleModel
# ---------------------------------------------------------------------------


def test_decoder_block_preserves_shape_and_residual_path() -> None:
    torch.manual_seed(10)
    config = make_tiny_config()
    block = DecoderBlock(config)
    rotary_embedding = RotaryEmbedding(config)

    with torch.no_grad():
        block.attention.output_projection.weight.zero_()
        block.feed_forward.down_projection.weight.zero_()

    hidden_states = torch.randn(2, 4, config.hidden_size)
    position_ids = make_position_ids(batch_size=2, sequence_length=4)

    output = block(
        hidden_states=hidden_states,
        rotary_embedding=rotary_embedding,
        position_ids=position_ids,
        attention_mask=None,
        kv_cache=None,
    )

    assert output.shape == hidden_states.shape
    assert torch.allclose(output, hidden_states, atol=1e-6)


def test_llama_model_forward_and_cached_decode() -> None:
    torch.manual_seed(11)
    config = make_tiny_config()
    model = LlamaStyleModel(config)
    model.eval()

    input_ids = torch.tensor([[1, 5, 9, 12]], dtype=torch.long)

    with torch.no_grad():
        full_logits = model(input_ids)

    assert full_logits.shape == (1, 4, config.vocab_size)
    assert torch.isfinite(full_logits).all()

    caches = [
        KVCache.create(
            batch_size=1,
            max_sequence_length=8,
            config=config,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for _ in range(config.num_hidden_layers)
    ]

    with torch.no_grad():
        model(input_ids[:, :3], kv_caches=caches)
        decode_logits = model(input_ids[:, 3:4], kv_caches=caches)

    assert torch.allclose(decode_logits, full_logits[:, 3:4, :], atol=1e-4)
