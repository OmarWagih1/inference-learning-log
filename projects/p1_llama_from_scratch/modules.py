import math
from typing import Any

import torch
from torch import Tensor, nn

from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states * torch.rsqrt(torch.mean(hidden_states * hidden_states, -1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

    def rotate_half(self, vector):
        half = vector.shape[-1] // 2
        first_half = vector[..., :half]
        second_half = vector[..., half:]

        return torch.cat([-second_half, first_half], dim=-1)


    def forward(
        self,
        query: Tensor,
        key: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:

        # calculate inverse frequencies

        even_indices = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=query.device)
        inv_freq = 1.0 / (self.rope_theta ** (even_indices / self.head_dim))


        # calculate angles
        angles = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq

        # calculate cos and sin 
        angles_concat = torch.cat([angles, angles], dim=-1)
        cos = angles_concat.cos().to(query.dtype).unsqueeze(1)
        sin = angles_concat.sin().to(query.dtype).unsqueeze(1)


        # rotate query and key
        return (query * cos + self.rotate_half(query) * sin, key * cos + self.rotate_half(key) * sin)

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_projection = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_projection = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def silu(self, z):
        return z * torch.sigmoid(z)

    def forward(self, hidden_states: Tensor) -> Tensor:
        # TODO: Implement SwiGLU:
        # silu(gate(hidden_states)) * up(hidden_states), then down-project.

        return self.down_projection(self.silu(self.gate_projection(hidden_states)) * self.up_projection(hidden_states) )


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        hidden_size = config.hidden_size
        query_projection_size = config.num_attention_heads * config.head_dim
        kv_projection_size = config.num_key_value_heads * config.head_dim

        self.query_projection = nn.Linear(
            hidden_size, query_projection_size, bias=False
        )
        self.key_projection = nn.Linear(
            hidden_size, kv_projection_size, bias=False
        )
        self.value_projection = nn.Linear(
            hidden_size, kv_projection_size, bias=False
        )
        self.output_projection = nn.Linear(
            query_projection_size, hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: Tensor,
        rotary_embedding: RotaryEmbedding,
        position_ids: Tensor,
        attention_mask: Tensor | None = None,
        kv_cache: Any | None = None,
    ) -> Tensor:
        # Project q/k/v.

        batch, seq_len, _ = hidden_states.shape

        Q, K, V = self.query_projection(hidden_states), self.key_projection(hidden_states), self.value_projection(hidden_states)
        Q = Q.view(Q.shape[0], Q.shape[1], self.num_attention_heads, self.head_dim).transpose(1, 2)
        K = K.view(K.shape[0], K.shape[1], self.num_key_value_heads, self.head_dim).transpose(1, 2)
        V = V.view(V.shape[0], V.shape[1], self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE.
        Q, K = rotary_embedding(Q, K, position_ids)

        # Use KV cache

        if kv_cache is not None:
            K, V = kv_cache.append(K, V)

        groups = self.num_attention_heads // self.num_key_value_heads
        K = K.repeat_interleave(groups, dim=1)
        V = V.repeat_interleave(groups, dim=1)

        # Add scores

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores += attention_mask
        elif attention_mask is None and seq_len > 1:
            causal_mask = torch.triu(torch.ones(seq_len, K.shape[2], device=scores.device, dtype=torch.bool), diagonal=K.shape[2] - hidden_states.shape[1] + 1).bool()
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Apply projections
        weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.output_projection(attn_output)
