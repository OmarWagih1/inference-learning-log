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
        # TODO: Implement RMSNorm:
        # return hidden_states * rsqrt(mean(hidden_states^2) + eps) * weight
        raise NotImplementedError("Implement RMSNorm in modules.RMSNorm.forward")


class RotaryEmbedding(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # TODO: Compute cosine/sine frequencies for position_ids.
        # TODO: Rotate the query and key last dimension in pairs.
        raise NotImplementedError(
            "Implement RoPE in modules.RotaryEmbedding.forward"
        )


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

    def forward(self, hidden_states: Tensor) -> Tensor:
        # TODO: Implement SwiGLU:
        # silu(gate(hidden_states)) * up(hidden_states), then down-project.
        raise NotImplementedError(
            "Implement SwiGLU in modules.FeedForward.forward"
        )


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
        # TODO: Project q/k/v.
        # TODO: Apply RoPE.
        # TODO: Expand or repeat K/V heads for each query head.
        # TODO: Update kv_cache and compute scaled dot-product attention.
        # TODO: Reshape and apply the output projection.
        raise NotImplementedError(
            "Implement GQA in modules.GroupedQueryAttention.forward"
        )
