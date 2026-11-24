from pathlib import Path

from torch import Tensor, nn

from config import ModelConfig
from kv_cache import KVCache
from modules import (
    FeedForward,
    GroupedQueryAttention,
    RMSNorm,
    RotaryEmbedding,
)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.feed_forward_norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        hidden_states: Tensor,
        rotary_embedding: RotaryEmbedding,
        position_ids: Tensor,
        attention_mask: Tensor | None,
        kv_cache: KVCache | None,
    ) -> Tensor:
        # TODO: Pre-norm residual for attention.
        # TODO: Pre-norm residual for the feed-forward block.
        raise NotImplementedError(
            "Implement DecoderBlock.forward"
        )


class LlamaStyleModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_embedding = RotaryEmbedding(config)
        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        kv_caches: list[KVCache] | None = None,
    ) -> Tensor:
        # TODO: Embed tokens, run each decoder block with its layer KV cache,
        # apply final norm, and project to vocab logits.
        raise NotImplementedError(
            "Implement LlamaStyleModel.forward"
        )

    @classmethod
    def from_safetensors(
        cls,
        model_directory: str | Path,
    ) -> "LlamaStyleModel":
        # TODO: Load config.json and safetensors weights.
        # TODO: Do not use transformers modeling code.
        raise NotImplementedError(
            "Implement direct safetensors weight loading"
        )
