import json
from pathlib import Path

import torch
from safetensors.torch import load_file
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
        normalized_states = self.attention_norm(hidden_states)
        attn_output = self.attention(normalized_states, rotary_embedding, position_ids, attention_mask, kv_cache)

        hidden_states = hidden_states + attn_output

        feed_forward_norm = self.feed_forward_norm(hidden_states)
        feed_forward = self.feed_forward(feed_forward_norm)

        hidden_states = hidden_states + feed_forward

        return hidden_states


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
        batch_size, sequence_length = input_ids.shape
        hidden_states = self.token_embedding(input_ids)

        ## START -  Written by Gemini Neon 4 low
        if position_ids is None:
            past_length = 0 if kv_caches is None else kv_caches[0].length
            positions = torch.arange(past_length, past_length + sequence_length, device=input_ids.device, dtype=torch.long)
            position_ids = positions.unsqueeze(0).expand(batch_size, sequence_length)

        for layer_index, block in enumerate(self.blocks):
            layer_cache = None if kv_caches is None else kv_caches[layer_index]
            hidden_states = block(
                hidden_states,
                self.rotary_embedding,
                position_ids,
                attention_mask,
                layer_cache,
            )

        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)
        ## End

    @classmethod
    def from_safetensors(
        cls,
        model_directory: str | Path,
    ) -> "LlamaStyleModel":

        ## Written by Gemini Neon 4 low, i was bored
        model_path = Path(model_directory)

        with (model_path / "config.json").open("r", encoding="utf-8") as config_file:
            raw_config = json.load(config_file)

        if raw_config.get("rope_scaling") is not None:
            raise NotImplementedError("rope_scaling is not supported yet")

        hidden_size = raw_config["hidden_size"]
        num_attention_heads = raw_config["num_attention_heads"]
        config = ModelConfig(
            vocab_size=raw_config["vocab_size"],
            hidden_size=hidden_size,
            intermediate_size=raw_config["intermediate_size"],
            num_hidden_layers=raw_config["num_hidden_layers"],
            num_attention_heads=num_attention_heads,
            num_key_value_heads=raw_config.get(
                "num_key_value_heads", num_attention_heads
            ),
            head_dim=raw_config.get(
                "head_dim", hidden_size // num_attention_heads
            ),
            max_position_embeddings=raw_config.get(
                "max_position_embeddings", 4096
            ),
            rms_norm_eps=raw_config.get("rms_norm_eps", 1e-6),
            rope_theta=raw_config.get("rope_theta", 10000.0),
        )

        safetensor_files = sorted(model_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(
                f"No .safetensors files found in {model_path}"
            )

        key_renames = (
            ("model.embed_tokens.", "token_embedding."),
            ("model.norm.", "final_norm."),
            ("model.layers.", "blocks."),
            ("input_layernorm", "attention_norm"),
            ("post_attention_layernorm", "feed_forward_norm"),
            ("self_attn.q_proj", "attention.query_projection"),
            ("self_attn.k_proj", "attention.key_projection"),
            ("self_attn.v_proj", "attention.value_projection"),
            ("self_attn.o_proj", "attention.output_projection"),
            ("mlp.gate_proj", "feed_forward.gate_projection"),
            ("mlp.up_proj", "feed_forward.up_projection"),
            ("mlp.down_proj", "feed_forward.down_projection"),
        )

        def rename_key(key: str) -> str:
            for hf_name, local_name in key_renames:
                key = key.replace(hf_name, local_name)
            return key

        mapped_weights: dict[str, Tensor] = {}
        for safetensor_file in safetensor_files:
            for key, tensor in load_file(str(safetensor_file)).items():
                mapped_weights[rename_key(key)] = tensor

        if "lm_head.weight" not in mapped_weights and raw_config.get(
            "tie_word_embeddings", False
        ):
            mapped_weights["lm_head.weight"] = mapped_weights[
                "token_embedding.weight"
            ]

        with torch.device("meta"):
            model = cls(config)

        model.load_state_dict(mapped_weights, strict=True, assign=True)
        return model
