from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    use_qk_norm: bool = False
    rope_theta: float = 1000000.0

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
