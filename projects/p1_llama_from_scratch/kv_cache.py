from dataclasses import dataclass

import torch
from torch import Tensor

from config import ModelConfig


@dataclass
class KVCache:
    key: Tensor
    value: Tensor
    length: int = 0

    @classmethod
    def create(
        cls,
        batch_size: int,
        max_sequence_length: int,
        config: ModelConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "KVCache":
        cache_shape = (
            batch_size,
            config.num_key_value_heads,
            max_sequence_length,
            config.head_dim,
        )
        key = torch.empty(cache_shape, device=device, dtype=dtype)
        value = torch.empty(cache_shape, device=device, dtype=dtype)
        return cls(key=key, value=value)

    def append(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        # TODO: Preallocate already; copy new K/V into key/value and increase length.
        # TODO: Return all valid K/V tensors up to the current length.
        raise NotImplementedError("Implement KVCache.append")

    def reset(self) -> None:
        self.length = 0
