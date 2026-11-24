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

        new_token = key.shape[2]
        start = self.length
        end = start + new_token
        if end > self.key.shape[2]:
            raise ValueError("more Ks than we can hold")

        self.key[:, :, start:end, :] = key
        self.value[:, :, start:end, :] = value
        self.length = end

        return self.key[:, :, :self.length, :], self.value[:, :, :self.length, :]

    def reset(self) -> None:
        self.length = 0
