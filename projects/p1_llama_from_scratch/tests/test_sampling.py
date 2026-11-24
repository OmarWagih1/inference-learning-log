import pytest
import torch

from sampling import (
    sample_greedy,
    sample_min_p,
    sample_temperature,
    sample_top_k,
    sample_top_p,
)


@pytest.mark.parametrize(
    ("sampler", "arguments"),
    [
        (sample_greedy, {}),
        (sample_temperature, {"temperature": 0.8}),
        (sample_top_k, {"top_k": 5}),
        (sample_top_p, {"top_p": 0.9}),
        (sample_min_p, {"min_p": 0.05}),
    ],
)
def test_sampling_starter_blocks(
    sampler,
    arguments,
) -> None:
    logits = torch.tensor([[0.1, 2.0, 0.5], [1.0, 0.2, 0.7]])

    with pytest.raises(NotImplementedError):
        sampler(logits, **arguments)
