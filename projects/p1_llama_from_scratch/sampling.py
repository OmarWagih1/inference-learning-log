import torch
from torch import Tensor


def sample_greedy(logits: Tensor) -> Tensor:
    # TODO: Return the highest-logit token for every batch item.
    raise NotImplementedError("Implement greedy sampling")


def sample_temperature(logits: Tensor, temperature: float) -> Tensor:
    # TODO: Validate temperature and sample from softmax(logits / temperature).
    raise NotImplementedError("Implement temperature sampling")


def sample_top_k(logits: Tensor, top_k: int) -> Tensor:
    # TODO: Set logits outside the largest top_k values to negative infinity.
    raise NotImplementedError("Implement top-k sampling")


def sample_top_p(logits: Tensor, top_p: float) -> Tensor:
    # TODO: Sort logits/probabilities and keep the smallest set with mass >= top_p.
    raise NotImplementedError("Implement top-p sampling")


def sample_min_p(logits: Tensor, min_p: float) -> Tensor:
    # TODO: Keep tokens whose probability is at least min_p * maximum probability.
    raise NotImplementedError("Implement min-p sampling")
