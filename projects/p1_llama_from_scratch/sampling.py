import torch
from torch import Tensor


def sample_greedy(logits: Tensor) -> Tensor:
    return torch.argmax(logits, dim=-1)


def sample_temperature(logits: Tensor, temperature: float) -> Tensor:
    if temperature <= 0:
        raise ValueError("Temp can't be that low, we freezing out here")

    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

def sample_top_k(logits: Tensor, top_k: int) -> Tensor:
    if top_k <= 0: 
        raise ValueError("Can't have that")
    k = min(top_k, logits.shape[-1])
    top_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_values[..., -1:]
    filtered = logits.masked_fill(logits < threshold, float("-inf"))
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_top_p(logits: Tensor, top_p: float) -> Tensor:

    ## Written by gemini 4 neon, got bored
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    sorted_logits, sorted_indices = torch.sort(
        logits, descending=True, dim=-1
    )
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    remove_mask = cumulative_probs > top_p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    filtered_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
    filtered_probs = torch.softmax(filtered_logits, dim=-1)
    sampled_sorted = torch.multinomial(filtered_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled_sorted).squeeze(-1)


def sample_min_p(logits: Tensor, min_p: float) -> Tensor:
    ## Written by gemini 4 neon, got bored
    
    if not 0.0 <= min_p <= 1.0:
        raise ValueError("min_p must be in [0, 1]")

    probs = torch.softmax(logits, dim=-1)
    threshold = min_p * probs.max(dim=-1, keepdim=True).values
    filtered_logits = logits.masked_fill(probs < threshold, float("-inf"))
    filtered_probs = torch.softmax(filtered_logits, dim=-1)
    return torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
