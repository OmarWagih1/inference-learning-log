## COMPLETELY written by 4 neon, just wanted to try things out.


import argparse
import time

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from kv_cache import KVCache
from model import LlamaStyleModel
from sampling import (
    sample_greedy,
    sample_min_p,
    sample_temperature,
    sample_top_k,
    sample_top_p,
)


def pick_next_token(
    logits: torch.Tensor,
    strategy: str,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
) -> torch.Tensor:
    if strategy == "greedy":
        return sample_greedy(logits)
    if strategy == "temperature":
        return sample_temperature(logits, temperature=temperature)
    if strategy == "top_k":
        return sample_top_k(logits / temperature, top_k=top_k)
    if strategy == "top_p":
        return sample_top_p(logits / temperature, top_p=top_p)
    if strategy == "min_p":
        return sample_min_p(logits / temperature, min_p=min_p)
    raise ValueError(f"Unknown sampling strategy: {strategy}")


def generate_text(
    model: LlamaStyleModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    strategy: str = "greedy",
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    min_p: float = 0.05,
) -> str:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]
    max_sequence_length = prompt_length + max_new_tokens

    kv_caches = [
        KVCache.create(
            batch_size=1,
            max_sequence_length=max_sequence_length,
            config=model.config,
            device=device,
            dtype=dtype,
        )
        for _ in range(model.config.num_hidden_layers)
    ]

    generated_ids: list[int] = []
    start_time = time.perf_counter()

    with torch.no_grad():
        # Prefill phase: process the entire prompt at once.
        logits = model(input_ids, kv_caches=kv_caches)
        next_token = pick_next_token(
            logits[:, -1, :],
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
        )

        # Decode phase: generate one token at a time using the KV cache.
        for _ in range(max_new_tokens):
            token_id = int(next_token.item())
            if token_id == tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)
            print(tokenizer.decode([token_id]), end="", flush=True)

            next_input = next_token.unsqueeze(0)
            logits = model(next_input, kv_caches=kv_caches)
            next_token = pick_next_token(
                logits[:, -1, :],
                strategy=strategy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
            )

    elapsed = time.perf_counter() - start_time
    tokens_per_second = len(generated_ids) / max(elapsed, 1e-6)
    print(f"\n\n[{len(generated_ids)} tokens in {elapsed:.2f}s — {tokens_per_second:.1f} tok/s]")
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run text generation with your from-scratch Llama model."
    )
    parser.add_argument(
        "--model-id",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="HuggingFace repo ID of a Llama-architecture checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain what a KV cache is in two sentences:",
        help="Prompt to complete.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--strategy",
        choices=["greedy", "temperature", "top_k", "top_p", "min_p"],
        default="greedy",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--min-p", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Downloading / locating {args.model_id}...")
    model_directory = snapshot_download(
        repo_id=args.model_id,
        allow_patterns=["config.json", "*.safetensors", "tokenizer*"],
    )

    print("Loading tokenizer and model weights...")
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = LlamaStyleModel.from_safetensors(model_directory)
    model.eval()

    print(f"\nPrompt: {args.prompt}\n---")
    generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
    )


if __name__ == "__main__":
    main()
