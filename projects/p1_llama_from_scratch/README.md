# P1: Llama-style inference from scratch

**Target:** weeks 1-3

Build a small Llama-style decoder-only model in PyTorch and match Hugging Face
output logits for Qwen3-0.6B or Llama-3.2-1B.

Do not implement with `transformers` modeling code. `transformers` is allowed
only as a weight source and numerical reference.

## Week 1 blocks

1. Implement `RMSNorm`.
2. Implement rotary position embeddings.
3. Implement grouped-query attention.
4. Implement the SwiGLU feed-forward block.
5. Load a model configuration and real weights.
6. Implement the preallocated KV cache.
7. Implement greedy and probabilistic sampling.
8. Compare logits token-for-token against Hugging Face.

## Starter layout

- `config.py`: model configuration.
- `modules.py`: RMSNorm, RoPE, attention, and feed-forward blocks.
- `kv_cache.py`: preallocated key/value storage.
- `sampling.py`: greedy, temperature, top-k, top-p, and min-p sampling.
- `model.py`: decoder model and weight-loading entry points.
- `tests/`: correctness checks for each block.

## Reference model

Use Qwen3-0.6B as the first target:

```text
Qwen/Qwen3-0.6B
```

## Accept for later weeks

- Greedy token-for-token match with Hugging Face on at least 20 prompts.
- KV cache faster than repeated full-context forward passes after 200 tokens.
- Sampling unit-tested.

## Commands

```bash
cd projects/p1_llama_from_scratch
python -m pip install torch transformers safetensors pytest
pytest -q
```

## Notes

- The function skeletons and scaffolding were implemented with a Gemini 4 Neon (low) model, and the main functions were implemented by the user.
