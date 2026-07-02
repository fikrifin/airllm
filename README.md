# AirLLM on macOS Apple Silicon

Setup tested for running [AirLLM](https://github.com/lyogavin/airllm) on macOS with Apple Silicon. This project documents the working environment, the patches required to make inference succeed on the MLX backend, and example scripts.

## Status

| Task | Status | Notes |
|------|--------|-------|
| Install airllm + mlx | ✅ | venv at `.venv` |
| Model load & generate | ✅ | tested with `Qwen/Qwen2.5-0.5B` |
| Generate output end-to-end | ✅ | two upstream bugs patched (see below) |
| HF token for gated models | ✅ | via `.env` |

## Quick start

```bash
cd /path/to/this/project
python3 -m venv .venv
source .venv/bin/activate
pip install airllm mlx
python test_setup.py        # verify environment
python example_run.py       # inference with a public model (Qwen2.5-0.5B)
```

For gated models (Llama, Gemma, etc.):

```bash
cp .env.example .env
# edit .env, set HF_TOKEN=***
python run_gated.py
```

Or override per-run:

```bash
MODEL_ID=meta-llama/Llama-3.2-1B HF_TOKEN=*** python run_gated.py
```

## Patches required

AirLLM v3.0.1 + MLX backend has two issues on macOS:

1. **Unknown keys (`bias`)** — Some models (including Qwen2.5) save `q_proj.bias` / `k_proj.bias` / `v_proj.bias` even when `qkv_bias` is not set in `config.json`. MLX strictly rejects unknown parameters and crashes.
2. **Tied embeddings (`lm_head`)** — Models with `tie_word_embeddings=True` (Qwen2, Llama-3.2, etc.) cause AirLLM to skip saving the `lm_head.mlx.npz` file because the weight is shared with `embed_tokens`. Loading `lm_head` later fails.

The fix: `mlx_patch.py` (in this repo) monkey-patches:

- `mlx.nn.Module.update` to drop keys the layer doesn't know about
- `MlxModelPersister.load_model` to fall back to `model.embed_tokens.mlx.npz` when `lm_head` is missing

Usage: import `mlx_patch` **before** `from airllm import AutoModel`. Both `example_run.py` and `run_gated.py` handle this.

## Project structure

```
.
├── .env.example          # HF_TOKEN template
├── .gitignore
├── README.md
├── example_run.py        # public model demo (Qwen2.5-0.5B)
├── run_gated.py          # gated model demo (requires HF_TOKEN)
├── test_setup.py         # environment verifier
├── mlx_patch.py          # upstream compatibility shim
└── .venv/                # virtualenv (not tracked in git)
```

## Tested environment

- **Platform:** macOS Apple Silicon (M-series)
- **Inference device:** `mps` (Apple GPU via MLX)
- **Speed:** ~20-22 it/s for Qwen2.5-0.5B on initial run
- **Disk usage:** ~990 MB for Qwen2.5-0.5B (weights) + ~750 MB (per-layer split cache)

The setup also works on Linux with NVIDIA GPUs (uses `cuda` device) and on CPU, but those paths have not been verified in this project.

## Known limitations

- Base models like `Qwen2.5-0.5B` produce low-quality output — they are not instruction-tuned. For coherent conversation, use instruct variants like `meta-llama/Llama-3.2-1B-Instruct` (requires HF token) or `Qwen/Qwen2.5-1.5B-Instruct`.
- Python 3.9.6 (the macOS system Python) is what this project was tested with. Python 3.10 or 3.11 is recommended for production use; install via Homebrew or pyenv.
- The first run on a new model downloads weights and writes per-layer split files. Subsequent runs reuse the cache and are much faster.
- The patches in `mlx_patch.py` are workarounds for upstream bugs, not a proper fix. Contributions back to the AirLLM project would be welcome.

## Additional notes

- `airllm_llama_mlx.py` line 264: the `generate()` method returns a decoded string, not token ids. Do not pass `return_dict_in_generate=True`.
- Non-LLaMA-architecture models may require a different backend. See `airllm_chatglm.py`, `airllm_qwen.py` in the installed AirLLM package.
