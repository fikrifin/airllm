# AirLLM on Mac Apple Silicon

Tested setup AirLLM di iMac (macOS 26.x, Apple Silicon, Python 3.9.6 bawaan sistem).

## Status

| Task | Status | Catatan |
|------|--------|---------|
| Install airllm + mlx | ✅ | venv di `.venv` |
| Model load & generate | ✅ | tested dengan `Qwen/Qwen2.5-0.5B` |
| Generate output end-to-end | ✅ | ada 2 patch yang dibutuhkan (lihat di bawah) |
| HF token untuk model gated | ✅ | lewat `.env` |

## Quick start

```bash
cd /Users/kominfo/Documents/Fikri/Project/airllm
source .venv/bin/activate
python test_setup.py        # cek env valid
python example_run.py       # inference publik model (Qwen2.5-0.5B)
```

Untuk model gated (Llama, Gemma, dll):

```bash
cp .env.example .env
# edit .env, set HF_TOKEN=hf_xxx
python run_gated.py
```

Atau override per-run:

```bash
MODEL_ID=meta-llama/Llama-3.2-1B HF_TOKEN=hf_xxx python run_gated.py
```

## Patches yang aku tambahkan

Library AirLLM v3.0.1 + MLX backend punya dua masalah di Mac:

1. **Unknown keys (`bias`)** — Beberapa model (termasuk Qwen2.5) save `q_proj.bias`/`k_proj.bias`/`v_proj.bias` walaupun config tidak declare `qkv_bias=True`. MLX strict-reject dan crash.
2. **Tied embeddings (`lm_head`)** — Untuk model dengan `tie_word_embeddings=True` (Qwen2, Llama-3.2, dsb), AirLLM skip save file `lm_head.mlx.npz` karena weight-nya = embed_tokens. Saat inference butuh lm_head, load gagal.

Solusinya: `mlx_patch.py` (di folder ini) monkey-patch:

- `mlx.nn.Module.update` → drop keys yang tidak ada di module
- `MlxModelPersister.load_model` → fall back ke `model.embed_tokens.mlx.npz` saat `lm_head` hilang

Cara pakainya: import `mlx_patch` **sebelum** `from airllm import AutoModel`. Script `example_run.py` dan `run_gated.py` sudah handle ini.

## Struktur folder

```
airllm/
├── .env.example          # template HF_TOKEN
├── .gitignore
├── README.md
├── example_run.py        # demo model publik (Qwen2.5-0.5B)
├── run_gated.py          # demo model gated (perlu HF_TOKEN)
├── test_setup.py         # cek env valid
├── mlx_patch.py          # shim untuk 2 bug di AirLLM Mac backend
└── .venv/                # virtualenv (tidak masuk git)
```

## Hardware

- **Tested on:** iMac Apple Silicon (M-series), macOS 26.x
- **Device saat inference:** `mps` (Apple GPU via MLX)
- **Speed:** ~20-22 it/s untuk Qwen2.5-0.5B di model pertama
- **Disk usage:** ~990MB untuk Qwen2.5-0.5B (weight) + ~750MB (split cache per layer)

## Limitasi yang diketahui

- Output dari base model `Qwen2.5-0.5B` agak kacau — model ini bukan instruct-tuned. Untuk percakapan yang lebih masuk akal, coba `meta-llama/Llama-3.2-1B-Instruct` (perlu HF token).
- Python masih 3.9.6 (bawaan macOS). Untuk produksi, lebih ideal pakai Python 3.10/3.11 via Homebrew/pyenv.
- Setiap run pertama kali di model baru akan download + split. Setelah itu cache sudah ada dan run berikutnya jauh lebih cepat.

## Catatan tambahan

- `airllm_airllm_llama_mlx.py` line 264: `generate()` method return string (sudah decode), bukan ids. Jangan pakai `return_dict_in_generate=True`.
- Model yang **bukan** keluarga LLaMA-architecture mungkin butuh backend lain (lihat `airllm_chatglm.py`, `airllm_qwen.py`).
