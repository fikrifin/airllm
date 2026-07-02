"""Run a model that needs a Hugging Face token (gated or private)."""
import os
import mlx_patch  # noqa: F401
from airllm import AutoModel
import torch
import mlx.core as mx

# Load .env (no external dep; simple parser)
def _load_env(path: str = ".env"):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env()

MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
HF_TOKEN = os.environ.get("HF_TOKEN")

if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Model: {MODEL_ID}")
print(f"Device: {device}")
print(f"Token: {'provided' if HF_TOKEN else 'NONE (gated models will fail)'}")

model = AutoModel.from_pretrained(MODEL_ID, hf_token=HF_TOKEN)

input_tokens = model.tokenizer(
    ["Halo, perkenalkan dirimu secara singkat."],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

raw_input_ids = input_tokens['input_ids']
if device == "cuda":
    model_inputs = raw_input_ids.cuda()
elif device == "mps":
    model_inputs = mx.array(raw_input_ids.numpy())
else:
    model_inputs = raw_input_ids

print("Generating...")
out = model.generate(model_inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)
print("---OUTPUT---")
print(out)
