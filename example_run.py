import mlx_patch  # noqa: F401  (must be first to monkey-patch mlx)
from airllm import AutoModel
import torch
import mlx.core as mx

mlx_patch._install_persister_patch()

MAX_LENGTH = 128
MODEL_ID = "Qwen/Qwen2.5-0.5B"

if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Loading model: {MODEL_ID}")
print(f"Using device: {device}")
model = AutoModel.from_pretrained(MODEL_ID)

input_text = ["Halo, jelaskan secara singkat apa itu AirLLM."]
input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False,
)

raw_input_ids = input_tokens['input_ids']

if device == "cuda":
    model_inputs = raw_input_ids.cuda()
elif device == "mps":
    model_inputs = mx.array(raw_input_ids.numpy())
else:
    model_inputs = raw_input_ids

generation_output = model.generate(
    model_inputs,
    max_new_tokens=32,
    use_cache=True,
)

print("---OUTPUT---")
print(generation_output)
