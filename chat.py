"""Interactive chat: ask anything from the terminal, get a reply from the model."""
import argparse
import os
import sys

import mlx_patch  # noqa: F401  (must be first)
from airllm import AutoModel
import torch
import mlx.core as mx


def _to_device(raw, device):
    if device == "cuda":
        return raw.cuda()
    if device == "mps":
        return mx.array(raw.numpy())
    return raw


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--system", default="You are a helpful assistant. Answer briefly in Indonesian.")
    parser.add_argument("--prompt", default=None, help="one-shot prompt (otherwise REPL)")
    args = parser.parse_args()

    _load_env()
    hf_token = ***"HF_TOKEN")

    device = pick_device()
    print(f"[setup] device={device} model={args.model}", flush=True)

    model = AutoModel.from_pretrained(args.model, hf_token=hf_token)

    template = "{system}\nUser: {user}\nAssistant:"
    system = args.system

    def _ask(user_text: str) -> str:
        prompt = template.format(system=system, user=user_text)
        toks = model.tokenizer(
            [prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=512,
            padding=False,
        )
        out = model.generate(
            _to_device(toks["input_ids"], device),
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
        # Strip the prompt echo from the output if present
        if isinstance(out, str) and prompt in out:
            return out.split(prompt, 1)[1].strip()
        return out if isinstance(out, str) else str(out)

    if args.prompt is not None:
        print(_ask(args.prompt))
        return

    print("[chat] type a prompt and press Enter (Ctrl-D or 'exit' to quit)\n")
    try:
        while True:
            user_text = input("> ").strip()
            if not user_text or user_text.lower() in {"exit", "quit"}:
                break
            print(_ask(user_text), flush=True)
            print()
    except (EOFError, KeyboardInterrupt):
        print("\n[chat] bye")


if __name__ == "__main__":
    sys.exit(main())
