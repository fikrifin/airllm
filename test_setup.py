"""Verify the AirLLM environment on this machine."""
import sys

print("Python:", sys.version.split()[0])

try:
    import airllm
    print("airllm: OK")
except Exception as e:
    print(f"airllm: FAILED ({e})")
    sys.exit(1)

try:
    import mlx
    print(f"mlx: OK ({getattr(mlx, '__version__', 'unknown')})")
except Exception as e:
    print(f"mlx: FAILED ({e})")
    sys.exit(1)

try:
    import torch
    print(f"torch: OK ({torch.__version__})")
    if torch.cuda.is_available():
        print(f"  cuda: available ({torch.cuda.get_device_name(0)})")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("  mps: available (Apple Silicon GPU)")
    else:
        print("  device: cpu only")
except Exception as e:
    print(f"torch: FAILED ({e})")

# Check our compatibility shim
try:
    import mlx_patch
    mlx_patch._install_persister_patch()
    print("mlx_patch: installed (handles tied embeddings + bias keys)")
except Exception as e:
    print(f"mlx_patch: FAILED ({e})")

# Optional: HF token
import os
from pathlib import Path
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    print(f".env: found ({env_path})")
else:
    print(f".env: missing (copy from .env.example to enable gated models)")

print("\nSetup OK. Try: python example_run.py")
