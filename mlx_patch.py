"""Compatibility shim for AirLLM + MLX on Mac Apple Silicon.

Two patches:
1. mlx.nn.Module.update — drop unknown keys (e.g. `bias`) before delegating.
2. MlxModelPersister.load_model — if a layer file is missing, fall back to
   `model.embed_tokens.mlx.npz`. This covers tied embeddings, where the model
   shares lm_head.weight with embed_tokens and AirLLM skips saving lm_head.
"""
import mlx.core as mx
import mlx.nn as nn

_orig_update = nn.Module.update


def _safe_update(self, parameters):
    try:
        return _orig_update(self, parameters)
    except ValueError as e:
        if "does not have parameter named" not in str(e):
            raise
        own_keys = set(self.parameters().keys())

        def _strip(params):
            if not isinstance(params, dict):
                return
            for k in [k for k in params if k not in own_keys]:
                del params[k]
            for v in params.values():
                if isinstance(v, dict):
                    _strip(v)

        _strip(parameters)
        return _orig_update(self, parameters)


nn.Module.update = _safe_update


# Patch the persister after airllm is importable.
def _install_persister_patch():
    from airllm.persist.mlx_model_persister import MlxModelPersister
    from mlx.utils import tree_unflatten
    from pathlib import Path
    from airllm.persist.mlx_model_persister import map_torch_to_mlx
    from contextlib import contextmanager
    import gc

    _orig_load = MlxModelPersister.load_model

    def _load_with_fallback(self, layer_name, path):
        from pathlib import Path as _P
        primary = _P(path) / (layer_name + ".mlx.npz")
        if not primary.exists():
            fallback = _P(path) / "model.embed_tokens.mlx.npz"
            if fallback.exists():
                print(f"[mlx_patch] {layer_name} missing -> using embed_tokens")
                layer_state_dict = mx.load(str(fallback))
                layer_state_dict = map_torch_to_mlx(layer_state_dict)
                layer_state_dict = {k.replace("tok_embeddings", "output"): v for k, v in layer_state_dict.items()}
                weights = tree_unflatten(list(layer_state_dict.items()))
                return weights
        return _orig_load(self, layer_name, path)

    MlxModelPersister.load_model = _load_with_fallback
