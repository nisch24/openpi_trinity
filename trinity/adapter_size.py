"""Count the per-robot (trainable) parameters for each LoRA variant.

Builds parameter shapes only via jax.eval_shape, so it needs no GPU and allocates no weights.
Run from the repo root:  uv run python trinity/adapter_size.py
"""
import collections

import flax.nnx as nnx
import jax

from openpi.training import config as _config

CONFIGS = ["pi05_libero", "pi05_libero_lora", "pi05_libero_lora_strict"]

for name in CONFIGS:
    cfg = _config.get_config(name)
    # Parameter shapes for the whole model, without allocating anything.
    state = jax.eval_shape(lambda cfg=cfg: nnx.state(cfg.model.create(jax.random.key(0))))
    trainable = state.filter(cfg.trainable_filter)
    total = sum(x.size for x in jax.tree.leaves(state))
    n = sum(x.size for x in jax.tree.leaves(trainable))

    # Group trainable params by top-level module so it's clear what each variant trains.
    groups = collections.Counter()
    for path, leaf in jax.tree_util.tree_flatten_with_path(trainable)[0]:
        keys = [str(getattr(k, "key", k)) for k in path]
        top = "/".join(keys[:2]) if keys[0] == "PaliGemma" else keys[0]
        groups[top] += leaf.size

    print(f"\n{name}: {n / 1e6:,.1f}M of {total / 1e9:.2f}B params trained per robot "
          f"(~{n * 2 / 1e9:.2f} GB in bf16)")
    for top, size in groups.most_common():
        print(f"    {top:<28} {size / 1e6:>9,.1f}M")
